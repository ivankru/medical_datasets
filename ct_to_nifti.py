#%%
# constants for the Hubert normalization https://radiopaedia.org/articles/windowing-ct
#"CTAC 3.75 Thick" и "WB 3D MAC" coinsedes, 'CTAC 3.75 mm' - Балашиха
from ast import Break
import os
from shutil import rmtree
import json
import warnings
#from tqdm import tqdm
import copy
import numpy as np

import pydicom
#from pydicom.data import get_testdata_file
#from pydicom.pixel_data_handlers.util import apply_modality_lut
import SimpleITK as sitk
from nilearn.image import resample_img
import nibabel as nb
#import zstandard as zstd
from io import BytesIO
import matplotlib.pyplot as plt
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset

#from create_dataset import normalization_window #, lung_files
#from utils.pdf_extraction import extract_lungs_description, extract_lungs_from_txt
#from utils.pi_files import rescale_image, interpolate_image, find_le, find_ge


def dcm_to_dict(file_path, verbose=True):
    #zd = zstd.ZstdDecompressor()
    #compressed = open(file_path, "rb").read()
    try:
        if file_path[-3:] == "zst":
            compressed = open(file_path, "rb").read()
            data = zd.decompress(compressed)
            ds = pydicom.dcmread(BytesIO(data))
        elif file_path[-3:] == "dcm":
            ds = pydicom.dcmread(file_path)
        else:
            ds = pydicom.dcmread(file_path)
    except:
        if verbose:
            print("error in reading", file_path)
        return None 

    arr = ds.get('pixel_array', None)#ds.pixel_array
    #print(ds["SeriesDescription"], ds["PixelSpacing"])
    if arr is None:
        if verbose:
            print("No image in", file_path)
        return None     

    slice_loc = ds.get('SliceLocation', None)
    sex = ds.get('PatientSex', None)
    weight = ds.get('PatientWeight', None)
    weight = float(weight)
    length = ds.get('PatientSize', None)
    # length = float(length)
    patient_id = ds.get('PatientID', None)
    study_id = ds.get('StudyID', None)
    age = ds.get('PatientAge', None)
    age = int(age[:-1])
    modality = ds.get('Modality', None)
    SOPInstanceUID = ds.get("SOPInstanceUID", None)
    if modality is None:
        print("No modality in:", file_path)
        return None

    series_descrip = ds.SeriesDescription.upper()
    if slice_loc is None:
        print("No slice in:", file_path)
        return None
    else:
        slice_loc = float(slice_loc)

    return_dict = {"slice":slice_loc, "ct":arr, "sex":sex, "weight":weight, \
                    "length":length, "id":patient_id, "age":age, "position":None, \
                    "file":file_path, "modality":modality, "series":series_descrip,
                    "study_id":study_id, "SOPInstanceUID":SOPInstanceUID}
    return return_dict
               

def load_rescale_dcm(study_serie_folder, new_size=None):
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(study_serie_folder)
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3D = series_reader.Execute()
    orig_nii = sitk.Cast(image3D, sitk.sitkFloat32)
    temp_file_path = '/ayb/vol1/kruzhilov/tmp_dir/temp_nii.nii'
    sitk.WriteImage(orig_nii, temp_file_path)
    orig_nii = nb.load(temp_file_path)
    if new_size is not None:
        transform_coeff = orig_nii.shape[0] / new_size
        affine_matrix = np.copy(orig_nii.affine)
        affine_matrix[:2,:3] = transform_coeff * affine_matrix[:2,:3]
        downsampled_nii = resample_img(orig_nii, target_shape = [new_size, new_size, orig_nii.shape[-1]], target_affine=affine_matrix, interpolation='nearest')
    else:
        downsampled_nii = orig_nii
    os.remove(temp_file_path)
    return downsampled_nii


def convert_pet(pet, suv_factor):
    # function for conversion of PET values to SUV (should work on Siemens PET/CT)
    affine = pet.affine
    pet_data = pet.get_fdata()
    pet_suv_data = (pet_data*suv_factor).astype(np.float32)
    pet_suv = nb.Nifti1Image(pet_suv_data, affine)
    return pet_suv


def load_rescale_dcm_pet(study_serie_folder, new_size=None):
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(study_serie_folder)
    suv_factor_kudin = calculate_suv_factor_kudin(series_file_names[0])
    suv_factor = calculate_suv_factor(series_file_names[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3D = series_reader.Execute()
    orig_nii = sitk.Cast(image3D, sitk.sitkFloat32)
    sitk.WriteImage(orig_nii, 'temp.nii')
    orig_nii = nb.load('temp.nii')
    orig_nii = convert_pet(orig_nii, suv_factor)
    if new_size is not None:
        transform_coeff = orig_nii.shape[0] / new_size
        affine_matrix = np.copy(orig_nii.affine)
        affine_matrix[:2,:3] = transform_coeff * affine_matrix[:2,:3]
        downsampled_nii = resample_img(orig_nii, target_shape = [new_size, new_size, orig_nii.shape[-1]], target_affine=affine_matrix, interpolation='nearest')
    else:
        downsampled_nii = orig_nii
    #os.remove('temp.nii')
    return downsampled_nii



def ct_folder(study_pet_folder):
    ctac375_serie = False

    for serie in os.listdir(study_folder):
        if serie == "metadata.json":
            continue
     
        study_serie_folder = os.path.join(study_folder, serie)
        for dcm_file in os.listdir(study_serie_folder):
            dcm_path = os.path.join(study_serie_folder, dcm_file)
            dcm_dict = dcm_to_dict(dcm_path)
            serie_description = dcm_dict["series"]
            break

        if "CTAC 3.75 MM" in serie_description and dcm_dict["modality"] == "CT":
            ctac375_serie = True
            return study_serie_folder


def pet_serie_func(pet_dataset, study):
    study_pet_metadata = os.path.join(pet_dataset, study)
    study_pet_metadata = os.path.join(study_pet_metadata, "000000")
    study_pet_metadata = os.path.join(study_pet_metadata, "metadata.json")
    with open(study_pet_metadata, 'r') as outfile:
        json_dict = json.load(outfile)
    pet_serie = json_dict["individual"]["original"]["series_uid"]
    return pet_serie


def conv_time(time_str):
    # function for time conversion in DICOM tag
    return (float(time_str[:2]) * 3600 + float(time_str[2:4]) * 60 + float(time_str[4:13]))


def calculate_suv_factor(dcm_path):
    # reads a PET dicom file and calculates the SUV conversion factor
    ds = pydicom.dcmread(str(dcm_path))
    total_dose = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
    start_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
    half_life = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
    acq_time = ds.AcquisitionTime
    weight = ds.PatientWeight
    time_diff = conv_time(acq_time) - conv_time(start_time)
    act_dose = total_dose * 0.5 ** (time_diff / half_life)
    suv_factor = 1000 * weight / act_dose
    return suv_factor


def calculate_suv_factor_kudin(dcm_path):
    # reads a PET dicom file and calculates the SUV conversion factor
    ds = pydicom.dcmread(str(dcm_path))
    radiopharmaceutical_start_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
    radionuclide_half_life = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
    radionuclide_total_dose = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
    patient_weight = float(ds.PatientWeight)
    acquisition_time = float(ds.AcquisitionTime)
    radiopharmaceutical_start_time = float(radiopharmaceutical_start_time)
    delta_time = (acquisition_time - radiopharmaceutical_start_time) / 100.0
    half_life = radionuclide_half_life / 60.0
    corrected_dose = radionuclide_total_dose * np.exp(-delta_time * np.log(2.0) / half_life)
    suv_factor = patient_weight * 1000.0 / corrected_dose
    return suv_factor


if __name__ == "__main__":
    raw_data = "/ayb/vol1/kudin/dev/datasets/raw_data_part_1/"
    nii_save_path = "/ayb/vol1/kruzhilov/nnUnet/nnUNet_preprocessed/Task001_TCIA/imagesTS"
    pet_dataset_path = "/ayb/vol1/kruzhilov/datasets/dataset_v_5/val"
    # study = "0.8.290.0.5618086.8.0.0.3576454119.76021.9253413059.52961"    
    # study = "1.0.329.466197.3.333.0.6234929077.287.9246067012.360"
    resolution = 256 

    for study in os.listdir(pet_dataset_path):
        study_folder = os.path.join(raw_data, study)

        pet_serie = pet_serie_func(pet_dataset_path, study)
        study_ct_folder = os.path.join(pet_dataset_path, study)
        study_serie_folder = ct_folder(study_ct_folder)
    
        #CT data from a serie into one nii file
        ct_nii = load_rescale_dcm(study_serie_folder, new_size=400)

        #PET data from a serie into one nii file
        study_serie_folder = os.path.join(study_folder, pet_serie)
        pet_nii = load_rescale_dcm_pet(study_serie_folder, new_size=400)
        
        #image = ct_nii.dataobj[:,:,200].transpose()
        #plt.imshow(image)
        
        file_name = study + "_0001.nii.gz"
        save_nii_path = os.path.join(nii_save_path, file_name)
        nb.save(ct_nii, save_nii_path)
        file_name = study + "_0000.nii.gz"
        save_nii_path = os.path.join(nii_save_path, file_name)
        nb.save(pet_nii, save_nii_path)
        #series_type="RECON 2: CT LUNG") CTAC 3.75 MM, WB 3D MAC
        
                



# %%
