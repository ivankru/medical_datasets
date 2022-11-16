import pydicom
#from pydicom.data import get_testdata_file
from pydicom.pixel_data_handlers.util import apply_modality_lut
#import zstandard as zstd
import os


def dcm_to_dict(file_path):
    # zd = zstd.ZstdDecompressor()
    # #compressed = open(file_path, "rb").read()
    # try:
    #     if file_path[-3:] == "zst":
    #         compressed = open(file_path, "rb").read()
    #         data = zd.decompress(compressed)
    #         ds = pydicom.dcmread(BytesIO(data))
    #    elif file_path[-3:] == "dcm":
    ds = pydicom.dcmread(file_path)
 
    # except:
    #     if verbose:
    #         print("error in reading", file_path)
    #     return None 

    sex = ds.get('PatientSex', None)
    weight = ds.get('PatientWeight', None)
    weight = float(weight)
    height = ds.get('PatientSize', None)
    height = float(height)
    # patient_id = ds.get('PatientID', None)
    # study_id = ds.get('StudyID', None)
    # age = ds.get('PatientAge', None)
    # age = int(age[:-1])
    #modality = ds.get('Modality', None)
    #SOPInstanceUID = ds.get("SOPInstanceUID", None)

    return_dict = {"sex":sex, "weight":weight, "height":length}
    return return_dict


if __name__ == "__main__":
    raw_data = "/ayb/vol1/kudin/dev/datasets/raw_data_part_1/"
    for study in os.listdir(raw_data):
        study_folder = os.path.join(raw_data, study)
        serie = os.listdir(study_folder)[1]
        serie_folder = os.path.join(study_folder, serie)
        dcm_file = os.listdir(serie_folder)[0]
        dcm_file_path = os.path.join(serie_folder, dcm_file)
        dcm_dict = dcm_to_dict(dcm_file_path)
