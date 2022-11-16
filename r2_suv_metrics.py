from re import I
import nibabel as nb
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import os
from skimage.transform import resize
import cc3d
#import sklearn as sk  
#import matplotlib.lines as mlines
from nilearn.image import resample_img
from sklearn.metrics import r2_score
#import json
from scipy.ndimage import gaussian_filter
import scipy 


def suv_peak_funct(image_pet_original, image_pet, labels_out, i, pixel_radius=3.6, pixel_radius_z=1.89):
    tumor_coords = np.argwhere(labels_out == i)
    max_number = np.argmax(image_pet_original[tumor_coords[:,0], tumor_coords[:,1], tumor_coords[:,1]])
    coord_max = tumor_coords[max_number,:]
    r = round(pixel_radius - 0.5) #minus halve of the pixel
    rz = round(pixel_radius_z - 0.5)
    x, y, z = coord_max[0], coord_max[1], coord_max[2]
    rect_in_1cm = image_pet[x-r:x+r+1, y-r:y+r+1, z-rz:z+rz+1]
    suv_sum = np.sum(rect_in_1cm)
    im_szx, im_szy, im_szz  = image_pet_original.shape
    im_szx, im_szy, im_szz  =  im_szx - 1, im_szy - 1, im_szz - 1
    suv_sum += image_pet[min(im_szx, x+r+2),y,z] + image_pet[max(x-r-1,0),y,z] + \
               image_pet[x, min(im_szy, y+r+2),z] + image_pet[x,max(y-r-1,0),z] + \
               image_pet[x,y,min(z+rz+2, im_szz)] + image_pet[x,y,max(z-rz-1,0)]
    rect_size = np.product(rect_in_1cm.shape)
    suv_peak = suv_sum / (rect_size + 6)
    return suv_peak



def read_pet_reconstruct(pet_reconstr_path, study_folder, new_size=None):
    pet_list = []
    study_path = os.path.join(pet_reconstr_path, study_folder) 
    for index in range(len(os.listdir(study_path))):     
        result_path = os.path.join(study_path, str(index).zfill(6) + ".npy")
        pet_net = np.load(result_path)
        pet_net[pet_net < 0] = 0
        if new_size is not None:
            pet_net = resize(pet_net, (new_size, new_size))
        pet_list.append(pet_net)
    pet_3d = np.stack(pet_list, axis=2)
    return pet_3d
    #nifti_file = nb.Nifti1Image(pet_3d, affine)


def read_pet_reconstruct_kudin(pet_reconstr_path, study_folder, new_size=None):
    pet_list = []
    study_path = os.path.join(pet_reconstr_path, study_folder) 
    for index in range(len(os.listdir(study_path))):     
        result_path = os.path.join(study_path, str(index).zfill(6) + ".npz")
        pet_net = np.load(result_path, allow_pickle=True)["image"]
        pet_net[pet_net < 0] = 0
        if new_size is not None:
            pet_net = resize(pet_net, (new_size, new_size))
        pet_list.append(pet_net)
    pet_3d = np.stack(pet_list, axis=2)
    return pet_3d


def read_pet_original(pet_original_path, study_folder, new_size=None):
    study_path = os.path.join(pet_original_path, study_folder)
    pet90_list = []
    weight_list = []
    for index in range(len(os.listdir(study_path))):
        index_path = os.path.join(study_path, str(index).zfill(6))
        pet_90_path = os.path.join(index_path, "original_in_suv.npz")
        pet_90 = np.load(pet_90_path, allow_pickle=True)["image"]
        if new_size is not None:
            pet_90 = resize(pet_90, (new_size, new_size))
        pet90_list.append(pet_90)
        #metadata_path = os.path.join(index_path, "metadata.json")
        # with open(metadata_path) as f:
        #     patient_weight = float(json.load(f)["individual"]["original"]["PatientWeight"])
        #     weight_list.append(patient_weight)
    pet_3d = np.stack(pet90_list, axis=2)
    return pet_3d


def read_pet_time(pet_original_path, study_folder, new_size=None, sec=90):
    study_path = os.path.join(pet_original_path, study_folder)
    pet_list = []
    file_dict = {30:"30_in_suv.npz", 60:"60_in_suv.npz", 90:"original_in_suv.npz"}
    file_name = file_dict[sec]
    for index in range(len(os.listdir(study_path))):
        index_path = os.path.join(study_path, str(index).zfill(6))
        pet_path = os.path.join(index_path, file_name)
        pet = np.load(pet_path, allow_pickle=True)["image"]
        if new_size is not None:
            pet = resize(pet, (new_size, new_size))
        pet_list.append(pet)
    pet_3d = np.stack(pet_list, axis=2)
    return pet_3d


def gauss_convol(pet_original_path, study_folder, sec, new_size=None):
    study_path = os.path.join(pet_original_path, study_folder)
    pet_list = []
    file_dict = {30:"30_in_suv.npz", 60:"60_in_suv.npz", 90:"original_in_suv.npz"}
    file_name = file_dict[sec]
    for index in range(len(os.listdir(study_path))):
        index_path = os.path.join(study_path, str(index).zfill(6))
        pet_path = os.path.join(index_path, file_name)
        pet = np.load(pet_path, allow_pickle=True)["image"]
        if sec == 60:
            pet = gaussian_filter(pet, sigma=0.65) 
        elif sec == 30:
            pet = gaussian_filter(pet, sigma=1.1)
        else:
            return None
        if new_size is not None:
            pet = resize(pet, (new_size, new_size))
        pet_list.append(pet)
    pet_3d = np.stack(pet_list, axis=2)
    return pet_3d


#tumors suv max, mean
def suv_data(pet_original_path, output_path, pet_reconstr_path, min_mean_suv=0.5, min_turmor_size=5):
    print(pet_reconstr_path)
    suv_list = []
    for i, study in enumerate(os.listdir(pet_original_path)):
        # if i == 5:
        #     break
        segment_path = join(output_path, study + ".nii.gz")
        #pet_path = join(input_path, study + "_0000.nii.gz")
        nii_pet_segm = nb.load(segment_path)
        new_size = 256
        transform_coeff = nii_pet_segm.shape[0] / new_size
        affine_matrix = np.copy(nii_pet_segm.affine)
        affine_matrix[:2,:3] = transform_coeff * affine_matrix[:2,:3]
        nii_pet_segm = resample_img(nii_pet_segm, target_shape = [new_size, new_size, nii_pet_segm.shape[-1]], target_affine=affine_matrix, interpolation='nearest')
        #image_pet_original = np.array(nb.load(pet_path).dataobj).transpose((1,0,2))

        #image_segm = np.array(nii_pet_segm.dataobj)#[:,:,slice].transpose()
        #image_pet_reconstruct = read_pet_time(pet_original_path, study, sec=30)
        #image_pet_reconstruct = gauss_convol(pet_original_path, study, sec=60)
        image_pet_reconstruct = read_pet_reconstruct_kudin(pet_reconstr_path, study)
        image_pet_original = read_pet_original(pet_original_path, study)

        labels_in = np.round(np.array(nii_pet_segm.dataobj))
        connectivity = 6 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
        labels_out, n_blobs = cc3d.connected_components(labels_in, connectivity=connectivity, delta=3, return_N=True)
        stats = cc3d.statistics(labels_out)
        for i in range(1, n_blobs + 1):
            # if stats["voxel_counts"][i] < min_turmor_size:
            #     continue
            delta_x = stats["bounding_boxes"][i][0].stop - stats["bounding_boxes"][i][0].start          
            delta_y = stats["bounding_boxes"][i][1].stop - stats["bounding_boxes"][i][1].start 
            delta_z = stats["bounding_boxes"][i][2].stop - stats["bounding_boxes"][i][2].start 
            if delta_x < 3 and delta_y < 3 and delta_z < 2:
                continue
            tumor_pixel_original = image_pet_original[labels_out == i]
            suv_max = tumor_pixel_original.max()
            suv_mean = tumor_pixel_original.mean()
            if suv_mean < min_mean_suv:
                continue
            suv_peak = suv_peak_funct(image_pet_original, image_pet_original, labels_out, i)
            suv_dict_original = {"mean":suv_mean, "max":suv_max, "peak":suv_peak}
            tumor_pixel_restored = image_pet_reconstruct[labels_out == i]
            suv_max = tumor_pixel_restored.max()
            suv_mean = tumor_pixel_restored.mean()
            suv_peak = suv_peak_funct(image_pet_original, image_pet_reconstruct, labels_out, i)
            suv_dict_reconstruct = {"mean":suv_mean, "max":suv_max, "peak":suv_peak}
            suv_list.append({"original":suv_dict_original, "reconstruct":suv_dict_reconstruct})
    return suv_list


def suv_statistics(suv_list):
    coeff_suv = 1 / 1.3
    suv_full_statistics = dict()
    for mode in ["mean", "peak", "max"]:
        suv2 = np.array([x["original"][mode] for x in suv_list])
        suv1 = np.array([x["reconstruct"][mode] for x in suv_list])
        suv1 *= coeff_suv
        suv2 *= coeff_suv
        suv = np.vstack([suv1, suv2]).T
        r2 = r2_score(suv1, suv2)
        diffs = np.diff(suv, axis=1)
        # Average difference (aka the bias)
        bias = np.mean(diffs)
        bias_median = np.median(diffs)
        iqr = scipy.stats.iqr(diffs.flatten())
        # Sample standard deviation
        sd = np.std(diffs, ddof=1)  
        stat_dict = {"r2":1 - r2, "bias":bias, "bias_median":bias_median, "std":sd, "iqr":iqr}
        suv_full_statistics.update({mode:stat_dict})
    return suv_full_statistics


if __name__ == "__main__":
    suv_list = []
    for val_test in ["val", "test"]:
        if val_test == "test":
            output_path = "/ayb/vol1/kruzhilov/pet_test_denoising/nii_result/"
        elif val_test == "val":
            output_path = "/ayb/vol1/kruzhilov/pet_test_denoising/nii_val_result/"
        #pet_reconstr_path = "/ayb/vol1/luka/inference/cycleGAN_60sec/cycleGAN_delta_alpha0_unpaired"
        #pet_reconstr_path = "/ayb/vol1/luka/inference/cycleGAN_30sec/cycleGAN_delta_alpha0"
        pet_reconstr_path = "/ayb/vol1/luka/inference/pix2pix_resnet_30sec/resnet_gan_newdelta_alpha270_" + val_test 
        pet_original_path = "/ayb/vol1/kruzhilov/datasets/dataset_v_5/" + val_test
        suv_list.append(suv_data(pet_original_path, output_path, pet_reconstr_path))

    suv_list = suv_list[0] #+ suv_list[1]
    suv_stat = suv_statistics(suv_list)
    print("SUV mean")
    string = "iR2={0:.5f}, bias={1:.4f}, median bias={2:.4f}, std={3:.4f}, iqr={4:.4f}"
    format_params = [suv_stat["mean"]["r2"], suv_stat["mean"]["bias"], suv_stat["mean"]["bias_median"], suv_stat["mean"]["std"], suv_stat["mean"]["iqr"]]
    print(string.format(*format_params))
    print("SUV peak")
    format_params = [suv_stat["peak"]["r2"], suv_stat["peak"]["bias"], suv_stat["peak"]["bias_median"], suv_stat["peak"]["std"], suv_stat["peak"]["iqr"]]
    print(string.format(*format_params))
    print("SUV max")
    format_params = [suv_stat["max"]["r2"], suv_stat["max"]["bias"], suv_stat["max"]["bias_median"], suv_stat["max"]["std"], suv_stat["max"]["iqr"]]
    print(string.format(*format_params))
