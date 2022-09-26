import nibabel as nb
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import os
from skimage.transform import resize
import cc3d
import sklearn as sk  
import matplotlib.lines as mlines
from sklearn.metrics import r2_score
import json



def read_pet_reconstruct(pet_reconstr_path, study_folder, new_size=400):
    pet_list = []
    study_path = os.path.join(pet_reconstr_path, study_folder) 
    for index in range(len(os.listdir(study_path))):     
        result_path = os.path.join(study_path, str(index).zfill(6) + ".npy")
        pet_net = np.load(result_path)
        pet_net[pet_net < 0] = 0
        pet_net = resize(pet_net, (new_size, new_size))
        pet_list.append(pet_net)
    pet_3d = np.stack(pet_list, axis=2)
    return pet_3d
    #nifti_file = nb.Nifti1Image(pet_3d, affine)


def read_pet_reconstruct_kudin(pet_reconstr_path, study_folder, new_size=400):
    pet_list = []
    study_path = os.path.join(pet_reconstr_path, study_folder) 
    for index in range(len(os.listdir(study_path))):     
        result_path = os.path.join(study_path, str(index).zfill(6) + ".npz")
        pet_net = np.load(result_path, allow_pickle=True)["image"]
        pet_net[pet_net < 0] = 0
        pet_net = resize(pet_net, (new_size, new_size))
        pet_list.append(pet_net)
    pet_3d = np.stack(pet_list, axis=2)
    return pet_3d


def read_pet_original(pet_original_path, study_folder, new_size=400):
    study_path = os.path.join(pet_original_path, study_folder)
    pet90_list = []
    for index in range(len(os.listdir(study_path))):
        index_path = os.path.join(study_path, str(index).zfill(6))
        pet_90_path = os.path.join(index_path, "original_in_suv.npz")
        pet_90 = np.load(pet_90_path, allow_pickle=True)["image"]
        pet_90 = resize(pet_90, (new_size, new_size))
        pet90_list.append(pet_90)
        metadata_path = os.path.join(index_path, "metadata.json")
        with open(metadata_path) as f:
            patien_weight = json.load(f)["original"]["PatientWeight"]
    pet_3d = np.stack(pet90_list, axis=2)
    return pet_3d


#tumors suv max, mean
def suv_data(pet_original_path, output_path, pet_reconstr_path):
    suv_list = []
    for i, study in enumerate(os.listdir(pet_original_path)):
        # if i == 10:
        #     break
        segment_path = join(output_path, study + ".nii.gz")
        #pet_path = join(input_path, study + "_0000.nii.gz")
        nii_pet_segm = nb.load(segment_path)
        #image_pet_original = np.array(nb.load(pet_path).dataobj).transpose((1,0,2))

        #image_segm = np.array(nii_pet_segm.dataobj)#[:,:,slice].transpose()
        image_pet_reconstruct = read_pet_reconstruct(pet_reconstr_path, study)
        image_pet_original = read_pet_original(pet_original_path, study)

        labels_in = np.array(nii_pet_segm.dataobj)
        connectivity = 6 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
        labels_out, n_blobs = cc3d.connected_components(labels_in, connectivity=connectivity, delta=3, return_N=True)
        for i in range(1, n_blobs + 1):
            tumor_pixel_original = image_pet_original[labels_out == i]
            suv_max = tumor_pixel_original.max()
            suv_mean = tumor_pixel_original.mean()
            suv_dict_original = {"mean":suv_mean, "max":suv_max}
            tumor_pixel_restored = image_pet_reconstruct[labels_out == i]
            suv_max = tumor_pixel_restored.max()
            suv_mean = tumor_pixel_restored.mean()
            suv_dict_reconstruct = {"mean":suv_mean, "max":suv_max}
            suv_list.append({"original":suv_dict_original, "reconstruct":suv_dict_reconstruct})
        return suv_list


def suv_statistics(suv_list):
    coeff_suv = 1 / 1.3
    suv_full_statistics = dict()
    for mode in ["mean", "max"]:
        suv2 = np.array([x["original"][mode] for x in suv_list])
        suv1 = np.array([x["reconstruct"][mode] for x in suv_list])
        suv1 *= coeff_suv
        suv2 *= coeff_suv
        suv = np.vstack([suv1, suv2]).T
        r2 = r2_score(suv1, suv2)
        diffs = np.diff(suv, axis=1)
        # Average difference (aka the bias)
        bias = np.mean(diffs)
        # Sample standard deviationx
        sd = np.std(diffs, ddof=1)  
        stat_dict = {"r2":1 - r2, "bias":bias, "std":sd}
        suv_full_statistics.update({mode:stat_dict})
    return suv_full_statistics


if __name__ == "__main__":
    output_path = "/ayb/vol1/kruzhilov/pet_test_denoising/nii_result"
    #pet_reconstr_path = "/gim/lv01/kudin/data/low_dose_of_fast_pet_ct/inference/60_s_options_tests/suv_60_s_patch_32_norm_ln_epoch_51"
    pet_reconstr_path = "/ayb/vol1/luka/inference/pix2pix_unet_60sec/unet_delta"
    pet_original_path = "/ayb/vol1/kruzhilov/datasets/dataset_v_5/test"

    suv_list = suv_data(pet_original_path, output_path, pet_reconstr_path)
    suv_stat = suv_statistics(suv_list)
    print("SUV mean")
    print("iR2={0:.5f}, bias={1:.4f}, std={2:.4f}".format(suv_stat["mean"]["r2"], suv_stat["mean"]["bias"], suv_stat["mean"]["std"]))
    print("SUV max")
    print("iR2={0:.5f}, bias={1:.4f}, std={2:.4f}".format(suv_stat["max"]["r2"], suv_stat["max"]["bias"], suv_stat["max"]["std"]))