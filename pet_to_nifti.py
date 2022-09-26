import numpy as np
import nibabel as nib
import os

#path to id_series for each study in Kudin dataset
#train\0.0.891.686346.0.167.5.6355073221.240.4893416175.279\000000\metadata.json
 
if __name__ == "__main__":
    test_path = "/ayb/vol1/kruzhilov/datasets/dataset_v_5/test"
    affine = np.eye(4)
    # nifti_file = nib.Nifti1Image(converted_array, affine)
    # nib.save(nifti_file, path_to_save)

    for study_folder in os.listdir(test_path):
        folder_path = os.path.join(test_path, study_folder)
        #save_folder = os.path.join(result_path, study_folder)
        # if not os.path.isdir(save_folder):
        #     os.mkdir(save_folder)
        pet90_list = []
        for index in os.listdir(folder_path):
            index_path = os.path.join(folder_path, index)
            pet_30_path = os.path.join(index_path, "30_in_suv.npz")
            pet_60_path = os.path.join(index_path, "60_in_suv.npz")
            pet_90_path = os.path.join(index_path, "original_in_suv.npz")
            #pet_30 = np.load(pet_30_path, allow_pickle=True)["image"]
            #pet_60 = np.load(pet_60_path, allow_pickle=True)["image"]
            pet_90 = np.load(pet_90_path, allow_pickle=True)["image"]
            pet90_list.append(pet_90)
            # index_folder = os.path.join(save_folder, index)
            # if not os.path.isdir(index_folder):
            #     os.mkdir(index_folder)
        pet_3d = np.stack(pet90_list, axis=2)
        nifti_file = nib.Nifti1Image(pet_3d, affine)
        break