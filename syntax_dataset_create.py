import os
import shutil
import pydicom
import numpy as np

def save_projection(projection_path, save_npz_path):
    if not os.path.isfile(projection_path):
        dcm_file = os.listdir(projection_path)[0]
        #for dcm_file in dcm_files:
        dcm_file = os.path.join(projection_path, dcm_file)
    else:
        dcm_file = projection_path
    assert dcm_file[-3:] == "dcm"
    ds = pydicom.dcmread(dcm_file)
    a = ds.pixel_array
    np.savez(save_npz_path, a)


def delete_create_folder(folder_path):
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)
    else:
        os.mkdir(folder_path)


def create_syntax_dataset(xray_path, path_to_save):
    studies = os.listdir(xray_path)
    for study in studies:
        study_path = os.path.join(xray_path, study)
        path_to_save_study = os.path.join(path_to_save, study[5:])
        delete_create_folder(path_to_save_study)

        for subdir in os.listdir(study_path):
            if "лев" in subdir:
                left_path = os.path.join(study_path, subdir)
            if "прав" in subdir:
                right_path = os.path.join(study_path, subdir)
        
        for lr in ["left", "right"]:
            path_to_save_study_lr = os.path.join(path_to_save_study, lr)
            delete_create_folder(path_to_save_study_lr)
            if lr == "left":
                lr_path = left_path
            elif lr == "right":
                lr_path = right_path

            projections = os.listdir(lr_path)
            for projection in projections:
                projection_path = os.path.join(lr_path, projection)
                path_to_save_study_lr_projection = os.path.join(path_to_save_study_lr, projection + ".npz")
                save_projection(projection_path, path_to_save_study_lr_projection)


if __name__ == "__main__":
    xray_path = "/ayb/vol2/datasets/tymen_syntax/rasmetka/right_type_230"
    path_to_save = "/ayb/vol2/datasets/tymen_syntax_npz/right_type"
