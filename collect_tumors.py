#%%
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
    pet_3d = np.stack(pet90_list, axis=2)
    return pet_3d


def draw_r2_plot(suv1, suv2):
    r2 = r2_score(suv1, suv2)
    fig, ax = plt.subplots()
    #ax.set_facecolor('mediumpurple')
    line = mlines.Line2D([0, np.max(suv1)], [0, np.max(suv2)], color="magenta")
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    ax.scatter(suv1, suv2, s=10, cmap="Reds")
    ax.set_title("R2={0:.5f}".format(r2))
    plt.xlabel("PET 90 sec")
    plt.ylabel("PET denoised from 30 sec")
    plt.show()
    return r2


def bland_alman_plot(suv_max_pair_list):
    means = suv_max_pair_list.mean(axis=1)
    diffs = np.diff(suv_max_pair_list, axis=1)
    # Average difference (aka the bias)
    bias = np.mean(diffs)
    # Sample standard deviationx
    sd = np.std(diffs, ddof=1)  
    upper_loa = bias + 2 * sd
    lower_loa = bias - 2 * sd

    ax = plt.axes()
    ax.set(
        title='Bland-Altman Plot for PET denoising',
        xlabel='(orginal + denoised) / 2', ylabel='denoised - orginal'
    )
    # Scatter plot
    ax.scatter(means, diffs, c='k', s=10, alpha=0.6, marker='o')
    # Plot the zero line
    ax.axhline(y=0, c='k', lw=0.5)
    # Plot the bias and the limits of agreement
    ax.axhline(y=upper_loa, c='grey', ls='--')
    ax.axhline(y=bias, c='grey', ls='--')
    ax.axhline(y=lower_loa, c='grey', ls='--')
    # Get axis limits
    left, right = plt.xlim()
    bottom, top = plt.ylim()
    # Set y-axis limits
    max_y = max(abs(bottom), abs(top))
    ax.set_ylim(-max_y * 1.1, max_y * 1.1)
    # Set x-axis limits
    domain = right - left
    ax.set_xlim(left, left + domain * 1.1)
    # Add the annotations
    ax.annotate('+2×SD', (right, upper_loa), (0, 7), textcoords='offset pixels')
    ax.annotate(f'{upper_loa:+4.2f}', (right, upper_loa), (0, -25), textcoords='offset pixels')
    ax.annotate('Bias', (right, bias), (0, 7), textcoords='offset pixels')
    ax.annotate(f'{bias:+4.2f}', (right, bias), (0, -25), textcoords='offset pixels')
    ax.annotate('-2×SD', (right, lower_loa), (0, 7), textcoords='offset pixels')
    ax.annotate(f'{lower_loa:+4.2f}', (right, lower_loa), (0, -25), textcoords='offset pixels')
    # Show plot
    plt.show()
    return bias, sd


if __name__  == "__main__":
    output_path = "/ayb/vol1/kruzhilov/pet_test_denoising/nii_result"
    #pet_reconstr_path = "/gim/lv01/kudin/data/low_dose_of_fast_pet_ct/inference/60_s_options_tests/suv_60_s_patch_32_norm_ln_epoch_51"
    pet_reconstr_path = "/ayb/vol1/luka/inference/pix2pix_unet_60sec/unet_delta"
    pet_original_path = "/ayb/vol1/kruzhilov/datasets/dataset_v_5/test"
    input_path = "/ayb/vol1/kruzhilov/nnUnet/nnUNet_preprocessed/Task001_TCIA/imagesTS"
    #study = "2.7.093.9.7269334.5.7.7.1815510387.88765.6195363352.13749"
    suv_list = []


    for i, study in enumerate(os.listdir(pet_original_path)):
        if i == 50:
            break
        segment_path = join(output_path, study + ".nii.gz")
        pet_path = join(input_path, study + "_0000.nii.gz")
        nii_pet_segm = nb.load(segment_path)
        image_pet_original = np.array(nb.load(pet_path).dataobj).transpose((1,0,2))

        image_segm = np.array(nii_pet_segm.dataobj)#[:,:,slice].transpose()
        image_pet_reconstruct = read_pet_reconstruct(pet_reconstr_path, study)
        #image_pet_original = read_pet_original(pet_original_path, study)

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

    mode = "mean"
    suv0 = np.array([x["original"][mode] for x in suv_list])
    suv1 = np.array([x["reconstruct"][mode] for x in suv_list])
    r2 = draw_r2_plot(suv0,  1/1.3*suv1)
    print("r2:", r2)
    # suv = np.vstack([suv0, suv1]).T
    # bias, sd = bland_alman_plot(suv)
    # print(bias, sd)

# %%
