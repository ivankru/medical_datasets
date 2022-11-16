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
from r2_suv_metrics import read_pet_time, read_pet_reconstruct, read_pet_reconstruct_kudin, read_pet_original
from nilearn.image import resample_img
import scipy

# def read_pet_reconstruct(pet_reconstr_path, study_folder, new_size=256):
#     pet_list = []
#     study_path = os.path.join(pet_reconstr_path, study_folder) 
#     for index in range(len(os.listdir(study_path))):     
#         result_path = os.path.join(study_path, str(index).zfill(6) + ".npy")
#         pet_net = np.load(result_path)
#         pet_net[pet_net < 0] = 0
#         pet_net = resize(pet_net, (new_size, new_size))
#         pet_list.append(pet_net)
#     pet_3d = np.stack(pet_list, axis=2)
#     return pet_3d
#     #nifti_file = nb.Nifti1Image(pet_3d, affine)


# def read_pet_reconstruct_kudin(pet_reconstr_path, study_folder, new_size=256):
#     pet_list = []
#     study_path = os.path.join(pet_reconstr_path, study_folder) 
#     for index in range(len(os.listdir(study_path))):     
#         result_path = os.path.join(study_path, str(index).zfill(6) + ".npz")
#         pet_net = np.load(result_path, allow_pickle=True)["image"]
#         pet_net[pet_net < 0] = 0
#         pet_net = resize(pet_net, (new_size, new_size))
#         pet_list.append(pet_net)
#     pet_3d = np.stack(pet_list, axis=2)
#     return pet_3d


# def read_pet_original(pet_original_path, study_folder, new_size=256):
#     study_path = os.path.join(pet_original_path, study_folder)
#     pet90_list = []
#     for index in range(len(os.listdir(study_path))):
#         index_path = os.path.join(study_path, str(index).zfill(6))
#         pet_90_path = os.path.join(index_path, "original_in_suv.npz")
#         pet_90 = np.load(pet_90_path, allow_pickle=True)["image"]
#         pet_90 = resize(pet_90, (new_size, new_size))
#         pet90_list.append(pet_90)
#     pet_3d = np.stack(pet90_list, axis=2)
#     return pet_3d


def draw_r2_plot(suv1, suv2):
    r2 = r2_score(suv1, suv2)
    plt.rcParams["figure.figsize"] = (3.5,3.5)
    #ax.set_facecolor('mediumpurple')
    line = mlines.Line2D([0, np.max(suv1)], [0, np.max(suv2)], color="magenta")
    transform = plt.gca().transAxes
    line.set_transform(transform)
    plt.gca().add_line(line)
    plt.scatter(suv1, suv2, s=10)
    plt.gca().set_title("R2={0:.4f}".format(r2))
    plt.xlabel("SUVmax 90 sec, MBq/g")
    plt.ylabel("SUVmax 30 sec, MBq/g")
    plt.xlim([0,15])
    plt.ylim([0,15])
    plt.show()
    plt.savefig("R2.pdf", format="pdf", bbox_inches='tight', dpi=600)


def bias_sd_up_low(suv_max_pair_list):
    means = suv_max_pair_list.mean(axis=1)
    diffs = np.diff(suv_max_pair_list, axis=1)
    # Average difference (aka the bias)
    bias = np.median(diffs)
    # Sample standard deviationx
    sd = scipy.stats.iqr(diffs)#, ddof=1)  
    upper_loa = bias + 2 * sd
    lower_loa = bias - 2 * sd
    return means, diffs, bias, sd, (lower_loa, upper_loa)


def draw_subplot(ax, means, diffs, bias, sd, loa, y_lim):
    lower_loa, upper_loa = loa
    # Scatter plot
    ax.scatter(means, diffs, c='b', s=10, alpha=0.6, marker='o')
    # Plot the zero line
    ax.axhline(y=0, c='k', lw=0.5)
    # Plot the bias and the limits of agreement
    ax.axhline(y=upper_loa, c='grey', ls='--')
    ax.axhline(y=bias, c='grey', ls='--')
    ax.axhline(y=lower_loa, c='grey', ls='--')
    # Get axis limits
    # Set y-axis limits
    left, right = plt.xlim()
    #bottom, top = plt.ylim()
    #max_y = max(abs(bottom), abs(top))
    ax.set_ylim(y_lim[0], y_lim[1])
    # Set x-axis limits
    #domain = right - left
    #ax.set_xlim(left, left + domain * 1.1)
    ax.set_xlim(0, 10)
    left, right = plt.xlim()
    left, right = left, right
    #bottom, top = plt.ylim()
    # Add the annotations
    right_step = -15
    ax.annotate('+2×IQR', (right, upper_loa), (right_step, 7), textcoords='offset pixels')
    ax.annotate(f'{upper_loa:+4.2f}', (right, upper_loa), (right_step, -10), textcoords='offset pixels')
    ax.annotate('Median' + "\n" + 'bias', (right, bias), (right_step, 7), textcoords='offset pixels')
    ax.annotate(f'{bias:+4.2f}', (right, bias), (right_step, -10), textcoords='offset pixels')
    ax.annotate('-2×IQR', (right, lower_loa), (right_step, 7), textcoords='offset pixels')
    ax.annotate(f'{lower_loa:+4.2f}', (right, lower_loa), (right_step, -10), textcoords='offset pixels')


def draw_subplot2(ax, means, diffs, bias, sd, loa, y_lim):
    lower_loa, upper_loa = loa
    # Scatter plot
    ax.scatter(means, diffs, c='b', s=10, alpha=0.6, marker='o')
    # Plot the zero line
    ax.axhline(y=0, c='k', lw=0.5)
    # Plot the bias and the limits of agreement
    ax.axhline(y=upper_loa, c='grey', ls='--')
    ax.axhline(y=bias, c='grey', ls='--')
    ax.axhline(y=lower_loa, c='grey', ls='--')
    # Get axis limits
    # Set y-axis limits
    left, right = plt.xlim()
    #bottom, top = plt.ylim()
    #max_y = max(abs(bottom), abs(top))
    ax.set_ylim(y_lim[0], y_lim[1])
    # Set x-axis limits
    #domain = right - left
    #ax.set_xlim(left, left + domain * 1.1)
    ax.set_xlim(0, 10)
    left, right = plt.xlim()
    left, right = left, right
    #bottom, top = plt.ylim()
    # Add the annotations
    right_step = 160
    ax.annotate('+2×IQR', (right, upper_loa), (right_step, 7), textcoords='offset pixels')
    ax.annotate(f'{upper_loa:+4.2f}', (right, upper_loa), (right_step, -10), textcoords='offset pixels')
    ax.annotate('Median' + "\n" + 'bias', (right, bias), (right_step, 7), textcoords='offset pixels')
    ax.annotate(f'{bias:+4.2f}', (right, bias), (right_step, -10), textcoords='offset pixels')
    ax.annotate('-2×IQR', (right, lower_loa), (right_step, 7), textcoords='offset pixels')
    ax.annotate(f'{lower_loa:+4.2f}', (right, lower_loa), (right_step, -10), textcoords='offset pixels')


def bland_alman_plot_v2(suv_max_pair_list, suv_max_pair_list2):
    fig, axs = plt.subplots(2, figsize=(3.5, 7.1))
    #fig.tight_layout()
    #fig.suptitle('Bland-Altman Plot for PET 90 sec' + "\n")
    plt.subplots_adjust(hspace=0.4)
    #fig.tight_layout(h_pad=5)
    #plt.subplots_adjust(top=0.05)

    means1, diffs1, bias1, sd1, loa1 = bias_sd_up_low(suv_max_pair_list2)
    means2, diffs2, bias2, sd2, loa2 = bias_sd_up_low(suv_max_pair_list)
    lower_loa = 1.1 * min(loa1[0], loa2[0])
    upper_loa = 1.1 * max(loa1[1], loa2[1])
    y_lim = (lower_loa, upper_loa)

    axs[0].set(
        title='ResNet denoising' + "\n" + "from 30 sec", 
        xlabel='SUVmax (MBq/g)' + "\n" + '(PET ResNet 30 sec + PET 90 sec) / 2',
        ylabel='SUVmax (MBq/g)' + "\n" + "PET 90 sec - PET ResNet 30 sec"
    )
    draw_subplot2(axs[0], means1, diffs1, bias1, sd1, loa1, y_lim)   
    
    #----------------------------------------------------------------------
    axs[1].set(
        title='Noised images 30 sec', 
        xlabel='SUVmax (MBq/g)' + "\n" + '(PET 30 sec + PET 90 sec) / 2',
        ylabel='SUVmax (MBq/g)' + "\n" + "PET 90 sec - PET 30 sec"
    )
    draw_subplot(axs[1], means2, diffs2, bias2, sd2, loa2, y_lim)   

    # Show plot
    plt.savefig("bland_alman2.pdf", format="pdf", dpi=600, bbox_inches='tight')
    plt.show()


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
        title='Bland-Altman Plot for PET 90 sec and' + "\n" + 'ResNet denoising from 30 sec', 
        xlabel='SUVmax (g/ml)' + "\n" + '(PET ResNet 30 sec + PET 90 sec) / 2',
        ylabel='SUVmax (g/ml)' + "\n" + "PET 90 sec - PET ResNet 30 sec"
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
    # Set y-axis limits
    left, right = plt.xlim()
    bottom, top = plt.ylim()
    max_y = max(abs(bottom), abs(top))
    ax.set_ylim(bias - sd * 2.5, bias + sd * 2.5)
    # Set x-axis limits
    domain = right - left
    #ax.set_xlim(left, left + domain * 1.1)
    ax.set_xlim(left, 15)
    left, right = plt.xlim()
    left, right = left - 3, right - 3
    bottom, top = plt.ylim()
    # Add the annotations
    ax.annotate('+2×SD', (right, upper_loa), (0, 7), textcoords='offset pixels')
    ax.annotate(f'{upper_loa:+4.2f}', (right, upper_loa), (0, -15), textcoords='offset pixels')
    ax.annotate('Bias', (right, bias), (0, 7), textcoords='offset pixels')
    ax.annotate(f'{bias:+4.2f}', (right, bias), (0, -15), textcoords='offset pixels')
    ax.annotate('-2×SD', (right, lower_loa), (0, 7), textcoords='offset pixels')
    ax.annotate(f'{lower_loa:+4.2f}', (right, lower_loa), (0, -15), textcoords='offset pixels')
    # Show plot
    plt.show()
    return bias, sd


def suv_peak_funct(image_pet_original, image_pet, labels_out, i, pixel_radius=3.4, pixel_radius_z=1.89):
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
    

def mean_peak_max(image_pet, labels_out, i):
    tumor_pixel= image_pet[labels_out == i]
    suv_max = tumor_pixel.max()
    suv_mean = tumor_pixel.mean()
    suv_peak = suv_peak_funct(image_pet, image_pet, labels_out, i)
    dict_reconstruct = {"mean":suv_mean, "max":suv_max, "peak":suv_peak}
    return dict_reconstruct 


if __name__  == "__main__":
    output_path = "/ayb/vol1/kruzhilov/pet_test_denoising/nii_result"
    output_val_path = "/ayb/vol1/kruzhilov/pet_test_denoising/nii_val_result"
    #pet_reconstr_path = "/gim/lv01/kudin/data/low_dose_of_fast_pet_ct/inference/60_s_options_tests/suv_60_s_patch_32_norm_ln_epoch_51"
    #pet_reconstr_path = "/ayb/vol1/luka/inference/pix2pix_unet_60sec/unet_delta"
    pet_reconstr_path = "/ayb/vol1/luka/inference/pix2pix_resnet_30sec/resnet_gan_newdelta_alpha270_test"
    pet_reconstr_path_val = "/ayb/vol1/luka/inference/pix2pix_resnet_30sec/resnet_gan_newdelta_alpha270_val"
    pet_original_path = "/ayb/vol1/kruzhilov/datasets/dataset_v_5/test"
    pet_original_path_val = "/ayb/vol1/kruzhilov/datasets/dataset_v_5/val"
    input_path = "/ayb/vol1/kruzhilov/nnUnet/nnUNet_preprocessed/Task001_TCIA/imagesTS"
    #study = "2.7.093.9.7269334.5.7.7.1815510387.88765.6195363352.13749"
    suv_list = []
    blob_size = []

    original_pathes = os.listdir(pet_original_path) + os.listdir(pet_original_path_val) 
    for i, study in enumerate(original_pathes):
        # if i == 5:
        #     break
        if study in os.listdir(pet_original_path_val):
            segment_path = join(output_val_path, study + ".nii.gz")
        else:
            segment_path = join(output_path, study + ".nii.gz")
        pet_path = join(input_path, study + "_0000.nii.gz")
        nii_pet_segm = nb.load(segment_path)
        new_size = 256
        transform_coeff = nii_pet_segm.shape[0] / new_size
        affine_matrix = np.copy(nii_pet_segm.affine)
        affine_matrix[:2,:3] = transform_coeff * affine_matrix[:2,:3]
        nii_pet_segm = resample_img(nii_pet_segm, target_shape = [new_size, new_size, nii_pet_segm.shape[-1]], target_affine=affine_matrix, interpolation='nearest')
        #image_pet_original = np.array(nb.load(pet_path).dataobj).transpose((1,0,2))

        image_segm = np.array(nii_pet_segm.dataobj)#[:,:,slice].transpose()
        if study in os.listdir(pet_original_path_val):
            image_pet_reconstruct = read_pet_reconstruct_kudin(pet_reconstr_path_val, study)
            image_pet_noised = read_pet_time(pet_original_path_val, study, sec=30)
            image_pet_original = read_pet_original(pet_original_path_val, study)
        else:
            image_pet_reconstruct = read_pet_reconstruct_kudin(pet_reconstr_path, study)
            image_pet_noised = read_pet_time(pet_original_path, study, sec=30)
            image_pet_original = read_pet_original(pet_original_path, study)

        labels_in = np.round(np.array(nii_pet_segm.dataobj))
        connectivity = 6 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
        labels_out, n_blobs = cc3d.connected_components(labels_in, connectivity=connectivity, delta=3, return_N=True)
        stats = cc3d.statistics(labels_out)
        for i in range(1, n_blobs + 1):
            delta_x = stats["bounding_boxes"][i][0].stop - stats["bounding_boxes"][i][0].start          
            delta_y = stats["bounding_boxes"][i][1].stop - stats["bounding_boxes"][i][1].start 
            delta_z = stats["bounding_boxes"][i][2].stop - stats["bounding_boxes"][i][2].start 
            if delta_x < 3 and delta_y < 3 and delta_z < 2:
                continue
            suv_dict_original = mean_peak_max(image_pet_original, labels_out, i)
            if suv_dict_original["mean"] < 0.5:
                continue
            suv_dict_noised = mean_peak_max(image_pet_noised, labels_out, i)
            suv_dict_reconstruct = mean_peak_max(image_pet_reconstruct, labels_out, i)
            suv_list.append({"original":suv_dict_original, "noised":suv_dict_noised, "reconstruct":suv_dict_reconstruct})
        blob_size += list(stats['voxel_counts'][1:])

    # bins = np.arange(0, 120, 5)
    # counts, bins = np.histogram(np.array(blob_size), bins=bins)
    # plt.hist(bins[:-1], bins, weights=counts)
    # plt.xlabel("tumor size, number of voxels")
    # plt.ylabel("number of tumors")

    #print(suv_list)
    mode = "max"
    suv0 = np.array([x["original"][mode] for x in suv_list])
    suv1 = np.array([x["reconstruct"][mode] for x in suv_list])
    suv2 = np.array([x["noised"][mode] for x in suv_list])
    #r2 = draw_r2_plot(suv0,  suv1)
    #print("r2:", r2)
    coeff_suv = 1 / 1.3
    suv0 *= coeff_suv
    suv1 *= coeff_suv
    suv2 *= coeff_suv
    #suv = np.vstack([suv1, suv0]).T
    suv_a = np.vstack([suv1, suv0]).T
    suv_b = np.vstack([suv2, suv0]).T
    #bias, sd = bland_alman_plot(suv)
    draw_r2_plot(suv0, suv2)
    #bland_alman_plot_v2(suv_b, suv_a)
# %%
