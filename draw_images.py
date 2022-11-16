#%%
""""
nnUNet_predict -i /ayb/vol1/kruzhilov/nnUnet/nnUNet_preprocessed/Task001_TCIA/imagesTS  -o /ayb/vol1/kruzhilov/pet_test_denoising/nii_result -t 1 -m 3d_fullres
how to get volume in mm https://stackoverflow.com/questions/67652500/calculating-volume-of-3d-model-in-python
"""
import nibabel as nb
import matplotlib.pyplot as plt
import numpy as np
#from numpy import unravel_index
from os.path import join
import os
#import SimpleITK as sitk
import cc3d
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_modality_lut
#SUV - 0, CT - 1

def normalization_window(image, window_center=-600, window_width=1500):
    image_min = window_center - window_width / 2
    image_max = window_center + window_width / 2
    image = np.minimum(image, image_max)
    image = np.maximum(image, image_min)
    return image


def draw_ct_pet_segm():
    #study = "2.7.093.9.7269334.5.7.7.1815510387.88765.6195363352.13749"
    #study = "1.0.329.466197.3.333.0.6234929077.287.9246067012.360"
    #study = "1.7.100.4.1493614.9.8.0.0712460420.2220.8986200291.93384"
    study = "2.8.406.7.2604203.6.6.9.9475838781.4997.2616928332.46586"
    output_path = "/ayb/vol1/kruzhilov/pet_test_denoising/nii_result"
    input_path = "/ayb/vol1/kruzhilov/nnUnet/nnUNet_preprocessed/Task001_TCIA/imagesTS_test"
    dcm_path = "/ayb/vol1/kudin/dev/datasets/raw_data_part_1/"
    #input_path = "/ayb/vol1/kruzhilov/nnUnet/nnUNet_preprocessed/Task002_TCIA/imagesTS"
    segment_path = join(output_path, study + ".nii.gz")
    ct_path = join(input_path, study + "_0001.nii.gz")
    pt_path = join(input_path, study + "_0000.nii.gz")
    nii_pet_segm = nb.load(segment_path)

    dcm_file = join(dcm_path, study)
    dcm_file = join(dcm_file, os.listdir(dcm_file)[0])
    dcm_file = join(dcm_file, os.listdir(dcm_file)[0])
    ds = dcmread(dcm_file)

    #seg_path = "/gim/lv01/palevas/autopet/FDG-PET-CT-Lesions/PETCT_0011f3deaf/03-23-2003-NA-PET-CT Ganzkoerper  primaer mit KM-10445/SEG.nii.gz"
    #seg_path = "/gim/lv01/palevas/autopet/FDG-PET-CT-Lesions/PETCT_01140d52d8/08-13-2005-NA-PET-CT Ganzkoerper  primaer mit KM-56839/SEG.nii.gz"
    #nii_pet_segm = nb.load(seg_path)

    #https://github.com/seung-lab/connected-components-3d/blob/master/README.md
    labels_in = np.array(nii_pet_segm.dataobj)
    connectivity = 6 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    labels_out, n_blobs = cc3d.connected_components(labels_in, connectivity=connectivity, delta=3, return_N=True)
    # Image statistics like voxel counts, bounding boxes, and centroids.
    stats = cc3d.statistics(labels_out)
    print("number of centroids:", len(stats["centroids"] - 1))
    slice = int(stats["centroids"][1][-1])
    #print(stats["centroids"])

    nii_ct = nb.load(ct_path)
    nii_pet = nb.load(pt_path)  
    #slice = unravel_index(np.argmax(nii_pet_segm.dataobj), nii_pet_segm.dataobj.shape)[-1] 
    image_sem = nii_pet_segm.dataobj[:,:,slice].transpose()
    image_ct = nii_ct.dataobj[:,:,slice].transpose()
    image_ct = apply_modality_lut(image_ct, ds)
    image_ct = normalization_window(image_ct, window_center=-600, window_width=1500)
    image_pet = nii_pet.dataobj[:,:,slice].transpose()
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,4.9))
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=7) 
    plt.rc('font', size=10)
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    ax1.imshow(image_ct)
    ax2.imshow(image_pet)
    image_sem = np.stack([image_sem, np.zeros_like(image_pet), 0.2*image_pet], axis=2)
    ax3.imshow(image_sem)


if __name__ == "__main__":
    #study = "2.8.406.7.2604203.6.6.9.9475838781.4997.2616928332.46586"
    study = "3.0.687.986273.7.876.9.7335420431.731.3639789653.003"
    from r2_suv_metrics import read_pet_reconstruct_kudin, read_pet_original, read_pet_time
    #pet_reconstr_path = "/ayb/vol1/luka/inference/pix2pix_resnet_30sec/resnet_gan_newdelta_alpha270_test"
    pet_reconstr_path = "/ayb/vol1/luka/inference/unet_gan_with_noise/unet_gan_with_noise/"
    pet_original_path = "/ayb/vol1/kruzhilov/datasets/dataset_v_5/test"
    original = read_pet_original(pet_original_path, study)
    reconstruct = read_pet_reconstruct_kudin(pet_reconstr_path, study)
    noised = read_pet_time(pet_original_path, study, sec=30)
    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(3.5, 3.5))
    # ax1.set_axis_off()
    # ax2.set_axis_off()
    # ax3.set_axis_off()
    plt.rcParams["figure.figsize"] = (3.5,3.5)
    plt.axis("off")
    slice = 219
    max_o, max_n, max_r = original.max(), noised.max(), reconstruct.max()
    sup_max = max(max_o, max_n, max_n)
    
    original_image =  (255.0 / sup_max) * original[50:200,50:200,slice]
    plt.imshow(original_image)
    plt.savefig("original.pdf", bbox_inches='tight', pad_inches=0, format="pdf", dpi=600)
    
    noised_image = (255.0 / sup_max) * noised[50:200,50:200,slice]
    plt.imshow(noised_image)
    plt.savefig("noised.pdf", bbox_inches='tight', pad_inches=0, format="pdf", dpi=600)
    #ax2.imshow(noised_image)
    
    rec_image = (255.0 / sup_max) * reconstruct[50:200,50:200,slice]
    plt.imshow(rec_image)
    plt.savefig("denoised.pdf", bbox_inches='tight', pad_inches=0, format="pdf", dpi=600)
    # # ax3.imshow(rec_image)

# %%
