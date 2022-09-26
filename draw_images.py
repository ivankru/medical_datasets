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
#import SimpleITK as sitk
import cc3d
#SUV - 0, CT - 1

study = "2.7.093.9.7269334.5.7.7.1815510387.88765.6195363352.13749"
#study = "1.0.329.466197.3.333.0.6234929077.287.9246067012.360"
#study = "1.7.100.4.1493614.9.8.0.0712460420.2220.8986200291.93384"
#study = "2.4.926.467454.6.978.6.8599986150.860.3680649099.978"
#study = "PETCT_0011f3deaf"
#study = "PETCT_01140d52d8"
#output_path = "/ayb/vol1/kruzhilov/pet_test_denoising/result_deleteme"
# output_path = "/ayb/vol1/kruzhilov/pet_test_denoising/nii_result"
# input_path = "/ayb/vol1/kruzhilov/nnUnet/nnUNet_preprocessed/Task001_TCIA/imagesTS"
# #input_path = "/ayb/vol1/kruzhilov/nnUnet/nnUNet_preprocessed/Task002_TCIA/imagesTS"
# segment_path = join(output_path, study + ".nii.gz")
# ct_path = join(input_path, study + "_0001.nii.gz")
# pt_path = join(input_path, study + "_0000.nii.gz")
# nii_pet_segm = nb.load(segment_path)

# #seg_path = "/gim/lv01/palevas/autopet/FDG-PET-CT-Lesions/PETCT_0011f3deaf/03-23-2003-NA-PET-CT Ganzkoerper  primaer mit KM-10445/SEG.nii.gz"
# #seg_path = "/gim/lv01/palevas/autopet/FDG-PET-CT-Lesions/PETCT_01140d52d8/08-13-2005-NA-PET-CT Ganzkoerper  primaer mit KM-56839/SEG.nii.gz"
# #nii_pet_segm = nb.load(seg_path)

# #https://github.com/seung-lab/connected-components-3d/blob/master/README.md
# labels_in = np.array(nii_pet_segm.dataobj)
# connectivity = 6 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
# labels_out, n_blobs = cc3d.connected_components(labels_in, connectivity=connectivity, delta=3, return_N=True)
# # Image statistics like voxel counts, bounding boxes, and centroids.
# stats = cc3d.statistics(labels_out)
# print("number of centroids:", len(stats["centroids"] - 1))
# slice = int(stats["centroids"][2][-1])
# #print(stats["centroids"])

# nii_ct = nb.load(ct_path)
# nii_pet = nb.load(pt_path)  
# #slice = unravel_index(np.argmax(nii_pet_segm.dataobj), nii_pet_segm.dataobj.shape)[-1] 
# image_sem = nii_pet_segm.dataobj[:,:,slice].transpose()
# image_ct = nii_ct.dataobj[:,:,slice].transpose()
# image_pet = nii_pet.dataobj[:,:,slice].transpose()
# plt.rc('xtick', labelsize=8)
# plt.rc('ytick', labelsize=8)
# plt.rc('axes', labelsize=7) 
# plt.rc('font', size=10)
# ax1.set_axis_off()
# ax2.set_axis_off()
# ax3.set_axis_off()
# ax1.imshow(image_ct)
# ax2.imshow(image_pet)
# image_sem = np.stack([image_sem, np.zeros_like(image_pet), 0.2*image_pet], axis=2)
# ax3.imshow(image_sem)
# ct = nb.load(ct_path) 

if __name__ == "__main__":
    from collect_tumors import read_pet_reconstruct_kudin, read_pet_original
    pet_reconstr_path = "/gim/lv01/kudin/data/low_dose_of_fast_pet_ct/inference/60_s_options_tests/suv_60_s_patch_32_norm_ln_epoch_51"
    pet_original_path = "/ayb/vol1/kruzhilov/datasets/dataset_v_5/test"
    original = read_pet_original(pet_original_path, study)
    reconstruct = read_pet_reconstruct_kudin(pet_reconstr_path, study)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax1.imshow(original[:,:,290])
    ax2.imshow(reconstruct[:,:,290])

# %%
