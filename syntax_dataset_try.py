#%%
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import EuclideanTransform
import cc3d
#from copy import deepcopy


def substract_background(image):
    median = cv2.medianBlur(image.astype(np.uint8), ksize=91)
    background_substract = image.astype(np.float32) - median
    return background_substract


def image_alignment(image1, image2):
    # Extract and match features from both images
    orb = cv2.ORB_create()
    queryKP1, queryDes1 = orb.detectAndCompute(image1, None)
    #final_img = cv2.drawKeypoints(image1, queryKP, image1)
    queryKP2, queryDes2 = orb.detectAndCompute(image2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(queryDes1, queryDes2)
    matches = sorted(matches, key = lambda x:x.distance)
    src_pts = np.float32([ queryKP1[m.queryIdx].pt for m in matches[:100] ]).reshape(-1,1,2)
    dst_pts = np.float32([ queryKP2[m.trainIdx].pt for m in matches[:100] ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h,w = image1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    
    # img2 = cv2.polylines(image2*5, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)
    # plt.imshow(img2)
    # res = cv2.drawMatches(image1*5, queryKP1, img2, queryKP2, matches[:25],None,flags=2)
    # plt.imshow(res)

    #plt.imshow(image2)
    warp_image = cv2.warpPerspective(image1, M, (h,w))
    plt.imshow(image1 + warp_image)

 
def get_artery_segment_weak(input_image):
    median = cv2.medianBlur(input_image.astype(np.uint8), ksize=91)
    background_substract = input_image.astype(np.float32) - median
    segmentation = np.zeros_like(background_substract)
    #gauss_blur = background_substract#cv2.GaussianBlur(background_substract, (7, 7), 0)
    #segmentation[background_substract < thresh] = 0
    thresh = threshold_otsu(background_substract)
    segmentation[background_substract >= thresh] = 1
    # median = cv2.medianBlur(gauss_blur.astype(np.uint8), ksize=51)
    # background_substract = gauss_blur - median
    # Set up the detector with default parameters.
    connectivity = 4 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    labels_out, n_blobs = cc3d.connected_components(segmentation, connectivity=connectivity, delta=3, return_N=True)
    #min_voxel = 50
    stats = cc3d.statistics(labels_out)
    max_blob = np.argmax(np.array(stats["voxel_counts"][1:])) + 1
    segmentation[labels_out != max_blob] = 0
    # for i, voxel_count in enumerate(stats["voxel_counts"][1:]):
    #     if voxel_count < min_voxel:
    #         gauss_blur[labels_out == i + 1] = 0
    background_substract[segmentation == 0] = 0
    gauss_blur = cv2.GaussianBlur(background_substract, (7, 7), 0)
    # th3 = cv2.adaptiveThreshold(gauss_blur.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
    #          cv2.THRESH_BINARY, 17, 0)
    thresh = threshold_otsu(gauss_blur)
    gauss_blur[gauss_blur < thresh] = 0
    #gauss_blur[gauss_blur >= thresh] = 1
    return gauss_blur.astype(np.uint8)


def get_artery_segment_strong(input_image):
    median = cv2.medianBlur(input_image.astype(np.uint8), ksize=91)
    background_substract = input_image.astype(np.float32) - median
    segmentation = np.zeros_like(background_substract)
    #gauss_blur = background_substract#cv2.GaussianBlur(background_substract, (7, 7), 0)
    #segmentation[background_substract < thresh] = 0
    thresh = threshold_otsu(background_substract)
    segmentation[background_substract >= thresh] = 1
    # median = cv2.medianBlur(gauss_blur.astype(np.uint8), ksize=51)
    # background_substract = gauss_blur - median
    # Set up the detector with default parameters.
    connectivity = 4 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    labels_out, n_blobs = cc3d.connected_components(segmentation, connectivity=connectivity, delta=3, return_N=True)
    #min_voxel = 50
    stats = cc3d.statistics(labels_out)
    max_blob = np.argmax(np.array(stats["voxel_counts"][1:])) + 1
    segmentation[labels_out != max_blob] = 0
    # for i, voxel_count in enumerate(stats["voxel_counts"][1:]):
    #     if voxel_count < min_voxel:
    #         gauss_blur[labels_out == i + 1] = 0
    background_substract[segmentation == 0] = 0
    gauss_blur = cv2.GaussianBlur(background_substract, (7, 7), 0)
    th3 = cv2.adaptiveThreshold(gauss_blur.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
             cv2.THRESH_BINARY, 17, 0)
    # thresh = threshold_otsu(gauss_blur)
    gauss_blur[th3] = 0
    #gauss_blur[gauss_blur >= thresh] = 1
    return gauss_blur




if __name__ == "__main__":
    #file_path = "/ayb/vol2/datasets/tymen_syntax_npz/right_type/00005/left/Left_Coro_15_fps_Normal_2.npz"
    #file_path = "/ayb/vol2/datasets/tymen_syntax_npz/right_type/00001/right/Left_Coro_15_fps_Normal_7.npz"
    file_path = "/ayb/vol2/datasets/tymen_syntax_npz/left_type/00003/left/Left_Coro_15_fps_Normal_3.npz"
    #file_path = "/ayb/vol2/datasets/tymen_syntax_npz/left_type/00019/right/Left_Coro_15_fps_Normal_7.npz"
    a = np.load(file_path)["arr_0"]
    
    #edge = cv2.dilate(cv2.Canny(median.astype(np.uint8), 20, 300), None)
    #edge = cv2.dilate(cv2.Canny(median.astype(np.uint8), 10, 100), None)

    # blur = cv2.GaussianBlur(image,(7,7), 0)
    # laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    # sobelxy = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    
    # cnt = sorted(cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    # mask = np.zeros((512,512), np.uint8)
    # masked = cv2.drawContours(mask, [cnt],-1, 255, 0)

    # thresh = threshold_otsu(gauss_blur)
    # img_otsu  = gauss_blur < thresh
    # gauss_blur[img_otsu] = 0
    # gauss_blur[gauss_blur > thresh] = 1

    # th3 = cv2.adaptiveThreshold(gauss_blur.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
    #         cv2.THRESH_BINARY, 21, 0)

    # kernel_3x3 = np.ones((21, 21), np.float32)
    # kernel_3x3 = kernel_3x3 / sum(abs(kernel_3x3))
    # blurred = cv2.filter2D(image, -1, kernel_3x3)

    i = 30
    image1_orig = a[i-1,:,:].astype(np.uint8) #/ 10
    image1 = image1_orig.max() - image1_orig
    seg1_w = get_artery_segment_weak(image1)
    seg1_s = get_artery_segment_strong(image1)
    image2 = a[i,:,:].astype(np.uint8) #/ 10
    image2 = image2.max() - image2
    seg2 = get_artery_segment_weak(image2)
    # image3 = a[i+1,:,:].astype(np.uint8) #/ 10
    # image3 = image3.max() - image3
    # seg3 = get_artery_segment(image3)

    image_alignment(seg1_w, seg2)
    
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(10,4.9))
    # ax1.set_axis_off()
    # ax2.set_axis_off()
    # ax3.set_axis_off()
    # ax4.set_axis_off()
    # plt.subplots_adjust(wspace=0.4)
    # fig.tight_layout(h_pad=5)
    # fig.suptitle("Left type, right artery", fontsize=16)
    # ax1.set_title("Original X-ray")
    # ax2.set_title("Inverted image")
    # ax3.set_title("Weak segmentation")
    # ax4.set_title("Strong segmentation")
    # ax1.imshow(image1_orig, cmap="RdPu")
    # ax2.imshow(image1, cmap="RdPu")
    # ax3.imshow(seg1_w, cmap="RdPu")# cmap="RdPu"
    # ax4.imshow(seg1_s, cmap="RdPu")


# %%
