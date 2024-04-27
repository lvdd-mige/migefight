from skimage.measure import label,regionprops
import numpy as np
import cv2
import matplotlib.pyplot as plt

def largestConnectComponent(bw_img):
    labeled_img, num = label(bw_img, neighbors=4, background=0, return_num=True)
    max_label = 0
    max_num = 0
    for i in range(0, num):
        if np.sum(labeled_img == 1) > max_num:
            max_num = np.sum(labeled_img == 1)
            max_label = i
    mcr = (labeled_img == max_label)

    for region in regionprops(labeled_img):
        # skip small images
        if region.area < 50:
            continue
        # print(regionprops(labeled_img)[max_label])
        minr, minc, maxr, maxc = region.bbox

        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    return mcr

Img =cv2.cvtColor(cv2.imread("Pred_Image//3.png"),cv2.COLOR_RGB2GRAY)

mcr=largestConnectComponent(Img)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(mcr)
for region in regionprops(labeled_img):
    # skip small images
    if region.area < 50:
        continue
    # print(regionprops(labeled_img)[max_label])
    minr, minc, maxr, maxc = region.bbox

    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)