from skimage import measure
import cv2
import numpy as np
import matplotlib.pyplot as plt

def remove_small_points(image, threshold_point):
    img_label, num = measure.label(image, neighbors=8, return_num=True)  # 输出二值图像中所有的连通域
    props = measure.regionprops(img_label)
    # print(num)# 输出连通域的属性，包括面积等

    resMatrix = np.zeros(img_label.shape)
    for i in range(0, len(props)):
        if props[i].area > threshold_point:
            tmp = (img_label == i + 1).astype(np.uint8)
            resMatrix += tmp  # 组合所有符合条件的连通域
    # resMatrix *= 255
    return resMatrix
