from glob import glob
import numpy as np
import shutil
import cv2
from sklearn import preprocessing
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
import imageio

# img=glob("public_imgADTrain_npy//*")
# mask=glob("public_maskADTrain_npy//*")
#
# # for n in img:
# #     shutil.copy(n,"pytorch-CycleGAN-and-pix2pix-master//datasets//mydataset//result_mask//{}".format(n.split("/")[-1]))
# #
# # for n in mask:
# #     shutil.copy(n,"pytorch-CycleGAN-and-pix2pix-master//datasets//mydataset//input//{}".format(n.split("/")[-1]))
#
# for n in range(len(img)):
#     if img[n].split("/")[-1]!=mask[n].split("/")[-1]:
#         print("False!!!!!!")
#     img_tmp=np.load(img[n])
#     img_tmp=img_tmp.astype("float16")
#     a = np.max(img_tmp)
#     # plt.imshow(img_tmp)
#     # plt.show()
#     mask_tmp=np.load(mask[n])
#     dataset=np.zeros((img_tmp.shape[0],img_tmp.shape[1]*2))
#
#     min_max=preprocessing.StandardScaler()
#     # img_tmp=min_max.fit_transform(img_tmp)*255
#     # img_tmp = min_max.fit_transform(img_tmp)
#     # plt.imshow(img_tmp)
#     # plt.show()
#     dataset[:,0:img_tmp.shape[1]]=img_tmp
#     dataset[:,img_tmp.shape[1]:]=mask_tmp*a
#
#     # im = Image.fromarray(dataset)
#     # im.save("pytorch-CycleGAN-and-pix2pix-master//datasets//mydataset//train//{}.png".format(img[n].split("/")[-1].split(".")[0]))
#     imageio.imwrite("pytorch-CycleGAN-and-pix2pix-master//datasets//mydataset//public-train//{}.png".format(img[n].split("/")[-1].split(".")[0]),dataset)
#     # dataset_img=cv2.imwrite("pytorch-CycleGAN-and-pix2pix-master//datasets//mydataset//train//{}.png".format(img[n].split("/")[-1].split(".")[0]),dataset)
#
img=glob("public_imgADTest_npy//*")
mask=glob("public_maskADTest_npy//*")

# for n in img:
#     shutil.copy(n,"pytorch-CycleGAN-and-pix2pix-master//datasets//mydataset//result_mask//{}".format(n.split("/")[-1]))
#
# for n in mask:
#     shutil.copy(n,"pytorch-CycleGAN-and-pix2pix-master//datasets//mydataset//input//{}".format(n.split("/")[-1]))

for n in range(len(img)):
    if img[n].split("/")[-1]!=mask[n].split("/")[-1]:
        print("False!!!!!!")
    img_tmp=np.load(img[n])
    img_tmp = img_tmp.astype("float16")
    b = np.max(img_tmp)
    mask_tmp=np.load(mask[n])
    dataset=np.zeros((img_tmp.shape[0],img_tmp.shape[1]*2))

    min_max=preprocessing.MinMaxScaler()
    # img_tmp=min_max.fit_transform(img_tmp)*255

    dataset[:, 0:img_tmp.shape[1]] = img_tmp
    dataset[:,img_tmp.shape[1]:]=mask_tmp*b
    # dataset_img=cv2.imwrite("pytorch-CycleGAN-and-pix2pix-master//datasets//mydataset//test//{}.png".format(img[n].split("/")[-1].split(".")[0]),dataset)
    imageio.imwrite("pytorch-CycleGAN-and-pix2pix-master//datasets//mydataset//public-test//{}.png".format(img[n].split("/")[-1].split(".")[0]),dataset)


# img=glob("case100_imgADTest_npy//*")
# mask=glob("case100_maskADTest_npy//*")
#
# # for n in img:
# #     shutil.copy(n,"pytorch-CycleGAN-and-pix2pix-master//datasets//mydataset//result_mask//{}".format(n.split("/")[-1]))
# #
# # for n in mask:
# #     shutil.copy(n,"pytorch-CycleGAN-and-pix2pix-master//datasets//mydataset//input//{}".format(n.split("/")[-1]))
#
# for n in range(2):
#     if img[n].split("/")[-1]!=mask[n].split("/")[-1]:
#         print("False!!!!!!")
#     img_tmp=np.load(img[n])
#     img_tmp = img_tmp.astype("float16")
#     b = np.max(img_tmp)
#     mask_tmp=np.load(mask[n])
#     dataset=np.zeros((img_tmp.shape[0],img_tmp.shape[1]))
#
#     min_max=preprocessing.MinMaxScaler()
#     # img_tmp=min_max.fit_transform(img_tmp)*255
#
#     dataset[:,0:img_tmp.shape[1]]=mask_tmp*b
#     # dataset[:,img_tmp.shape[1]:]=mask_tmp*b
#     # dataset_img=cv2.imwrite("pytorch-CycleGAN-and-pix2pix-master//datasets//mydataset//test//{}.png".format(img[n].split("/")[-1].split(".")[0]),dataset)
#     imageio.imwrite("pytorch-CycleGAN-and-pix2pix-master//datasets//mydataset//test2//{}.png".format(img[n].split("/")[-1].split(".")[0]),dataset)
#
# img=glob("pytorch-CycleGAN-and-pix2pix-master//datasets//mydataset//test2//*")
#
#



# #计算旋转变换矩阵
# def handle_rotate_val(x,y,rotate):
#   cos_val = np.cos(np.deg2rad(rotate))
#   sin_val = np.sin(np.deg2rad(rotate))
#   return np.float32([
#       [cos_val, sin_val, x * (1 - cos_val) - y * sin_val],
#       [-sin_val, cos_val, x * sin_val + y * (1 - cos_val)]
#     ])
#
# # 图像旋转（以任意点为中心旋转）
# def image_rotate(src, rotate=0):
#   h,w,c = src.shape
#   M = handle_rotate_val(w//2,h//2,rotate)
#   img = cv2.warpAffine(src, M, (w,h))
#   return img
#
# def image_erode(img):
#     # 形态操作内核
#     k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     k2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
#     k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     # 形态操作内核
#     n_k1 = np.ones((5, 5), np.uint8)
#     n_k2 = np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]],
#                     dtype=np.uint8)
#     n_k3 = np.array([[0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0]],
#                     dtype=np.uint8)
#
#     # img1 = cv2.erode(img, k1, iterations=5)
#     img1 = cv2.dilate(img, n_k2, iterations=1)
#     img2 = cv2.erode(img, n_k1, iterations=5)
#     return img
#
# if __name__ == "__main__":
#     # img = glob("pytorch-CycleGAN-and-pix2pix-master//datasets//mydataset//test2//*")
#     img = glob("pytorch-CycleGAN-and-pix2pix-master//datasets//mydataset//pred//*")
#
#     for n in range(len(img)):
#         img_tmp = cv2.imread(img[n],cv2.IMREAD_GRAYSCALE)
#         dataset = np.zeros((img_tmp.shape[0], img_tmp.shape[1]*2))
#         dataset[:,0:img_tmp.shape[1]]=img_tmp
#         dataset[:,img_tmp.shape[1]:]=img_tmp
#         # img_new = image_rotate(img_tmp,-15)
#         cv2.imwrite("pytorch-CycleGAN-and-pix2pix-master//datasets//mydataset//results//{}.png".format(
#         img[n].split("/")[-1].split(".")[0]), dataset)
