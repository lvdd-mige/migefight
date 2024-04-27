from glob import glob
import SimpleITK as sitk
import numpy as np

img_list=glob("public-AD//public-train//*//image.nii.gz")[:108]

for h in img_list:
    img_tmp = sitk.GetArrayFromImage(sitk.ReadImage(h))
    mask_tmp_name=h.replace("image","mask")
    mask_tmp=sitk.GetArrayFromImage(sitk.ReadImage(mask_tmp_name))
    mask_tmp[mask_tmp!=0]=1
    print(np.unique(mask_tmp))
    for j in range(mask_tmp.shape[0]):
        if len(np.unique(mask_tmp[j]))!=1:
            print(np.unique(mask_tmp[j]))
            # np.save("")
            print(h.split("\image")[0].split("\\")[-1],j)
            np.save(r"E:\CHY\Dataset\public-AD\train\mask\{}_{}.npy".format(h.split("\image")[0].split("\\")[-1],j),mask_tmp[j])
            np.save(r"E:\CHY\Dataset\public-AD\train\image\{}_{}.npy".format(h.split("\image")[0].split("\\")[-1], j),
                    img_tmp[j])

        else:
            print(j,np.unique(mask_tmp[j]))


img_list=glob("public-AD//public-train//*//image.nii.gz")[108:]

for h in img_list:
    img_tmp = sitk.GetArrayFromImage(sitk.ReadImage(h))
    mask_tmp_name=h.replace("image","mask")
    mask_tmp=sitk.GetArrayFromImage(sitk.ReadImage(mask_tmp_name))
    mask_tmp[mask_tmp!=0]=1
    print(np.unique(mask_tmp))
    for j in range(mask_tmp.shape[0]):
        # if len(np.unique(mask_tmp[j]))!=1:
        print(np.unique(mask_tmp[j]))
        # np.save("")
        print(h.split("\image")[0].split("\\")[-1],j)
        np.save(r"E:\CHY\Dataset\public-AD\test\mask\{}_{}.npy".format(h.split("\image")[0].split("\\")[-1],j),mask_tmp[j])
        np.save(r"E:\CHY\Dataset\public-AD\test\image\{}_{}.npy".format(h.split("\image")[0].split("\\")[-1], j),img_tmp[j])


