import SimpleITK as sitk
import numpy as np
from glob import glob
import os
# mask_cases = glob("train//*mask.nii.gz")
#
# for n in range(len(mask_cases)):
#     case_mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_cases[n]))
#     case_image = sitk.GetArrayFromImage(sitk.ReadImage(mask_cases[n].split("mask")[0] + ".nii.gz"))
#     case_name = mask_cases[n].split("\\")[-1].split("mask")[0]
#     print(case_mask.shape,case_image.shape)
#     for k in range(case_mask.shape[0]):
#         # print(np.unique(case_mask[k]))
#         print("all_patchs_train\\masks\\" + case_name +"_"+ str(k+100) + ".npy")
#         np.save("all_patchs_train\\masks\\" + case_name +"_"+ str(k+100) + ".npy", case_mask[k])
#         np.save("all_patchs_train\\images\\" + case_name +"_"+ str(k+100) + ".npy", case_image[k])
#     # print(np.unique(case))
#
mask_cases = glob("test//*mask.nii.gz")

for n in range(len(mask_cases)):
    case_mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_cases[n]))
    case_image = sitk.GetArrayFromImage(sitk.ReadImage(mask_cases[n].split("mask")[0] + ".nii.gz"))
    case_name = mask_cases[n].split("\\")[-1].split("mask")[0]
    for k in range(case_mask.shape[0]):
        # print(np.unique(case_mask[k]))
        np.save("all_patchs_test\\masks\\" + case_name +"_"+ str(k+100) + ".npy", case_mask[k])
        np.save("all_patchs_test\\images\\" + case_name +"_"+ str(k+100) + ".npy", case_image[k])

# import SimpleITK as sitk
import numpy as np
from glob import glob

mask_cases = glob("public-AD/train//*mask.nii.gz")

for n in range(len(mask_cases)):
    case_mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_cases[n]),)
    case_image = sitk.GetArrayFromImage(sitk.ReadImage(mask_cases[n].split("mask")[0] + ".nii.gz"))
    case_name = mask_cases[n].split("\\")[-1].split("mask")[0]
    for k in range(case_mask.shape[0]):
        # print(np.unique(case_mask[k]))
        if len(np.unique(case_mask[k]))!=1:
            np.save("patchs_train\\masks\\" + case_name +"_"+ str(k+100) + ".npy", case_mask[k])
            np.save("patchs_train\\images\\" + case_name +"_"+ str(k+100) +  ".npy", case_image[k])
    # print(np.unique(case))


mask_cases = glob("val//*mask.nii.gz")

for n in range(len(mask_cases)):
    case_mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_cases[n]))
    case_image = sitk.GetArrayFromImage(sitk.ReadImage(mask_cases[n].split("mask")[0] + ".nii.gz"))
    case_name = mask_cases[n].split("\\")[-1].split("mask")[0]
    for k in range(case_mask.shape[0]):
        # print(np.unique(case_mask[k]))
        if len(np.unique(case_mask[k])) != 1:
            np.save("patchs_val\\masks\\" + case_name +"_"+ str(k+100) + ".npy", case_mask[k])
            np.save("patchs_val\\images\\" + case_name +"_"+ str(k+100) + ".npy", case_image[k])

mask_cases = glob("test//*mask.nii.gz")

for n in range(len(mask_cases)):
    case_mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_cases[n]))
    case_image = sitk.GetArrayFromImage(sitk.ReadImage(mask_cases[n].split("mask")[0] + ".nii.gz"))
    case_name = mask_cases[n].split("\\")[-1].split("mask")[0]
    for k in range(case_mask.shape[0]):
        # print(np.unique(case_mask[k]))
        if len(np.unique(case_mask[k])) != 1:
            np.save("patchs_test\\masks\\" + case_name +"_"+ str(k+100) + ".npy", case_mask[k])
            np.save("patchs_test\\images\\" + case_name +"_"+ str(k+100) + ".npy", case_image[k])