from glob import glob
from torch.utils.data import Dataset
import os
import SimpleITK as sitk
import numpy as np


class Dataset_with_Mask(Dataset):
    def __init__(self, dataset_name,Log_PATH):
        ################### Load DATASET ROOT PATH ###############
        DATASET_Root_Path = dataset_name
        print("\n------------------------------------------------------")
        print("\n The current Dataset name is : {}.".format(dataset_name.split("//")[-1]))

        ################### Load DATASET TRAIN AND TEST PATH ###############

        self.Masks_PATHS = sorted(glob(os.path.join(DATASET_Root_Path, "mask","*")))
        self.Images_PATHS = sorted(glob(os.path.join(DATASET_Root_Path, "image", "*")))

        print("\n There are {} cases in the Dataset.".format(len(self.Masks_PATHS)))

        with open(os.path.join(Log_PATH, 'setting.txt'), "a+") as f:
            f.writelines( "\n")
            f.writelines("The current Dataset name is : {}.".format(dataset_name.split("//")[-1]))
            f.writelines('There are {} cases in the Dataset.'.format(len(self.Masks_PATHS)) + "\n")

        print("\n------------------------------------------------------")

    def __len__(self):
        return len(self.Masks_PATHS)

    def __getitem__(self, idx):
        mask_path=self.Masks_PATHS[idx]
        image_path=self.Images_PATHS[idx]

        mask=np.expand_dims(np.load(mask_path),0).astype(np.float)
        image=np.expand_dims(np.load(image_path),0).astype(np.float)

        return image,mask

class Dataset_with_Mask_Pred(Dataset):
    def __init__(self, dataset_name,Log_PATH):
        ################### Load DATASET ROOT PATH ###############
        DATASET_Root_Path = dataset_name
        print("\n------------------------------------------------------")
        print("\n The current Dataset name is : {}.".format(dataset_name.split("//")[-1]))

        ################### Load DATASET TRAIN AND TEST PATH ###############

        self.Masks_PATHS = sorted(glob(os.path.join(DATASET_Root_Path, "mask","*")))
        self.Images_PATHS = sorted(glob(os.path.join(DATASET_Root_Path, "image", "*")))

        print("\n There are {} cases in the Dataset.".format(len(self.Masks_PATHS)))

        with open(os.path.join(Log_PATH, 'setting.txt'), "a+") as f:
            f.writelines( "\n")
            f.writelines("The current Dataset name is : {}.".format(dataset_name.split("//")[-1]))
            f.writelines('There are {} cases in the Dataset.'.format(len(self.Masks_PATHS)) + "\n")

        print("\n------------------------------------------------------")

    def __len__(self):
        return len(self.Masks_PATHS)

    def __getitem__(self, idx):
        mask_path=self.Masks_PATHS[idx]
        image_path=self.Images_PATHS[idx]

        mask=np.expand_dims(np.load(mask_path),0).astype(np.float)
        image=np.expand_dims(np.load(image_path),0).astype(np.float)

        return image,mask,image_path





