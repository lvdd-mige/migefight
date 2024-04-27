import SimpleITK as sitk
import os
from glob import glob

p=sorted(glob("Dataset//test//*mask.nii.gz"))
g=sorted(glob("Pred//*"))

for n in range(len(p)):
    img=sitk.ReadImage(p[n])
    img1=sitk.ReadImage(g[n])

    img1.SetOrigin(img.GetOrigin())
    img1.SetDirection(img.GetDirection())
    img1.SetSpacing(img.GetSpacing())
    sitk.WriteImage(img1,g[n])
