import argparse
import os



def Pred_args():
    parser = argparse.ArgumentParser(description="Options of Pred Process")
    # 文件路径设置
    parser.add_argument('--dataroot', default="Dataset//patchs_test")
    parser.add_argument('--logroot', default="Logs")
    parser.add_argument('--time', default="1650263691.1693661",help="the name of the file which saved the best model parm")

    #架构设置
    parser.add_argument('--Net_Name', default="U2Net",help="UNet,U2Net,EINET")
    parser.add_argument('--Is_Class', default=False)

    # 参数设置
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--Label_Class', default=1)
    parser.add_argument('--inchannel', default=1)
    parser.add_argument('--outchannel', default=2)
    parser.add_argument('--img_size', default=112)
    parser.add_argument('--with_Mask', default=False)
    parser.add_argument('--Save_Img', default=True)
    args = parser.parse_args()


    # 保存参数

    return args