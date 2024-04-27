import argparse
import os


def Train_args(Time):
    parser = argparse.ArgumentParser(description="Options of Train Process")
    # 文件路径设置
    parser.add_argument('--train_dataroot', default=r"Dataset\public-AD\train")
    parser.add_argument('--test_dataroot', default=r"Dataset\public-AD\test")
    parser.add_argument('--logroot', default="Logs")


    #架构设置
    parser.add_argument('--Net_Name', default="U2Net",help="UNet,U2Net,EINET")
    parser.add_argument('--optimizer_Name', default="SGD", help="Adam,SGD")
    parser.add_argument('--Loss_Name', default="Dice_Loss", help="Dice_Loss,CE_Loss")
    parser.add_argument('--Is_Class', default=False)


    # 参数设置
    parser.add_argument('--batch_size', default=2)
    parser.add_argument('--lr', default=0.1)
    parser.add_argument('--pred_batch_size', default=1)
    parser.add_argument('--epoch', default=100)
    parser.add_argument('--Label_Class', default=1)

    parser.add_argument('--inchannel', default=1)
    parser.add_argument('--outchannel', default=1)
    parser.add_argument('--img_size', default=112)

    args = parser.parse_args()


    # 保存参数
    argsDict = args.__dict__
    if not os.path.exists(os.path.join(args.logroot,args.Net_Name)):
        os.mkdir(os.path.join(args.logroot,args.Net_Name))
    Log_PATH=os.path.join(args.logroot,args.Net_Name,Time)
    os.mkdir(Log_PATH)

    with open(os.path.join(Log_PATH,'setting.txt'), "a+") as f:
        f.writelines('--------------- Train_args --------------' + "\n")
        for eachArg, value in argsDict.items():
            f.writelines(eachArg+":"+str(value)+"\n")


    return args,Log_PATH

