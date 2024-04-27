import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from Options import Train_Options
from Data import Load_Data
from torch.utils.data import DataLoader
from Model import Unet, U2NET
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from LOSS import Loss_Function
import SimpleITK as sitk




if __name__ == '__main__':
    Struct_time = time.localtime()
    time = str(Struct_time.tm_year) + "." + str(Struct_time.tm_mon) + "." + str(Struct_time.tm_mday) + "." + str(
        Struct_time.tm_hour) + "." + str(Struct_time.tm_min)+ "." + str(Struct_time.tm_sec)
    print(time)

    args, Log_PATH = Train_Options.Train_args(time)

    Train_Dataset = Load_Data.Dataset_with_Mask_Pred(args.train_dataroot, Log_PATH)
    Test_Dataset = Load_Data.Dataset_with_Mask_Pred(args.test_dataroot, Log_PATH)

    Train_Loader = DataLoader(Train_Dataset, batch_size=args.pred_batch_size, shuffle=False)
    Test_Loader = DataLoader(Test_Dataset, batch_size=args.pred_batch_size, shuffle=False)

    Model_name = args.Net_Name

    if Model_name == "Unet":
        net = Unet(args.Label_Class)
    elif Model_name == "U2Net":
        net = U2NET(args.inchannel, args.outchannel)
        net.load_state_dict(torch.load(r"Logs/U2Net/2024.4.20.17.6.9/best.pth"))

    if torch.cuda.is_available():
        net.cuda()

    if args.optimizer_Name == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=0.1)
    elif args.optimizer_Name == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    Best = 999

    net.eval()

    Pred_Dice = []
    y = 0
    for i, (inputs, labels, name) in enumerate(Test_Loader):

        patch = name[0].split("\\")[-1].split("_")[1].split(".")[0]
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        if Model_name == "Unet":
            Seg_out = net(inputs_v)
            Seg_loss = Loss_Function.dice_loss(Seg_out, labels_v)
            Pred_Dice.append(Seg_loss.item())
        elif Model_name == "U2Net":
            Seg_out, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            Seg_loss, loss = Loss_Function.muti_dice_loss_fusion(Seg_out, d1, d2, d3, d4, d5, d6, labels_v)
            Pred_Dice.append(loss.item())
            Seg_out[Seg_out < 0.5] = 0
            Seg_out[Seg_out >= 0.5] = 1

    print("Mean Dice is：",np.mean(Pred_Dice))
