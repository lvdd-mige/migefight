from torch.autograd import Variable
from LOSS.Loss_Function import *
from Data.data_loader import *
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from Options import Train_Options
import numpy as np
import os
from Data import Load_Data,Data_process
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from Model import Unet

if __name__ == '__main__':
    Time = str(time.time())
    args = Train_Options.Train_args(Time)

    Dataset = Load_Data.Dataset_with_Mask(args.dataroot)
    # Train_Loader = DataLoader(Dataset, batch_size=args.batch_size, shuffle=True)
    #
    # Model_name = args.Net_Name
    # if Model_name == "U2Net":
    #     net = U2NET(2, args.Label_Class)
    # elif Model_name == "UNet":
    #     net = Unet(args.Label_Class)
    #
    # if torch.cuda.is_available():
    #     net.cuda()
    #
    # if args.optimizer_Name == "SGD":
    #     optimizer = optim.SGD(net.parameters(), lr=0.01)
    # elif args.optimizer_Name == "Adam":
    #     optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # Acc_Best = 0
    # for epoch in range(0, 300):
    #     net.train()
    #     correct = torch.zeros(1).squeeze().cuda()
    #     total = torch.zeros(1).squeeze().cuda()
    #     Seg_Loss_Results = []
    #
    #     for i, (inputs, labels, Class_Label, name) in enumerate(Train_Loader):
    #
    #         inputs = inputs.type(torch.FloatTensor)
    #         labels = labels.type(torch.LongTensor).unsqueeze(1)
    #         # Class_Label = Class_Label.type(torch.LongTensor)
    #
    #         # wrap them in Variable
    #         if torch.cuda.is_available():
    #             inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
    #                                                                                         requires_grad=False)
    #         else:
    #             inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
    #
    #         # y zero the parameter gradients
    #         optimizer.zero_grad()
    #         # if torch.max(labels_v) == 1:
    #         #     labels_v = torch.nn.functional.one_hot(labels_v).transpose(1, 3).transpose(2, 3)
    #         # elif torch.max(labels_v) == 0:
    #         #     labels_v = Variable(
    #         #         torch.zeros(labels_v.shape[0], args.Label_Class, labels_v.shape[1], labels_v.shape[2]).type(
    #         #             torch.LongTensor).cuda(), requires_grad=False)
    #         #     labels_v[:,0,:,:]=1
    #
    #         if Model_name == "U2Net":
    #             Seg_out,d1,d2,d3,d4,d5,d6=net(inputs_v)
    #             Seg_loss=muti_dice_loss_fusion(Seg_out,d1,d2,d3,d4,d5,d6,labels_v)
    #         elif Model_name=="UNet":
    #             Seg_out = net(inputs_v)
    #             Seg_loss = dice_loss(Seg_out, labels_v)
    #
    #
    #         # print(name,Seg_out.shape,labels_v.shape)
    #
    #         Seg_loss.backward()
    #         optimizer.step()
    #         Seg_Loss_Results.append(Seg_loss.data.item())
    #
    #         Pred_Img=Seg_out.clone()
    #         Pred_Img[Pred_Img <= 0.5] = 0
    #         Pred_Img[Pred_Img > 0.5] = 1
    #         # a=Data_process.remove_small_points(Pred_Img[0],8)
    #
    #         Class_out = np.zeros(Class_Label.shape)
    #         for n in range(Class_Label.shape[0]):
    #             # print(np.unique(Pred_Img[n][0]),np.unique(Pred_Img[n][1]),np.unique(labels_v[n][0].cpu().detach().numpy()),np.unique(labels_v[n][1].cpu().detach().numpy()))
    #             Tmp=Pred_Img[n,0,:,:,].cpu().detach().numpy()
    #             # print(np.unique(Tmp))
    #
    #             Tmp=Data_process.remove_small_points(Tmp, 1000)
    #             if len(np.unique(Tmp)) == 1:
    #                 Class_out[n] = 0
    #             elif len(np.unique(Tmp)) != 1:
    #                 Class_out[n] = 1
    #             Save_png_Img_Tmp(Tmp, "Tmp_0\\" + str(epoch) + "_" + name[0])
    #             Save_png_Img_Tmp(Seg_out[n][0].cpu().detach().numpy(), "Tmp_1\\" + str(epoch) + "_1_" + name[0])
    #             Save_png_Img_Tmp(labels_v[n][0].cpu().detach().numpy(), "Tmp_2\\" + str(epoch) + "_2_" + name[0])
    #         correct += (Class_out == Class_Label.cpu().detach().numpy()).sum()
    #         total += len(Class_Label)
    #     Save_png_Img(Seg_out, "Train_" + str(epoch) + "_" + name[0])
    #     Save_png_Img(labels_v, "Train_" + str(epoch) + "_GT_" + name[0])
    #     print("Epoch:", epoch, "SEG_Loss:", np.mean(Seg_Loss_Results), "ACC:",
    #           correct / total)
    #     if (correct / total) > Acc_Best:
    #         print("Update Best!")
    #         Acc_Best = correct / total
    #
    #         torch.save(net.state_dict(), os.path.join(args.logroot, Time, "best.pth"))
    #
    #         net.train()
    #
    #         # if Class_out==Class_Label:
    #         # correct+=
    #
    #         # forward + backward + optimize
    #         # d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
    #         # loss2, loss = muti_dice_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
    #         # Clas_loss = CE_Loss(Class_Results, Class_Label.squeeze(1))
    #         # Loss = loss + Clas_loss * 100
    #         # Loss.backward()
    #         # prediction = torch.argmax(Class_Results, 1)
    #         # # Class_Results[Class_Results < 0.5] = 0
    #         # Class_Results[Class_Results > 0.5] = 1
    #         # correct += (prediction == Class_Label.squeeze(1)).sum().float()
    #         # total += len(Class_Label)
    #         # Total_Loss.append(Loss.data.item())
    #
    #     #     Pred_Image = Seg_out
    #     #
        #     optimizer.step()
        #
        #     # # print statistics
        #     Train_Loss.append(loss2.data.item())
        #
        #     # del temporary outputs and loss
        #     del d0, d1, d2, d3, d4, d5, d6, loss2, loss
        # Save_png_Img(Pred_Image, "Train_" + str(epoch) + "_" + name[0])
        # Save_png_Img(labels_v, "Train_" + str(epoch) + "_GT_" + name[0])
        # print("Epoch:", epoch, "SEG_Loss:", np.mean(Train_Loss), "Total Loss", np.mean(Total_Loss), "ACC:",
        #       correct / total)
        # if (correct / total) > Acc_Best:
        #     print("Update Best!")
        #     Acc_Best = correct / total
        #
        #     torch.save(net.state_dict(), os.path.join(Logs_path, "best.pth"))
        #
        #     net.train()
