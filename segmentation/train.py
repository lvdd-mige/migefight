import time
from Options import Train_Options
from Data import Load_Data
from torch.utils.data import DataLoader
from Model import Unet,U2NET
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from LOSS import Loss_Function
# import os
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':
    Struct_time = time.localtime()
    time=str(Struct_time.tm_year)+"."+str(Struct_time.tm_mon)+"."+str(Struct_time.tm_mday)+"."+str(Struct_time.tm_hour)+"."+str(Struct_time.tm_min)+"."+str(Struct_time.tm_sec)
    print(time)

    args,Log_PATH = Train_Options.Train_args(time)

    Train_Dataset = Load_Data.Dataset_with_Mask(args.train_dataroot,Log_PATH)
    Test_Dataset = Load_Data.Dataset_with_Mask(args.test_dataroot,Log_PATH)

    Train_Loader = DataLoader(Train_Dataset, batch_size=args.batch_size, shuffle=True)
    Test_Loader = DataLoader(Test_Dataset, batch_size=args.batch_size, shuffle=False)

    Model_name = args.Net_Name

    if Model_name == "Unet":
        net = Unet(args.Label_Class)
    elif Model_name=="U2Net":
        net=U2NET(args.inchannel,args.outchannel)
        # net.load_state_dict(torch.load("Logs/U2Net//2022.6.14.22.26/best.pth"))

    if torch.cuda.is_available():
        net.cuda()


    Best = 999
    T_num=0
    for epoch in range(0, args.epoch):
        if args.optimizer_Name == "SGD":
            optimizer = optim.SGD(net.parameters(), lr=args.lr)
        elif args.optimizer_Name == "Adam":
            optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        net.train()

        Train_Loss=[]

        for i, (inputs, labels) in enumerate(Train_Loader):

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            optimizer.zero_grad()


            if Model_name=="Unet":
                Seg_out = net(inputs_v)
                Seg_loss = Loss_Function.dice_loss(Seg_out, labels_v)
                Train_Loss.append(Seg_loss.item())
            elif Model_name=="U2Net":
                Seg_out,d1,d2,d3,d4,d5,d6 = net(inputs_v)
                Seg_loss,loss = Loss_Function.muti_dice_loss_fusion(Seg_out,d1,d2,d3,d4,d5,d6, labels_v)
                Train_Loss.append(loss.item())

            Seg_loss.backward()
            optimizer.step()

        print("Train_Epoch:", epoch, "SEG_Loss:", np.mean(Train_Loss))

        with open(os.path.join(Log_PATH, 'train_&_test.txt'), "a+") as f:
            f.writelines( "\n")
            f.writelines("Train_Epoch:  "+ str(epoch) +"  SEG_Loss:  "+str(np.mean(Train_Loss)))

        net.eval()

        Test_Loss = []

        for i, (inputs, labels) in enumerate(Test_Loader):

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
                Train_Loss.append(Seg_loss.item())
            elif Model_name == "U2Net":
                Seg_out, d1, d2, d3, d4, d5, d6 = net(inputs_v)
                Seg_loss, loss = Loss_Function.muti_dice_loss_fusion(Seg_out, d1, d2, d3, d4, d5, d6, labels_v)
                Test_Loss.append(loss.item())


        print("Test_Epoch:", epoch, "SEG_Loss:", np.mean(Test_Loss))

        with open(os.path.join(Log_PATH, 'train_&_test.txt'), "a+") as f:
            f.writelines("    Test_Epoch:  "+ str(epoch) +"  SEG_Loss:  "+str(np.mean(Test_Loss)))

        print("------------------------------------------------------")
        print(T_num)
        if np.mean(Test_Loss)<Best:
            T_num=0
            Best=np.mean(Test_Loss)
            print("Update and save")
            torch.save(net.state_dict(), os.path.join(Log_PATH, "best.pth"))
        else:
            T_num+=1
            if T_num==5:
                args.lr=args.lr/10
            elif T_num==10:
                print("Early Stop!")
                break