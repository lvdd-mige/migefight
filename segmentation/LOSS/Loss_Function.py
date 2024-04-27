import torch.nn as nn
import torch.nn.functional as F
import torch

# ------- 1. define BCE function --------

bce_loss = nn.BCELoss(size_average=True)


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss

# ------- 2. define DICE function --------

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, pred, targets):
        num = targets.size(0)
        smooth = 1

        # probs = F.sigmoid(logits)
        m1 = pred.reshape(num, -1)
        m2 = targets.reshape(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


# dice_loss = SoftDiceLoss()
class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        # pred = pred.squeeze(dim=1)

        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1) +
                                                target[:,i].pow(2).sum(dim=1).sum(dim=1) + smooth)
        # 返回的是dice距离
        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 1)


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        # pred = pred.squeeze(dim=1)

        smooth = 1

        dice = 0.
        # dice系数的定义
        for i in range(pred.size(1)):
            dice += 2 * (pred[:,i] * target[:,i]).sum(dim=1).sum(dim=1) / (pred[:,i].pow(2).sum(dim=1).sum(dim=1) +
                                                target[:,i].pow(2).sum(dim=1).sum(dim=1) + smooth)
        # 返回的是dice距离
        dice = dice / pred.size(1)
        return torch.clamp((1 - dice).mean(), 0, 1)


def dice_loss(pred, target):
    smooth = 1
    num = pred.size(0)
    m1 = pred.view(num, -1)
    m2 = target.view(num, -1)

    interesection = (m1 * m2).sum()

    return 1-((2. * interesection + smooth) / (m1.sum() + m2.sum() + smooth))

def muti_dice_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = dice_loss (d0, labels_v)
    loss1 = dice_loss(d1, labels_v)
    loss2 = dice_loss(d2, labels_v)
    loss3 = dice_loss(d3, labels_v)
    loss4 = dice_loss(d4, labels_v)
    loss5 = dice_loss(d5, labels_v)
    loss6 = dice_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss,loss0