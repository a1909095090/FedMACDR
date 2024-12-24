# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, emb_size):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(emb_size * 2, 1)

    def forward(self, input1, input2):
        if input1.dim() <= input2.dim():
            smaller, larger = input1, input2
        else:
            smaller, larger = input2, input1
        if input1.dim() != input2.dim():
            if smaller.dim() == 2:
                # (batch_size, 1, emb_size)
                smaller = smaller.view(smaller.size()[0], 1, -1) #将smaller 的维度从（smaller.size()[0], smaller.size()[1] ）转化为： smaller.size()[0],1, smaller.size()[1]
                # (batch_size, num_neg, emb_size) in training mode,
                # (batch_size, num_test_neg + 1, emb_size) in evaluation mode
                smaller = smaller.repeat(1, larger.size()[1], 1) #repeat() 方法用于重复张量中的元素以形成一个新的张量。这个方法允许你指定每个维度的重复次数
                #将smaller的维度从（smaller.size()[0], 1,smaller.size()[1] ）转化为： smaller.size()[0], larger.size()[1], smaller.size()[1]

            # if `smaller` is graph representation `z_s`
            elif smaller.dim() == 1:
                smaller = smaller.view(1, -1)  # (1, emb_size)
                # (batch_size, emb_size)
                smaller = smaller.repeat(larger.size()[0], 1)

        input = torch.cat([smaller, larger], dim=-1)
        output = self.fc(input)
        return output
def get_score(input1,input2):

    if input1.dim() != input2.dim():
        #
        #     # (batch_size, 1, emb_size)
        # input1 = input1.view(input1.size()[0], 1,
        #                            -1)  # 将smaller 的维度从（smaller.size()[0], smaller.size()[1] ）转化为： smaller.size()[0],1, smaller.size()[1]
        #     # (batch_size, num_neg, emb_size) in training mode,
        #     # (batch_size, num_test_neg + 1, emb_size) in evaluation mode
        # input1 = input1.repeat(1, input2.size()[1], 1)  # repeat() 方法用于重复张量中的元素以形成一个新的张量。这个方法允许你指定每个维度的重复次数
        #     # 将smaller的维度从（smaller.size()[0], 1,smaller.size()[1] ）转化为： smaller.size()[0], larger.size()[1], smaller.size()[1]
        input1 = input1.unsqueeze(1)  # 形状变为[1024, 1, 128]
        output = torch.bmm(input1, input2.transpose(2, 1))  # 转置negv的最后两个维度

        # 结果的形状将是[1024, 1, 99]
        # 如果你想要移除中间的维度，可以使用squeeze
        output = output.squeeze(1)  # 形状变为[1024, 99]
    else:
        output=torch.mul(input1,input2)
    # gamma=torch.sum(output,dim=1)

    return output

class HingeLoss(nn.Module):
    def __init__(self, margin):
        super(HingeLoss, self).__init__()
        self.margin = margin

    def forward(self, pos, neg):
        pos = F.sigmoid(pos) # 1/(1+e^x)
        neg = F.sigmoid(neg)
        gamma = torch.tensor(self.margin).to(pos.device)
        return F.relu(gamma - pos + neg) #relu(x)=max(0,x)


class JSDLoss(torch.nn.Module):
    def __init__(self):
        super(JSDLoss, self).__init__()

    def forward(self, pos, neg):
        pos = -F.softplus(-pos) #Softplus(x)=log(1+e^x)
        neg = F.softplus(neg)
        return neg - pos    # log(1+e^(neg))+log(1+e^(-pos))


class BiDiscriminator(torch.nn.Module):
    def __init__(self, emb_size):
        super(BiDiscriminator, self).__init__()
        self.f_k = nn.Bilinear(emb_size, emb_size, 1)
        # 用于实现双线性层（BilinearLayer）。双线性层通常用于处理两个输入张量，并通过学习两个输入之间的交互来进行加权求和。
        # 双线性层的工作原理可以表示为：out=(x1×W1) ⋅(x2×W2)+b

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0) #将所以元素修改为0.0

    def forward(self, input1, input2, s_bias=None):
        if input1.dim() <= input2.dim():
            smaller, larger = input1, input2
        else:
            smaller, larger = input2, input1
        if input1.dim() != input2.dim():
            if smaller.dim() == 2:
                # (batch_size, 1, emb_size)
                smaller = smaller.view(smaller.size()[0], 1, -1)
                # (batch_size, num_neg, emb_size) in training mode,
                # (batch_size, num_test_neg + 1, emb_size) in evaluation mode
                smaller = smaller.repeat(1, larger.size()[1], 1)
            # if `smaller` is graph representation `z_s`
            elif smaller.dim() == 1:
                smaller = smaller.view(1, -1)  # (1, emb_size)
                # (batch_size, emb_size)
                smaller = smaller.repeat(larger.size()[0], 1)

        score = self.f_k(smaller, larger)
        if s_bias is not None:
            score += s_bias
        return score
