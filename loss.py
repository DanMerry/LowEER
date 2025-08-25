import torch, math
import torch.nn as nn
import torch.nn.functional as F
from tools import *

class AAMsoftmax(nn.Module):
    def __init__(self, n_class, m, s):
        
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        # print(output)
        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1


class OriginalSoftmax(nn.Module):
    def __init__(self, n_class):
        super(OriginalSoftmax, self).__init__()

        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        self.output_layer = nn.Linear(192, n_class)
        nn.init.xavier_normal_(self.weight, gain=1)


    def forward(self, x, label=None):
        # cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        # phi = cosine * self.cos_m - sine * self.sin_m
        # phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        # one_hot = torch.zeros_like(cosine)
        # one_hot.scatter_(1, label.view(-1, 1), 1)
        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # output = output * self.s
        # print(output)
        output = self.output_layer(x)
        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1


class RangeLoss(nn.Module):

    def __init__(self, margin=0.3):  # 三元组的阈值margin
        super(RangeLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)  # 三元组损失函数
        # ap an margin y:倍率   Relu(ap - anxy + margin)这个relu就起到和0比较的作用

    def forward(self, anchor, positive, targets):
        """
        Args:
            inputs: visualization_feature_map matrix with shape (batch_size, feat_dim)#32x2048
            targets: ground truth labels with shape (num_classes)#tensor([32])[1,1,1,1,2,3,2,,,,2]32个数，一个数代表ID的真实标签
        """
        n = anchor.size(0)  # 取出输入的batch
        # Compute pairwise distance, replace by the official when merged
        # 计算距离矩阵，其实就是计算两个2048维之间的距离平方(a-b)**2=a^2+b^2-2ab
        # [1,2,3]*[1,2,3]=[1,4,9].sum()=14  点乘

        dist = torch.pow(anchor, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, anchor, anchor.t())  # 生成距离矩阵32x32，.t()表示转置
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability#clamp(min=1e-12)加这个防止矩阵中有0，对梯度下降不好
        # For each anchor, find the hardest positive and negative

        # print(dist.shape)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())  # 利用target标签的expand，并eq，获得mask的范围，由0，1组成，，红色1表示是同一个人，绿色0表示不是同一个人
        dist_ap, dist_an = [], []  # 用来存放ap，an
        for i in range(n):  # i表示行
            # dist[i][mask[i]],,i=0时，取mask的第一行，取距离矩阵的第一行，然后得到tensor([1.0000e-06, 1.0000e-06, 1.0000e-06, 1.0000e-06])
            # dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # 取某一行中，红色区域的最大值，mask前4个是1，与dist相乘
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # 取某一行，绿色区域的最小值,加一个.unsqueeze(0)将其变成带有维度的tensor

        # dist_pos = torch.pow(positive, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist_pos = dist_pos + dist_pos.t()
        # dist_pos.addmm_(1, -2, positive, positive.t())  # 生成距离矩阵32x32，.t()表示转置
        # dist_pos = dist_pos.clamp(min=1e-12).sqrt()
        dist_pos_2 = torch.pairwise_distance(anchor, positive, p=2)
        # print(dist_pos)
        # print(dist_pos_2)
        for i in range(n):
            dist_ap.append(dist_pos_2[i].unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)  # y是个权重，长度像dist-an
        loss = self.ranking_loss(dist_an, dist_ap, y)  # ID损失：交叉商输入的是32xf f.shape=分类数,然后loss用于计算损失
        # 度量三元组：输入的是dist_an（从距离矩阵中，挑出一行（即一个ID）的最大距离），dist_ap
        # ranking_loss输入 an ap margin y:倍率  loss： Relu(ap - anxy + margin)这个relu就起到和0比较的作用
        # from IPython import embed
        # embed()

        return loss
