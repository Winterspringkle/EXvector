import torch
import torch.nn as nn
from functions import ReverseLayerF


class CrossNetwork(nn.Module):
    """
    Cross Network
    """

    def __init__(self, layer_num, input_dim):
        super(CrossNetwork, self).__init__()
        self.layer_num = layer_num

        # 定义网络层的参数   221*3       三个
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.rand(input_dim, 1))
            for i in range(self.layer_num)
        ])
        self.cross_bias = nn.ParameterList([
            nn.Parameter(torch.rand(input_dim, 1))
            for i in range(self.layer_num)
        ])

    def forward(self, x):
        # x是(None, dim)的形状， 先扩展一个维度到(None, dim, 1)
        x_0 = torch.unsqueeze(x, dim=2)
        x = x_0.clone()  # 32*221*1
        xT = x_0.clone().permute((0, 2, 1))  # （None, 1, dim)  32*1*221
        for i in range(self.layer_num):
            x = torch.matmul(torch.bmm(x_0, xT), self.cross_weights[i]) + self.cross_bias[i] + x  # (None, dim, 1)32*221*1            bmm（32*221*1，32*1*221）， W=221*1， b=221*1
            xT = x.clone().permute((0, 2, 1))  # (None, 1, dim)

        x = torch.squeeze(x)  # (None, dim) 32*221  再降维
        return x

class EXvector(nn.Module):

    def __init__(self):
        super(EXvector, self).__init__()
        self.f_fc1=nn.Linear(512,1200)
        self.f_bn1= nn.BatchNorm1d(1200)
        self.f_relu1= nn.ReLU(True)
        self.f_drop1= nn.Dropout()

        self.cross = CrossNetwork(1, 1200)
        self.f_fc2= nn.Linear(1200, 1200)
        self.f_bn2= nn.BatchNorm1d(1200)
        self.f_relu2= nn.ReLU(True)
        self.f_drop2=nn.Dropout()

        self.f_fc3= nn.Linear(1200, 1200)
        self.f_bn3= nn.BatchNorm1d(1200)
        self.f_relu3= nn.ReLU(True)
        self.f_drop3= nn.Dropout()

        self.f_fc4= nn.Linear(1200, 1200)
        self.f_bn4= nn.BatchNorm1d(1200)
        self.f_relu4= nn.ReLU(True)
        self.f_drop4= nn.Dropout()

        self.f_fc5= nn.Linear(2400, 500)
        self.f_bn5= nn.BatchNorm1d(500)
        self.f_relu5= nn.ReLU(True)
        self.f_drop5= nn.Dropout()

        self.c_fc1= nn.Linear(500, 500)
        self.c_bn1= nn.BatchNorm1d(500)
        self.beta = nn.Sequential(nn.Linear(500, 500), nn.Tanh())
        self.c_relu1= nn.ReLU(True)
        self.c_drop1= nn.Dropout()
        self.c_fc2= nn.Linear(500, 500)
        self.beta2 = nn.Sequential(nn.Linear(500, 500), nn.Tanh())
        self.c_bn2= nn.BatchNorm1d(500)
        self.c_relu2= nn.ReLU(True)
        self.c_drop2= nn.Dropout()
        self.c_fc3= nn.Linear(500, 68)
        self.c_softmax= nn.LogSoftmax(dim=1)


        self.d_fc1= nn.Linear(500, 500)
        self.d_bn1= nn.BatchNorm1d(500)
        self.alpha= nn.Sequential(nn.Linear(500, 500),nn.Tanh())
        self.d_relu1= nn.ReLU(True)
        self.d_drop1= nn.Dropout()
        self.d_fc2= nn.Linear(500, 500)
        self.d_bn2= nn.BatchNorm1d(500)
        self.d_relu2= nn.ReLU(True)
        self.d_drop2= nn.Dropout()
        self.d_fc3= nn.Linear(500, 5)
        self.d_softmax= nn.LogSoftmax(dim=1)

    def forward(self, input_data, alpha):
        # input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        x=self.f_fc1(input_data)
        x=self.f_bn1(x)
        x=self.f_relu1(x)
        x1=self.f_drop1(x)
        x=self.f_fc2(x1)
        x_c=self.cross(x)
        # x_a = self.tanh3(self.attention1(x1))
        # x = x * x_a
        # x=x.view([-1,1,1200])
        # x,_=self.attention1(x,x,x)
        # x = x.view([-1,1200])
        x=self.f_bn2(x)
        x=self.f_relu2(x)
        x=self.f_drop2(x)
        x2=x+x1
        x=self.f_fc3(x2)
        # x_a=self.tanh2(self.attention2(x2))
        # x=x*x_a
        # x = x.view([-1, 1, 1200])
        # x,_=self.attention2(x,x,x)
        # x = x.view([-1, 1200])
        x=self.f_bn3(x)
        x=self.f_relu3(x)
        x=self.f_drop3(x)
        x3=x+x2
        x=self.f_fc4(x3)
        # x_a = self.tanh3(self.attention3(x3))
        # x = x * x_a
        # x = x.view([-1, 1, 1200])
        # x,_=self.attention3(x,x,x)
        # x = x.view([-1, 1200])
        x=self.f_bn4(x)
        x=self.f_relu4(x)
        x=self.f_drop4(x)
        x=torch.cat([x, x_c], axis=-1)
        x=self.f_fc5(x)
        x=self.f_bn5(x)
        x=self.f_relu5(x)
        x=self.f_drop5(x)
        feature = x.view(-1, 500)
        reverse_feature = ReverseLayerF.apply(feature, alpha)

        x=self.d_fc1(reverse_feature)
        x=self.d_bn1(x)
        x_wa=self.alpha(x)
        x_a=x*x_wa
        # x_b=x*(1-x_w)
        x=self.d_relu1(x_a)
        x=self.d_drop1(x)
        x=self.d_fc2(x)
        x=self.d_bn2(x)
        x=self.d_relu2(x)
        x=self.d_drop2(x)
        x=self.d_fc3(x)
        domain_output=self.d_softmax(x)

        # print("feature",feature.shape,"x_a",x_a.shape)
        x_wb=self.beta(feature)
        x = self.c_fc1(feature*x_wa)
        x=x+x*x_wb
        x = self.c_bn1(x)
        x = self.c_relu1(x)
        x = self.c_drop1(x)
        x_wb2=self.beta2(x)
        x = self.c_fc2(x)
        x=x+x*x_wb2
        x = self.c_bn2(x)
        x = self.c_relu2(x)
        x = self.c_drop2(x)
        x = self.c_fc3(x)
        class_output = self.c_softmax(x)

        return class_output, domain_output
