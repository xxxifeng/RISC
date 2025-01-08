from torchvision.models import vgg16
import torch.nn as nn
import torch


# 获取vgg16模型中指定层之间的卷积层
def get_conv(start, end, model='vgg16'):
    conv1, conv2, conv3 = None, None, None  # 移除
    if model == 'vgg16':
        net = vgg16(pretrained=True)
        return net.features[start:end]

    return None


class BackBone(nn.Module):
    def __init__(self, in_features, out_features):
        super(BackBone, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

    def forward(self, x):
        return self.net(x)

# 提取多尺度特征
class Feature_Extraction(nn.Module):
    def __init__(self, args):
        super(Feature_Extraction, self).__init__()
        # 获取不同部分作为特征提取器
        self.layer1 = get_conv(0, 19, args.model)
        self.layer2 = get_conv(19, 26, args.model)
        self.layer3 = get_conv(26, 31, args.model)

        # 最大池化层
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=4, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, imgs):
        # 处理图像
        x1 = self.layer1(imgs)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        # 下采样并连接
        new_x1 = x3
        new_x2 = self.maxpool2(x2)
        new_x3 = self.maxpool4(x1)
        x = torch.cat([new_x1, new_x2, new_x3], dim=1)

        return x


class SAFF(nn.Module):
    def __init__(self, a=0.5, b=2, sigma=0.0001):
        super(SAFF, self).__init__()
        # 初始化参数
        self.a = a
        self.b = b
        self.sigma = sigma

    def forward(self, x):
        """
        :param x: (n, c, h, w)
        :return:
        """
        n, K, h, w = x.shape
        S = x.sum(dim=1)  # n,h,w  沿通道维度求和，得到每个位置上所有通道值之和
        z = torch.sum(S ** self.a, dim=[1, 2])  # 对S应用幂运算并求和
        z = (z ** (1 / self.a)).view(n, 1, 1)  # 计算归一化因子z
        S = (S / z) ** (1 / self.b)  # 获得注意力权重
        S = S.unsqueeze(1)  # 扩展维度
        new_x = (x * S).sum(dim=[2, 3])  # 加权，并沿空间维度求和
        omg = (x > 0).sum(dim=[2, 3]) / (256 ** 2)  # 计算非零元素的比例
        omg_sum = omg.sum(dim=1).unsqueeze(1)  # 求和并扩展维度
        omg = (K * self.sigma + omg_sum) / (self.sigma + omg)  # 调整比例
        omg = torch.log(omg)  # 计算通道k的权重
        x = omg * new_x  # 加权求和，得到最终特征表示
        return x

# 主模型类，整合特征提取和SAFF
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.feature_extract = Feature_Extraction(args)
        self.saff = SAFF()

    def forward(self, img):
        # 首先进行特征提取，再通过SAFF处理输入图像
        x = self.feature_extract(img)
        x = self.saff(x)
        return x
