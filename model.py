# # 训练模型33333，通道数为16
# 导入 PyTorch 库，用于张量操作和神经网络构建
import torch
# 导入 PyTorch 的神经网络模块，用于定义层和模型
import torch.nn as nn
# 从 timm 库导入 DropPath 和 trunc_normal_，用于正则化和权重初始化
from timm.layers import DropPath, trunc_normal_
# 导入 torchsummary 的 summary 函数，用于显示模型结构详情
from torchsummary import summary

# 定义 ConvBN 类，继承自 torch.nn.Sequential，用于组合卷积和批量归一化操作
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        # 初始化父类 torch.nn.Sequential
        super().__init__()
        # 添加二维卷积层到序列容器
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        # 如果启用批量归一化，则添加并初始化批量归一化层
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            # 初始化批量归一化权重为 1
            torch.nn.init.constant_(self.bn.weight, 1)
            # 初始化批量归一化偏置为 0
            torch.nn.init.constant_(self.bn.bias, 0)

# 定义 Block 类，继承自 nn.Module，作为网络的基本构建模块
class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        # 初始化父类 nn.Module
        super().__init__()
        # 深度可分离卷积，带批量归一化，卷积核大小为7，填充以保持尺寸
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        # 1x1 卷积，扩展通道数，无批量归一化
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        # 另一个 1x1 卷积，扩展通道数，无批量归一化
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        # 1x1 卷积，缩减通道数，带批量归一化
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        # 第二个深度可分离卷积，无批量归一化
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        # ReLU6 激活函数，限制输出范围
        self.act = nn.ReLU6()
        # 如果 drop_path <= 0，使用 Identity，否则使用 DropPath 进行正则化
        self.drop_path = nn.Identity() if drop_path <= 0. else DropPath(drop_path)
    
    # 定义 Block 的前向传播
    def forward(self, x):
        # 保存输入，用于残差连接
        input = x
        # 应用第一个深度可分离卷积
        x = self.dwconv(x)
        # 分成两条路径，分别进行 1x1 卷积
        x1, x2 = self.f1(x), self.f2(x)
        # 对 x1 应用激活函数后与 x2 相乘（门控机制）
        x = self.act(x1) * x2
        # 缩减通道数并应用第二个深度可分离卷积
        x = self.dwconv2(self.g(x))
        # 添加残差连接并应用 DropPath
        x = input + self.drop_path(x)
        return x

# 定义 StarNet 类，继承自 nn.Module，作为主要的网络架构
class StarNet(nn.Module):
    def __init__(self, base_dim=16, depths=[3, 3, 12, 5, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1, **kwargs):
        # 初始化父类 nn.Module
        super().__init__()
        # 存储输出类别数
        self.num_classes = num_classes
        # 设置初始输入通道数为 16
        self.in_channel = 16
        # 茎层：卷积后接 ReLU6，用于初步特征提取
        self.stem = nn.Sequential(ConvBN(1, self.in_channel, kernel_size=3, stride=1, padding=1), nn.ReLU6())
        # 生成从 0 到 drop_path_rate 的线性分布的 DropPath 概率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        # 初始化 stages 为 ModuleList，用于存储网络的各个阶段
        self.stages = nn.ModuleList()
        # DropPath 概率索引计数器
        cur = 0
        # 构建每个阶段
        for i_layer in range(len(depths)):
            # 计算当前阶段的嵌入维度
            embed_dim = base_dim * 2 ** i_layer
            # 如果不是第一个阶段，使用卷积下采样，否则使用 Identity
            if i_layer > 0:
                down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            else:
                down_sampler = nn.Identity()
            # 更新输入通道数
            self.in_channel = embed_dim
            # 为当前阶段创建 Block 列表，使用对应的 DropPath 概率
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            # 更新 DropPath 索引
            cur += depths[i_layer]
            # 将下采样器和 Block 组合成一个序列层，添加到 stages
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # 上采样层，逐步增加空间分辨率并减少通道数
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 从 32x32 上采样到 64x64
            nn.Conv2d(128, 128, 3, 1, 1),                          # 卷积保持通道数
            nn.ReLU6(),                                            # 激活函数
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 从 64x64 上采样到 128x128
            nn.Conv2d(64, 64, 3, 1, 1),                            # 卷积保持通道数
            nn.ReLU6(),                                            # 激活函数
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # 从 128x128 上采样到 256x256
            nn.Conv2d(32, 32, 3, 1, 1),                            # 卷积保持通道数
            nn.ReLU6(),                                            # 激活函数
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),     # 从 256x256 上采样到 512x512，输出 1 通道
        )
        # 应用权重初始化
        self.apply(self._init_weights)

    # 定义权重初始化方法
    def _init_weights(self, m):
        # 如果是线性层或卷积层，初始化权重
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # 使用截断正态分布初始化权重，标准差为 0.02
            trunc_normal_(m.weight, std=.02)
            # 如果有偏置，初始化为 0
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # 如果是层归一化或批量归一化，初始化权重和偏置
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            # 初始化偏置为 0
            nn.init.constant_(m.bias, 0)
            # 初始化权重为 1
            nn.init.constant_(m.weight, 1.0)

    # 定义 StarNet 的前向传播
    def forward(self, x):
        # 通过茎层处理输入
        x = self.stem(x)
        # 依次通过每个阶段
        for stage in self.stages:
            x = stage(x)
        # 通过上采样层恢复分辨率
        x = self.upsampling(x)
        return x

# 主程序入口
if __name__ == "__main__":
    # 检查是否可用 GPU，若可用则使用 cuda，否则使用 cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化 StarNet 模型，设置 base_dim=16，并移动到指定设备
    model = StarNet(base_dim=16).to(device)
    # 打印模型结构摘要，输入尺寸为 (1, 512, 512)
    print(summary(model, (1, 512, 512)))
    # 打印使用的设备（cuda 或 cpu）
    print(device)









# # 训练模型33633，通道数为16
# # 导入 PyTorch 库，用于张量操作和神经网络构建
# import torch
# # 导入 PyTorch 的神经网络模块，用于定义层和模型
# import torch.nn as nn
# # 从 timm 库导入 DropPath 和 trunc_normal_，用于正则化和权重初始化
# from timm.layers import DropPath, trunc_normal_
# # 导入 torchsummary 的 summary 函数，用于显示模型结构详情
# from torchsummary import summary

# # 定义 ConvBN 类，继承自 torch.nn.Sequential，用于组合卷积和批量归一化操作
# class ConvBN(torch.nn.Sequential):
#     def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
#         # 初始化父类 torch.nn.Sequential
#         super().__init__()
#         # 添加二维卷积层到序列容器
#         self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
#         # 如果启用批量归一化，则添加并初始化批量归一化层
#         if with_bn:
#             self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
#             # 初始化批量归一化权重为 1
#             torch.nn.init.constant_(self.bn.weight, 1)
#             # 初始化批量归一化偏置为 0
#             torch.nn.init.constant_(self.bn.bias, 0)

# # 定义 Block 类，继承自 nn.Module，作为网络的基本构建模块
# class Block(nn.Module):
#     def __init__(self, dim, mlp_ratio=3, drop_path=0.):
#         # 初始化父类 nn.Module
#         super().__init__()
#         # 深度可分离卷积，带批量归一化，卷积核大小为7，填充以保持尺寸
#         self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
#         # 1x1 卷积，扩展通道数，无批量归一化
#         self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
#         # 另一个 1x1 卷积，扩展通道数，无批量归一化
#         self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
#         # 1x1 卷积，缩减通道数，带批量归一化
#         self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
#         # 第二个深度可分离卷积，无批量归一化
#         self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
#         # ReLU6 激活函数，限制输出范围
#         self.act = nn.ReLU6()
#         # 如果 drop_path <= 0，使用 Identity，否则使用 DropPath 进行正则化
#         self.drop_path = nn.Identity() if drop_path <= 0. else DropPath(drop_path)

#     # 定义 Block 的前向传播
#     def forward(self, x):
#         # 保存输入，用于残差连接
#         input = x
#         # 应用第一个深度可分离卷积
#         x = self.dwconv(x)
#         # 分成两条路径，分别进行 1x1 卷积
#         x1, x2 = self.f1(x), self.f2(x)
#         # 对 x1 应用激活函数后与 x2 相乘（门控机制）
#         x = self.act(x1) * x2
#         # 缩减通道数并应用第二个深度可分离卷积
#         x = self.dwconv2(self.g(x))
#         # 添加残差连接并应用 DropPath
#         x = input + self.drop_path(x)
#         return x

# # 定义 StarNet 类，继承自 nn.Module，作为主要的网络架构
# class StarNet(nn.Module):
#     def __init__(self, base_dim=16, depths=[3, 3, 6, 3, 3], mlp_ratio=4, drop_path_rate=0.0, num_classes=1, **kwargs):
#         # 初始化父类 nn.Module
#         super().__init__()
#         # 存储输出类别数
#         self.num_classes = num_classes
#         # 设置初始输入通道数为 16
#         self.in_channel = 16
#         # 茎层：卷积后接 ReLU6，用于初步特征提取
#         self.stem = nn.Sequential(ConvBN(1, self.in_channel, kernel_size=3, stride=1, padding=1), nn.ReLU6())
#         # 生成从 0 到 drop_path_rate 的线性分布的 DropPath 概率
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#         # 初始化 stages 为 ModuleList，用于存储网络的各个阶段
#         self.stages = nn.ModuleList()
#         # DropPath 概率索引计数器
#         cur = 0
#         # 构建每个阶段
#         for i_layer in range(len(depths)):
#             # 计算当前阶段的嵌入维度
#             embed_dim = base_dim * 2 ** i_layer
#             # 如果不是第一个阶段，使用卷积下采样，否则使用 Identity
#             if i_layer > 0:
#                 down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
#             else:
#                 down_sampler = nn.Identity()
#             # 更新输入通道数
#             self.in_channel = embed_dim
#             # 为当前阶段创建 Block 列表，使用对应的 DropPath 概率
#             blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
#             # 更新 DropPath 索引
#             cur += depths[i_layer]
#             # 将下采样器和 Block 组合成一个序列层，添加到 stages
#             self.stages.append(nn.Sequential(down_sampler, *blocks))
#         # 上采样层，逐步增加空间分辨率并减少通道数
#         self.upsampling = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 从 32x32 上采样到 64x64
#             nn.Conv2d(128, 128, 3, 1, 1),                          # 卷积保持通道数
#             nn.ReLU6(),                                            # 激活函数
#             nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 从 64x64 上采样到 128x128
#             nn.Conv2d(64, 64, 3, 1, 1),                            # 卷积保持通道数
#             nn.ReLU6(),                                            # 激活函数
#             nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # 从 128x128 上采样到 256x256
#             nn.Conv2d(32, 32, 3, 1, 1),                            # 卷积保持通道数
#             nn.ReLU6(),                                            # 激活函数
#             nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),     # 从 256x256 上采样到 512x512，输出 1 通道
#         )
#         # 应用权重初始化
#         self.apply(self._init_weights)

#     # 定义权重初始化方法
#     def _init_weights(self, m):
#         # 如果是线性层或卷积层，初始化权重
#         if isinstance(m, (nn.Linear, nn.Conv2d)):
#             # 使用截断正态分布初始化权重，标准差为 0.02
#             trunc_normal_(m.weight, std=.02)
#             # 如果有偏置，初始化为 0
#             if hasattr(m, 'bias') and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         # 如果是层归一化或批量归一化，初始化权重和偏置
#         elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
#             # 初始化偏置为 0
#             nn.init.constant_(m.bias, 0)
#             # 初始化权重为 1
#             nn.init.constant_(m.weight, 1.0)

#     # 定义 StarNet 的前向传播
#     def forward(self, x):
#         # 通过茎层处理输入
#         x = self.stem(x)
#         # 依次通过每个阶段
#         for stage in self.stages:
#             x = stage(x)
#         # 通过上采样层恢复分辨率
#         x = self.upsampling(x)
#         return x

# # 主程序入口
# if __name__ == "__main__":
#     # 检查是否可用 GPU，若可用则使用 cuda，否则使用 cpu
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # 实例化 StarNet 模型，设置 base_dim=16，并移动到指定设备
#     model = StarNet(base_dim=16).to(device)
#     # 打印模型结构摘要，输入尺寸为 (1, 512, 512)
#     print(summary(model, (1, 512, 512)))
#     # 打印使用的设备（cuda 或 cpu）
#     print(device)








# # # 训练模型33-12-55，通道数为16
# # 导入 PyTorch 库，用于张量操作和神经网络构建
# import torch
# # 导入 PyTorch 的神经网络模块，用于定义层和模型
# import torch.nn as nn
# # 从 timm 库导入 DropPath 和 trunc_normal_，用于正则化和权重初始化
# from timm.layers import DropPath, trunc_normal_
# # 导入 torchsummary 的 summary 函数，用于显示模型结构详情
# from torchsummary import summary

# # 定义 ConvBN 类，继承自 torch.nn.Sequential，用于组合卷积和批量归一化操作
# class ConvBN(torch.nn.Sequential):
#     def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
#         # 初始化父类 torch.nn.Sequential
#         super().__init__()
#         # 添加二维卷积层到序列容器
#         self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
#         # 如果启用批量归一化，则添加并初始化批量归一化层
#         if with_bn:
#             self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
#             # 初始化批量归一化权重为 1
#             torch.nn.init.constant_(self.bn.weight, 1)
#             # 初始化批量归一化偏置为 0
#             torch.nn.init.constant_(self.bn.bias, 0)

# # 定义 Block 类，继承自 nn.Module，作为网络的基本构建模块
# class Block(nn.Module):
#     def __init__(self, dim, mlp_ratio=3, drop_path=0.):
#         # 初始化父类 nn.Module
#         super().__init__()
#         # 深度可分离卷积，带批量归一化，卷积核大小为7，填充以保持尺寸
#         self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
#         # 1x1 卷积，扩展通道数，无批量归一化
#         self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
#         # 另一个 1x1 卷积，扩展通道数，无批量归一化
#         self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
#         # 1x1 卷积，缩减通道数，带批量归一化
#         self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
#         # 第二个深度可分离卷积，无批量归一化
#         self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
#         # ReLU6 激活函数，限制输出范围
#         self.act = nn.ReLU6()
#         # 如果 drop_path <= 0，使用 Identity，否则使用 DropPath 进行正则化
#         self.drop_path = nn.Identity() if drop_path <= 0. else DropPath(drop_path)
    
#     # 定义 Block 的前向传播
#     def forward(self, x):
#         # 保存输入，用于残差连接
#         input = x
#         # 应用第一个深度可分离卷积
#         x = self.dwconv(x)
#         # 分成两条路径，分别进行 1x1 卷积
#         x1, x2 = self.f1(x), self.f2(x)
#         # 对 x1 应用激活函数后与 x2 相乘（门控机制）
#         x = self.act(x1) * x2
#         # 缩减通道数并应用第二个深度可分离卷积
#         x = self.dwconv2(self.g(x))
#         # 添加残差连接并应用 DropPath
#         x = input + self.drop_path(x)
#         return x

# # 定义 StarNet 类，继承自 nn.Module，作为主要的网络架构
# class StarNet(nn.Module):
#     def __init__(self, base_dim=16, depths=[3, 3, 12, 5, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1, **kwargs):
#         # 初始化父类 nn.Module
#         super().__init__()
#         # 存储输出类别数
#         self.num_classes = num_classes
#         # 设置初始输入通道数为 16
#         self.in_channel = 16
#         # 茎层：卷积后接 ReLU6，用于初步特征提取
#         self.stem = nn.Sequential(ConvBN(1, self.in_channel, kernel_size=3, stride=1, padding=1), nn.ReLU6())
#         # 生成从 0 到 drop_path_rate 的线性分布的 DropPath 概率
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#         # 初始化 stages 为 ModuleList，用于存储网络的各个阶段
#         self.stages = nn.ModuleList()
#         # DropPath 概率索引计数器
#         cur = 0
#         # 构建每个阶段
#         for i_layer in range(len(depths)):
#             # 计算当前阶段的嵌入维度
#             embed_dim = base_dim * 2 ** i_layer
#             # 如果不是第一个阶段，使用卷积下采样，否则使用 Identity
#             if i_layer > 0:
#                 down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
#             else:
#                 down_sampler = nn.Identity()
#             # 更新输入通道数
#             self.in_channel = embed_dim
#             # 为当前阶段创建 Block 列表，使用对应的 DropPath 概率
#             blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
#             # 更新 DropPath 索引
#             cur += depths[i_layer]
#             # 将下采样器和 Block 组合成一个序列层，添加到 stages
#             self.stages.append(nn.Sequential(down_sampler, *blocks))
#         # 上采样层，逐步增加空间分辨率并减少通道数
#         self.upsampling = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 从 32x32 上采样到 64x64
#             nn.Conv2d(128, 128, 3, 1, 1),                          # 卷积保持通道数
#             nn.ReLU6(),                                            # 激活函数
#             nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 从 64x64 上采样到 128x128
#             nn.Conv2d(64, 64, 3, 1, 1),                            # 卷积保持通道数
#             nn.ReLU6(),                                            # 激活函数
#             nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # 从 128x128 上采样到 256x256
#             nn.Conv2d(32, 32, 3, 1, 1),                            # 卷积保持通道数
#             nn.ReLU6(),                                            # 激活函数
#             nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),     # 从 256x256 上采样到 512x512，输出 1 通道
#         )
#         # 应用权重初始化
#         self.apply(self._init_weights)

#     # 定义权重初始化方法
#     def _init_weights(self, m):
#         # 如果是线性层或卷积层，初始化权重
#         if isinstance(m, (nn.Linear, nn.Conv2d)):
#             # 使用截断正态分布初始化权重，标准差为 0.02
#             trunc_normal_(m.weight, std=.02)
#             # 如果有偏置，初始化为 0
#             if hasattr(m, 'bias') and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         # 如果是层归一化或批量归一化，初始化权重和偏置
#         elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
#             # 初始化偏置为 0
#             nn.init.constant_(m.bias, 0)
#             # 初始化权重为 1
#             nn.init.constant_(m.weight, 1.0)

#     # 定义 StarNet 的前向传播
#     def forward(self, x):
#         # 通过茎层处理输入
#         x = self.stem(x)
#         # 依次通过每个阶段
#         for stage in self.stages:
#             x = stage(x)
#         # 通过上采样层恢复分辨率
#         x = self.upsampling(x)
#         return x

# # 主程序入口
# if __name__ == "__main__":
#     # 检查是否可用 GPU，若可用则使用 cuda，否则使用 cpu
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # 实例化 StarNet 模型，设置 base_dim=16，并移动到指定设备
#     model = StarNet(base_dim=16).to(device)
#     # 打印模型结构摘要，输入尺寸为 (1, 512, 512)
#     print(summary(model, (1, 512, 512)))
#     # 打印使用的设备（cuda 或 cpu）
#     print(device)