import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image


class BRelu(nn.Module):
    def __init__(self):
        super(BRelu, self).__init__()
    
    def forward(self, x):
        return torch.clamp(x, 0, 1)


class DehazeNet(nn.Module):

    @staticmethod
    def Maxout(x, groups):
        """与原始代码完全相同的Maxout实现"""
        batch_size, channels, height, width = x.shape
        # 将通道分组
        x = x.reshape(batch_size, groups, channels // groups, height, width)
        # 在每个组内取最大值
        x, _ = torch.max(x, dim=2, keepdim=True)
        out = x.reshape(batch_size, groups, height, width)
        return out

    def __init__(self, input_channels=16, groups=4):
        '''注意：原始论文的MATLAB代码卷积使用SAME卷积，对称填充。为节省代码，采用SAME卷积、零填充'''
        super(DehazeNet, self).__init__()
        self.input_channels = input_channels
        self.groups = groups
        
        # Feature Extraction
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.input_channels, kernel_size=5, padding=2) # 和 https://github.com/thuBingo/DehazeNet_Pytorch 实现不同，使用SAME卷积
        # Multi-scale Mapping
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=7, padding=3)
        # Local Extremum
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1)
        # Non-linear Regression
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=6)
        self.brelu = BRelu() # 使用兼容的BRelu
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # 使用原始代码的初始化方式
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 保存输入尺寸
        orig_size = x.size()[2:]
        # Feature Extraction
        f1 = self.conv1(x)
        f1 = self.Maxout(f1, self.groups)
        # Multi-scale Mapping
        f2_3x3 = self.conv2(f1)
        f2_5x5 = self.conv3(f1)
        f2_7x7 = self.conv4(f1)
        f2 = torch.cat((f2_3x3, f2_5x5, f2_7x7), dim=1)
        # Local Extremum
        f3 = self.maxpool(f2)
        # Non-linear Regression
        f4 = self.conv5(f3)
        f4 = self.brelu(f4)
        # 使用双线性插值恢复到原始尺寸
        f4 = torch.nn.functional.interpolate(f4, size=orig_size, mode='bilinear', align_corners=False)
        # 返回全分辨率传输图
        return f4


class DehazeNetDriver:
    '''FIXME: 目前这一层类是否有存在的必要还存疑，需要等我熟悉更多深度学习模型再下结论'''

    def __init__(self, net=None):
        self.net = net or DehazeNet()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        self.net.eval()
    
    def open(self, filename='defog4_noaug.pth'):
        try:
            # 加载权重，注意兼容性处理
            state_dict = torch.load(filename, map_location=self.device)
            
            # 检查是否需要转换键名
            if 'conv1.weight' in state_dict:
                print('直接加载权重')
                self.net.load_state_dict(state_dict)
            else:
                print('尝试其他可能的键名格式')
                new_state_dict = {}
                for key, value in state_dict.items():
                    # 移除可能的模块前缀
                    new_key = key.replace('module.', '')
                    new_state_dict[new_key] = value
                
                # 尝试加载
                try:
                    self.net.load_state_dict(new_state_dict)
                except:
                    print("无法加载权重文件，使用随机初始化的权重")
            
            print(f"模型从 {filename} 加载成功")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("使用随机初始化的模型")
        
        return self
    
    def save(self, filename='defog4_noaug.pth'):
        torch.save(self.net.state_dict(), filename)
        print(f"模型保存到 {filename}")
        return self


class DehazeNetRunner:
    
    @staticmethod
    def transmission_map(img_np, driver):
        """
        直接处理整张图像，得到传输图
        
        Args:
            img_np: 输入图像，值在0-1之间
            driver: 模型驱动
        """
        # 论文原始代码的预处理：直接减去0.5；有转换函数来自 https://github.com/thuBingo/DehazeNet_Pytorch 移植代码
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()  # HWC -> CHW
        img_tensor = img_tensor - 0.5  # 原始代码的预处理

        # 添加batch维度
        img_tensor = img_tensor.unsqueeze(0).to(driver.device)
        
        with torch.no_grad():
            t_map = driver.net(img_tensor)
            t_map = t_map.squeeze().cpu().numpy()
        
        return t_map
    
    @staticmethod
    def guided_filter(guide, src, radius, eps):
        """
        引导滤波实现
        """
        # 简单实现：使用均值滤波
        # 实际应用中可以使用OpenCV的引导滤波
        guide = np.clip(guide, 0, 1)
        src = np.clip(src, 0, 1)
        
        # 计算局部均值
        mean_I = cv2.blur(guide, (radius, radius))
        mean_p = cv2.blur(src, (radius, radius))
        mean_Ip = cv2.blur(guide * src, (radius, radius))
        
        # 计算协方差
        cov_Ip = mean_Ip - mean_I * mean_p
        
        # 计算方差
        mean_II = cv2.blur(guide * guide, (radius, radius))
        var_I = mean_II - mean_I * mean_I
        
        # 计算a和b
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        # 计算均值
        mean_a = cv2.blur(a, (radius, radius))
        mean_b = cv2.blur(b, (radius, radius))
        
        # 输出
        q = mean_a * guide + mean_b
        
        return q
    
    @staticmethod
    def atmospheric_light(t_map, img_np, ratio=0.01):
        """
        估计大气光
        
        Args:
            t_map: 传输图
            img_np: 原始图像，值在0-1之间
            ratio: 用于估计的像素比例
        """
        h, w = t_map.shape[:2]
        
        # 找到透射率最小的像素
        flat_t = t_map.flatten()
        num_pixels = len(flat_t)
        num_top = max(1, int(num_pixels * ratio))
        
        # 找到透射率最小的像素索引
        indices = np.argpartition(flat_t, num_top)[:num_top]
        
        # 对应到原始图像像素
        flat_img = img_np.reshape(-1, 3)
        candidate_pixels = flat_img[indices]
        
        # 取这些像素中最亮的作为大气光
        # 计算亮度
        brightness = np.mean(candidate_pixels, axis=1)
        brightest_idx = np.argmax(brightness)
        A = candidate_pixels[brightest_idx]
        
        return A
    
    @staticmethod
    def dehaze(img_np, t_map, A, t_min=0.1):
        """
        去雾处理
        
        Args:
            img_np: 有雾图像，值在0-1之间
            t_map: 传输图
            A: 大气光
            t_min: 传输图的最小值，防止除零
        """
        # 确保t_map有正确的形状
        if t_map.ndim == 2:
            t_map = np.expand_dims(t_map, axis=2)
        # 防止除零
        t_map = np.clip(t_map, t_min, 1.0)
        # 去雾公式：J = (I - A)/t + A
        J = (img_np - A) / t_map + A
        # 限制在0-1之间
        J = np.clip(J, 0, 1)
        return J
    
    def forward(self, img, driver):
        """完整的去雾流程"""
        # 转换为numpy数组
        img_np = np.array(img) / 255.0
        # 计算传输图
        t_map = self.transmission_map(img_np, driver)
        # 转换为灰度图用于引导滤波
        gray_img = np.dot(img_np[..., :3], [0.299, 0.587, 0.114]) # RGB到灰度转换的标准权重系数
        # 引导滤波（与原始代码一致）
        t_map = self.guided_filter(gray_img, t_map, radius=50, eps=0.001)
        # 估计大气光
        A = self.atmospheric_light(t_map, img_np)
        # 去雾
        result = self.dehaze(img_np, t_map, A)
        # 保存调试信息
        self.debug = [
            img_np,
            t_map,
            np.full_like(img_np, A),
            result
        ]
        return result

from torch.utils.data.dataset import Dataset

class FogData(Dataset):

    # root：图像存放地址根路径
    # augment：是否需要图像增强
    def __init__(self, root, labels, augment=True):
        # 初始化 可以定义图片地址 标签 是否变换 变换函数
        self.image_files = root
        self.labels = torch.cuda.FloatTensor(labels)
        self.augment = augment   # 是否需要图像增强
        # self.transform = transform

    def __getitem__(self, index):
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1]),
        ])
        transform_augumentation = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std = [1, 1, 1]), # (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]) # TODO: transform的形式未来可以优化
        # 读取图像数据并返回
        if self.augment:
            img = Image.open(self.image_files[index])
            img = transform_augumentation(img)
            img = img.cuda()
            return img, self.labels[index]
        else:
            img = Image.open(self.image_files[index])
            img = transform(img)
            img = img.cuda()
            return img, self.labels[index]

    def __len__(self):
        # 返回图像的数量
        return len(self.image_files)


import os
import random
import torch.utils.data as data

class DehazeNetTrainer:
    
    @staticmethod
    def save_dataset(path_train, label_train, path_txt='path_train.txt', label_txt='label_train.txt'):
        # 将图像路径保存到文本文件
        file = open(path_txt, mode='a')
        for i in range(len(path_train)):
            file.write(str(path_train[i]) + '\n')
        file.close()
        
        # 将对应的标签（透射率t值）保存到文本文件
        file = open(label_txt, mode='a')
        for i in range(len(label_train)):
            file.write(str(label_train[i]) + '\n')
        file.close()

    @staticmethod
    def open_dataset(path_txt='path_train.txt', label_txt='label_train.txt'):
        path_train = []
        file = open(path_txt, mode='r')
        content = file.readlines()
        for i in range(len(content)):
            path_train.append(content[i][:-1])

        label_train = []
        file = open(label_txt, mode='r')
        content = file.readlines()
        for i in range(len(content)):
            label_train.append(float(content[i][:-1]))
        return path_train, label_train

    @staticmethod
    def new_dataset(img_dir, data_dir, num_t=10, patch_size=16):
        """
        从无雾图像生成有雾图像训练数据集
        
        参数:
        img_dir: 无雾图像目录路径
        data_dir: 生成的数据集保存目录，影响path_train的根目录
        num_t: 每个图像块生成的透射率t(x)数量
        patch_size: 图像块大小（像素）
        """
        # 获取无雾图像目录中的所有图像文件名
        img_path = os.listdir(img_dir)
        # 初始化存储路径和标签的列表
        path_train = []  # 存储生成的有雾图像路径
        label_train = []  # 存储对应的透射率t值
        
        # 遍历每张无雾图像
        for image_name in img_path:
            # 构建完整的图像路径
            fullname = os.path.join(img_dir, image_name)
            # 读取图像（OpenCV读取为BGR格式）
            img = cv2.imread(fullname)
            # 获取图像尺寸
            w, h, _ = img.shape
            # 计算图像可以被划分为多少个patch
            num_w = int(w / patch_size)
            num_h = int(h / patch_size)
            # 遍历图像中的每个patch（排除边缘的patch）
            for i in range(1, num_w-1):
                for j in range(1, num_h-1):
                    # 提取无雾图像块
                    free_patch = img[0 + i * patch_size:patch_size + i * patch_size,
                                    0 + j * patch_size:patch_size + j * patch_size, :]
                    # 为当前图像块生成num_t个不同的有雾版本
                    for k in range(num_t):
                        # 随机生成透射率t，范围在[0, 1)之间
                        t = random.random()
                        # 根据雾化模型生成有雾图像块: I = J * t + A * (1-t)
                        # 这里假设大气光A=255（白色雾）
                        hazy_patch = free_patch * t + 255 * (1 - t)
                        # 随机决定是否保存该样本（50%概率）
                        x = random.random()
                        if x > 0.5:
                            # 构建文件名: i坐标 + j坐标 + k序号 + 原图像名
                            picname = '%s'%i + '%s'%j + '%s'%k + image_name
                            hazy_path = os.path.join(data_dir, picname)
                            # 保存有雾图像块到文件
                            cv2.imwrite(hazy_path, hazy_patch) # FIXME: 这里虽然把IO放进循环体内不易于分离职责，但是节省了存储空间，不需要暂存图片
                            # 记录文件路径和对应的透射率t
                            path_train.append(hazy_path)
                            label_train.append(t)
        print(f"数据集创建完成！共生成 {len(path_train)} 个样本。")
        return path_train, label_train

    def __init__(self, path_txt='path_train.txt', label_txt='label_train.txt', batch_size=128):
        path_train, label_train = self.open_dataset(path_txt, label_txt)
        train_data = FogData(path_train, label_train, False)
        train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, )

        self.dataloader = train_loader
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0000005)
        self.loss_func = nn.MSELoss().cuda()

    def loop(self, driver):
        total_loss = 0
        for i, (x, y) in enumerate(self.dataloader):
            # 输入训练数据
            # 清空上一次梯度
            self.optimizer.zero_grad()
            output = driver.net(x)
            # 计算误差
            loss = self.loss_func(output, y)
            total_loss += loss
            # 误差反向传递
            loss.backward()
            # 优化器参数更新
            self.optimizer.step()
            if i % 10 == 5:
                print(' |step ', i, 'loss: %.4f' % loss.item(), )
        print(' |total_loss', total_loss.item())
        return self


class Main:

    @staticmethod
    def visualize_results(debug_images, titles=None):
        """可视化调试图像"""
        import matplotlib.pyplot as plt
        if titles is None:
            titles = ["Original", "Transmission Map", "Atmospheric Light", "Dehazed"]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i, (img, title) in enumerate(zip(debug_images, titles)):
            ax = axes[i]
            if img.ndim == 2:  # 灰度图
                ax.imshow(img, cmap='gray')
            else:  # RGB图
                ax.imshow(np.clip(img, 0, 1))
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    @classmethod
    def train(cls):
        EPOCH = 10
        BATCH_SIZE = 128
        train_driver = DehazeNetDriver(DehazeNet())
        train_driver.open("defog4_noaug.pth") # 加载预训练权重

        train_runner = DehazeNetTrainer(batch_size=BATCH_SIZE)
        for epoch in range(EPOCH):
            print('Epoch', epoch)
            train_runner.loop(train_driver)

        train_driver.save('defog4_noaug_enoch.pth')

    @classmethod
    def test(cls, img_path="dehaze_1.jpg", result_path="dehaze_1_result.jpg"):
        # 加载图像
        img = Image.open(img_path).convert("RGB")
        # 初始化模型
        driver = DehazeNetDriver(DehazeNet())
        driver.open("defog4_noaug.pth")  # 加载预训练权重
        # 执行去雾
        runner = DehazeNetRunner()
        result = runner.forward(img, driver)
        # 可视化结果
        cls.visualize_results(runner.debug)
        # 保存结果
        result_img = Image.fromarray((result * 255).astype(np.uint8))
        result_img.save(result_path)

import typer
if __name__ == "__main__":
    typer.run(Main.test)