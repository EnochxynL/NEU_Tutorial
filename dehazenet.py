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


# class DehazeNet(nn.Module):
#     def __init__(self):
#         super(DehazeNet, self).__init__()
        
#         # 与原始代码完全对应的网络结构
#         # Feature Extraction 层 (F1)
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        
#         # Multi-scale Mapping 层 (F2)
#         self.conv3x3 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
#         self.conv5x5 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, padding=2)
#         self.conv7x7 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=7, padding=3)
        
#         # Local Extremum 层 (F3) - 7x7最大池化
#         self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        
#         # Non-linear Regression 层 (F4) - 6x6卷积
#         self.conv_ip = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=6, padding=0)
        
#         self.brelu = BRelu()
        
#         self._initialize_weights()
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, mean=0, std=0.001)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
    
#     def Maxout(self, x, groups=4):
#         """与原始代码相同的Maxout实现"""
#         batch_size, channels, height, width = x.shape
#         channels_per_group = channels // groups
        
#         # 将通道分组
#         x = x.view(batch_size, groups, channels_per_group, height, width)
        
#         # 在每个组内取最大值
#         x, _ = torch.max(x, dim=2)
        
#         return x
    
#     def forward(self, x):
#         # 保存原始尺寸用于后续恢复
#         orig_h, orig_w = x.shape[2], x.shape[3]
        
#         # Feature Extraction F1
#         f1 = self.conv1(x)
#         f1 = self.Maxout(f1, groups=4)  # 16通道 -> 4通道
        
#         # Multi-scale Mapping F2
#         f2_3x3 = self.conv3x3(f1)  # 4通道 -> 16通道
#         f2_5x5 = self.conv5x5(f1)  # 4通道 -> 16通道
#         f2_7x7 = self.conv7x7(f1)  # 4通道 -> 16通道
#         f2 = torch.cat([f2_3x3, f2_5x5, f2_7x7], dim=1)  # 48通道
        
#         # Local Extremum F3
#         f3 = self.maxpool(f2)
        
#         # Non-linear Regression F4
#         # 注意：6x6卷积会使尺寸缩小5，我们需要计算输出尺寸
#         f4 = self.conv_ip(f3)
#         f4 = self.brelu(f4)  # 输出范围[0,1]
        
#         # 使用双线性插值恢复到原始尺寸
#         f4 = F.interpolate(f4, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        
#         return f4


class DehazeNet(nn.Module):

    @staticmethod
    def Maxout(x, groups):
        """与原始代码完全相同的Maxout实现"""
        batch_size, channels, height, width = x.shape
        
        x = x.reshape(batch_size, groups, channels // groups, height, width)
        x, _ = torch.max(x, dim=2, keepdim=True)
        out = x.reshape(batch_size, groups, height, width)
        
        return out

    def __init__(self, input_channels=16, groups=4):
        '''注意：原始论文的MATLAB代码卷积使用SAME卷积，对称填充。为节省代码，采用SAME卷积、零填充'''
        super(DehazeNet, self).__init__()
        self.input_channels = input_channels
        self.groups = groups
        
        # Feature Extraction
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.input_channels, kernel_size=5, padding=2)
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
        out = self.conv1(x)
        out = self.Maxout(out, self.groups)
        # Multi-scale Mapping
        out1 = self.conv2(out)
        out2 = self.conv3(out)
        out3 = self.conv4(out)
        y = torch.cat((out1, out2, out3), dim=1)
        # Local Extremum
        y = self.maxpool(y)
        # Non-linear Regression
        y = self.conv5(y)
        y = self.brelu(y)
        # 确保输出与输入相同尺寸
        y = torch.nn.functional.interpolate(y, size=orig_size, mode='bilinear', align_corners=False)
        # 返回全分辨率传输图
        return y


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
        # 原始代码的预处理：减去0.5
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
        gray_img = np.dot(img_np[..., :3], [0.299, 0.587, 0.114])
        
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


# 使用示例
if __name__ == "__main__":
    # 加载图像
    img_path = "dehaze_1.jpg"
    img = Image.open(img_path).convert("RGB")
    
    # 初始化模型
    driver = DehazeNetDriver(DehazeNet())
    driver.open("defog4_noaug.pth")  # 加载预训练权重
    
    # 执行去雾
    runner = DehazeNetRunner()
    result = runner.forward(img, driver)
    
    # 可视化结果
    visualize_results(runner.debug)
    
    # 保存结果
    result_img = Image.fromarray((result * 255).astype(np.uint8))
    result_img.save("dehaze_1_result.jpg")