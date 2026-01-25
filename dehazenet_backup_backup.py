import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from PIL import Image
import torchvision
from torchvision import transforms
import torch.utils.data as data
import cv2
import numpy as np


class BRelu(nn.Module):
    '''保持与原始代码兼容的BRelu'''
    def __init__(self):
        super(BRelu, self).__init__()
    
    def forward(self, x):
        # 使用Hardtanh模拟原始行为
        return torch.clamp(x, 0, 1)


class DehazeNet(nn.Module):
    def __init__(self, input_channels=16, groups=4):
        super(DehazeNet, self).__init__()
        self.input_channels = input_channels
        self.groups = groups
        
        # 与原始代码完全相同的层定义
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.input_channels, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=7, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=1, kernel_size=6)
        
        # 使用兼容的BRelu
        self.brelu = BRelu()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # 使用原始代码的初始化方式
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def Maxout(self, x, groups):
        """与原始代码完全相同的Maxout实现"""
        batch_size, channels, height, width = x.shape
        
        # 原始代码的实现
        x = x.reshape(batch_size, groups, channels // groups, height, width)
        x, _ = torch.max(x, dim=2, keepdim=True)
        
        # 注意：原始代码中这里的reshape可能有误，但为了兼容性保持相同
        # 原始代码：out = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        # 应该为：out = x.reshape(batch_size, groups, height, width)
        # 但为了兼容性，我们使用原始代码的方式
        out = x.reshape(batch_size, groups, height, width)
        
        return out
    
    def forward(self, x):
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
        # 展平输出，与原始代码相同
        y = y.reshape(y.shape[0], -1)
        
        return y


class DehazeNetDriver:

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
    
    @staticmethod
    def dehaze_image(open_dir='fogpic.jpg', save_dir='test.jpg', model_path='defog4_noaug.pth'):
        """去雾主函数 - 与原始代码defog函数保持相同逻辑
        评价：把估计大气光的过程也按照patch操作，不符合人的认知惯性，不便写代码
        原论文也不是这样的，见原论文https://caibolun.github.io/papers/DehazeNet.pdf公式(15)前面一点"""
        # 加载图像
        img = Image.open(open_dir).convert('RGB')

        # 初始化模型
        driver = DehazeNetDriver(DehazeNet()).open(model_path)
        
        # 创建数据转换
        loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img1 = loader(img)  # 归一化的tensor
        img2 = transforms.ToTensor()(img)  # 原始范围[0,1]的tensor
        
        c, h, w = img1.shape
        
        # 设置patch大小
        patch_size = 16
        num_w = int(w / patch_size)
        num_h = int(h / patch_size)
        
        t_list = []

        # 处理每个patch
        for i in range(num_w):
            for j in range(num_h):
                # 提取patch
                patch = img1[:, 
                             j * patch_size:patch_size + j * patch_size,
                             i * patch_size:patch_size + i * patch_size]
                
                # 添加batch维度
                patch = patch.unsqueeze(0)
                
                # 预测传输系数t
                with torch.no_grad():
                    t = driver.net(patch.to(driver.device))
                    t_value = t.cpu().item()  # 获取标量值
                
                print(t_value)

                t_list.append([i, j, t_value])
        
        # 按t值排序。sorted是从小到大排序
        t_list = sorted(t_list, key=lambda x: x[2])
        # 在透射图中取前1%的patch估计大气光，前！是取小的！见原论文https://caibolun.github.io/papers/DehazeNet.pdf公式(4)
        a_list = t_list[:max(1, len(t_list) // 100)]
        a0 = 0
        for k in range(len(a_list)):
            patch = img2[:,
                         a_list[k][1] * patch_size:patch_size + a_list[k][1] * patch_size,
                         a_list[k][0] * patch_size:patch_size + a_list[k][0] * patch_size]
            
            a = torch.max(patch)
            if a.item() > a0:
                a0 = a.item()
        
        # 应用去雾公式到每个patch
        for k in range(len(t_list)):
            i, j, t = t_list[k]
            # 获取原始图像patch
            patch = img2[:,
                         j * patch_size:patch_size + j * patch_size,
                         i * patch_size:patch_size + i * patch_size]
            
            # 应用去雾公式: J = (I - A*(1-t))/t
            # 注意：t可能为0，需要处理
            if t < 0.1:
                t = 0.1
            
            result_patch = (patch - a0 * (1 - t)) / t
            
            # 限制范围到[0, 1]
            result_patch = torch.clamp(result_patch, 0, 1)
            
            # 放回结果图像
            img2[:,
                 j * patch_size:patch_size + j * patch_size,
                 i * patch_size:patch_size + i * patch_size] = result_patch
        
        # 转换为PIL图像并保存
        defog_img = transforms.ToPILImage()(img2)
        defog_img.save(save_dir)
        print(f"去雾图像已保存到 {save_dir}")
        
        return defog_img
    
    @staticmethod
    def dehaze_image_fast(open_dir='fogpic.jpg', save_dir='test.jpg', model_path='defog4_noaug.pth'):
        """快速去雾版本 - 使用滑动窗口
        评价：不是用原图估计，而是用透射图的相反数（暗通道）！"""
        # 加载图像
        img = Image.open(open_dir).convert('RGB')
        
        # 转换为numpy数组
        img_np = np.array(img) / 255.0
        
        # 初始化模型
        driver = DehazeNetDriver(DehazeNet()).open(model_path)
        
        # 数据转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        h, w, _ = img_np.shape
        patch_size = 16
        
        # 计算需要padding的大小
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        
        if pad_h > 0 or pad_w > 0:
            img_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        h_pad, w_pad, _ = img_np.shape
        num_h = h_pad // patch_size
        num_w = w_pad // patch_size
        
        # 创建传输图
        t_map = np.zeros((h_pad, w_pad))
        # 处理每个patch
        for i in range(num_w):
            for j in range(num_h):
                # 提取patch
                patch_np = img_np[j*patch_size:(j+1)*patch_size, i*patch_size:(i+1)*patch_size, :]
                # 转换为tensor
                patch_tensor = transform(Image.fromarray((patch_np*255).astype(np.uint8)))
                patch_tensor = patch_tensor.unsqueeze(0)
                # 预测传输系数
                with torch.no_grad():
                    t = driver.net(patch_tensor.to(driver.device))
                    t_value = t.cpu().item()
                # 填充传输图
                t_map[j*patch_size:(j+1)*patch_size, i*patch_size:(i+1)*patch_size] = t_value
        # 裁剪回原始大小
        t_map = t_map[:h, :w]
        
        # 估计大气光，找到最亮的0.1%像素
        flat_img = img_np[:h, :w, :].reshape(-1, 3)
        # brightness = np.mean(flat_img, axis=1)
        brightness = t_map[:h, :w].reshape(-1) # 不是用原图估计，而是用透射图！
        num_pixels = brightness.shape[0]
        num_top = max(1, num_pixels // 1000)  # 0.1%
        # 获取最？像素的索引。argsort从小到大排序
        # indices = np.argsort(-brightness)[-num_top:] # 暗通道法用暗通道（与透射图负相关）的最大索引
        indices = np.argsort(brightness)[:num_top] # 论文中用透射图的最小索引，见原论文https://caibolun.github.io/papers/DehazeNet.pdf公式(4)
        # 计算大气光
        # A = np.max(flat_img[indices], axis=0) # 原论文用的最亮值
        A = np.mean(flat_img[indices], axis=0) # 当然也可以用平均值

        # 应用去雾公式
        t_map = np.clip(t_map, 0.1, 1.0)  # 避免除0
        t_map = t_map[..., np.newaxis]  # 添加通道维度
        
        result = (img_np[:h, :w, :] - A * (1 - t_map)) / t_map
        result = np.clip(result, 0, 1)
        
        # 保存结果
        result_img = Image.fromarray((result * 255).astype(np.uint8))
        result_img.save(save_dir)
        
        print(f"快速去雾图像已保存到 {save_dir}")
        return result_img


if __name__ == "__main__":
    # 测试去雾
    DehazeNetDriver.dehaze_image('city.jpg', 'city_result.jpg', 'defog4_noaug.pth')
    
    # 或者使用快速版本
    DehazeNetDriver.dehaze_image_fast('hazy2.jpg', 'hazyy2_result_fast.jpg', 'defog4_noaug.pth')
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()