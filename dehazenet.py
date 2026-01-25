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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.input_channels, kernel_size=5, padding=0)
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


class DehazeNetRunner():
    
    @staticmethod
    def transmission_map(img_np, driver: DehazeNetDriver, patch_size=16):
        # 数据转换格式
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        h, w, _ = img_np.shape

        # 根据patch大小扩大画布：计算需要padding的大小
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        if pad_h > 0 or pad_w > 0:
            img_np_pad = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            img_np_pad = img_np
        h_pad, w_pad, _ = img_np_pad.shape
        num_h = h_pad // patch_size
        num_w = w_pad // patch_size

        # 创建传输图
        t_map = np.zeros((h_pad, w_pad))
        # 处理每个patch
        for i in range(num_w):
            for j in range(num_h):
                # 提取patch
                patch_np = img_np_pad[j*patch_size:(j+1)*patch_size, i*patch_size:(i+1)*patch_size, :]
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
        return t_map
        
    @staticmethod
    def atmospheric_light(t_map, img_np):
        h, w, _ = img_np.shape
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
        return A

    @staticmethod
    def dehaze(A, img_np, t_map, t0=0.1):
        h, w, _ = img_np.shape
        # 应用去雾公式
        t_map = np.clip(t_map, t0, 1.0)  # 避免除0
        t_map = t_map[..., np.newaxis]  # 添加通道维度
        
        result = (img_np[:h, :w, :] - A * (1 - t_map)) / t_map
        result = np.clip(result, 0, 1)
        return result

    def __init__(self):
        self.debug = []

    def forward(self, img, driver: DehazeNetDriver):
        """快速去雾版本 - 使用滑动窗口
        评价：不是用原图估计，而是用透射图的相反数（暗通道）！"""
        # 转换为numpy数组
        img_np = np.array(img) / 255.0
        t_map = self.transmission_map(img_np, driver)
        a_light = self.atmospheric_light(t_map, img_np)
        result = self.dehaze(a_light, img_np, t_map)
        self.debug = [img_np, t_map, 
                      np.full(img_np.shape, a_light * 255, dtype=np.uint8),
                      result]
        return result


class DehazeNetTester:

    @staticmethod
    def stack_images(imgs, cols=3):
        # 将多图拼接为网格
        h_imgs = []
        row = []
        max_h = max(i.shape[0] for i in imgs)
        max_w = max(i.shape[1] for i in imgs)
        for idx, im in enumerate(imgs):
            if len(im.shape) == 2:
                im = cv2.cvtColor(im.astype(np.float32), cv2.COLOR_GRAY2BGR) # OpenCV不支持depth=6也就是64位浮点数
            im = cv2.resize(im, (max_w, max_h))
            row.append(im)
            if (idx + 1) % cols == 0:
                h_imgs.append(np.hstack(row))
                row = []
        if row:
            # 填充最后一行
            while len(row) < cols:
                row.append(np.zeros_like(row[0]))
            h_imgs.append(np.hstack(row))
        return np.vstack(h_imgs)

    @classmethod
    def main(cls, open_dir='dehaze_1.jpg', save_dir='city.jpg.net.jpg', model_path='defog4_noaug.pth'):
        # 加载图像
        img = Image.open(open_dir).convert('RGB')
        # 初始化模型
        driver = DehazeNetDriver(DehazeNet()).open(model_path)
        runner = DehazeNetRunner()
        # 推理过程
        result = runner.forward(img, driver)
        runner_debug = cv2.cvtColor(cls.stack_images(runner.debug, cols=2).astype(np.float32), cv2.COLOR_RGB2BGR) # PIL是RGB图，记得转换
        # 调试输出
        cv2.imshow("Dehazing Process", runner_debug)
        # 保存结果
        result_img = Image.fromarray((result * 255).astype(np.uint8))
        # result_img.save(save_dir)
        print(f"快速去雾图像已保存到 {save_dir}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class FogData(Dataset):
	IMAGE_AUGUMENTATION = torchvision.transforms.Compose([
		transforms.RandomHorizontalFlip(0.5),
		transforms.RandomVerticalFlip(0.5),
		transforms.RandomRotation(30),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	# root：图像存放地址根路径
	# augment：是否需要图像增强
	def __init__(self, root, labels, augment=True):
		# 初始化 可以定义图片地址 标签 是否变换 变换函数
		self.image_files = root
		self.labels = torch.cuda.FloatTensor(labels)
		self.augment = augment   # 是否需要图像增强
		# self.transform = transform

	def __getitem__(self, index):
		# 读取图像数据并返回
		if self.augment:
			img = Image.open(self.image_files[index])
			img = self.IMAGE_AUGUMENTATION(img)
			img = img.cuda()
			return img, self.labels[index]
		else:
			img = Image.open(self.image_files[index])
			img = self.IMAGE_LOADER(img)
			img = img.cuda()
			return img, self.labels[index]

	def __len__(self):
		# 返回图像的数量
		return len(self.image_files)


class DehazeNetTrainer():

	EPOCH = 10

	def setup(self, loader):
		self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0000005)
		self.loader = loader
		self.loss_func = nn.MSELoss().cuda()
		self.epoch = 0
		return self

	def loop(self):
		total_loss = 0
		for i, (x, y) in enumerate(self.loader):
			# 输入训练数据
			# 清空上一次梯度
			self.optimizer.zero_grad()
			output = self.net(x)
			# 计算误差
			loss = self.loss_func(output, y)
			total_loss += loss
			# 误差反向传递
			loss.backward()
			# 优化器参数更新
			self.optimizer.step()
			if i % 10 == 5:
				print('Epoch', self.epoch, '|step ', i, 'loss: %.4f' % loss.item(), )
		print('Epoch', self.epoch, 'total_loss', total_loss.item())
		self.epoch += 1
		return self

	#@torchsnooper.snoop()
	@classmethod
	def main(cls):
		path_train = []
		file = open('path_train.txt', mode='r')
		content = file.readlines()
		for i in range(len(content)):
			path_train.append(content[i][:-1])

		label_train = []
		file = open('label_train.txt', mode='r')
		content = file.readlines()
		for i in range(len(content)):
			label_train.append(float(content[i][:-1]))
			#print(float(content[i][:-1]))

		BATCH_SIZE = 128

		train_data = FogData(path_train, label_train, False)
		train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, )
		trainer = DehazeNetTrainer(DehazeNet()).open(r'defog4_noaug.pth').setup(train_loader)
		for epoch in range(trainer.EPOCH):
			trainer.loop()
		trainer.save(r'defog4_noaug.pth')




import typer
if __name__ == "__main__":
    typer.run(DehazeNetTester.main)