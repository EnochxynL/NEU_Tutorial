import time
import cv2
import cv2.ximgproc as xip
import numpy as np

class Dehaze:

    @staticmethod
    def dark_channel(image_original, kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=2):
        image_dark = np.min(image_original, axis = 2)
        image_dark = cv2.erode(image_dark, kernel, iterations=iterations)
        image_dark = cv2.dilate(image_dark, kernel, iterations=iterations) # [x]: 算法改进
        return image_dark
    
    @staticmethod
    def atmospheric_light(dark_channel: np.ndarray, image_original: np.ndarray):
        # 估计大气光
        num_pixels = image_original.shape[0] * image_original.shape[1]
        num_pixels_needed = int(max(num_pixels // 1000, 1))
        
        # 将暗通道和图像重塑为向量形式，准备排序
        dark_channel_vec = dark_channel.reshape(num_pixels)
        image_reshaped = image_original.reshape(num_pixels, 3)
        del num_pixels
        
        # 排序选择最亮的1/1000个暗通道像素
        indices = dark_channel_vec.argsort()[-num_pixels_needed:]

        atmospheric_light = np.mean(image_reshaped[indices], axis=0)

        dark_mask = np.zeros(image_original.shape[0] * image_original.shape[1], dtype=np.uint8)
        dark_mask[indices] = 255
        dark_mask = dark_mask.reshape(image_original.shape[0], image_original.shape[1])
        return atmospheric_light, indices, dark_mask
    
    @staticmethod
    def transmission_filter(image_guide, image_src, radius=60, eps=1e-6):
        return xip.guidedFilter(
            guide=(image_guide * 255).astype(np.uint8), 
            src=(image_src*255).astype(np.uint8), 
            radius=radius, 
            eps=eps
            ).astype(np.float32) / 255.0
    
    @staticmethod
    def sky_detection(image_bgr: np.ndarray,
                    grad_kernel_size: int = 3,
                    denoise_sigma: float = 1.0,
                    grad_thresh: float = 0.08,
                    bright_thresh: float = 0.6,
                    feather_sigma: float = 7.0) -> np.ndarray:
        """
        天空检测：
        1) 去色得到更具对比度的灰度图（优先使用 cv2.decolor）
        2) Sobel 梯度幅值
        3) 对梯度做高斯去噪
        4) 按梯度阈值与亮度阈值区分天空区域（天空通常亮且梯度小/边缘少）
        5) 高斯羽化得到柔和掩膜
        返回：float32 的掩膜，范围 [0,1]，值越大越可能是天空
        参考：https://www.cnblogs.com/Imageshop/p/3907639.html
        """
        # 1) 去色
        # 优先使用 decolor（若不可用，退化为普通灰度）
        gray, _ = cv2.decolor((image_bgr * 255).astype(np.uint8))
        gray = gray.astype(np.float32) / 255.0

        # 2) Sobel 梯度
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=grad_kernel_size)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=grad_kernel_size)
        grad_mag = cv2.magnitude(gx, gy)

        # 3) 去噪
        grad_smooth = cv2.GaussianBlur(grad_mag, (0, 0), denoise_sigma)
        bright = cv2.GaussianBlur(gray, (0, 0), denoise_sigma)

        # 4) 阈值区分（天空通常亮且梯度小）
        # 将阈值归一化到 [0,1] 的图像范围
        grad_mask = (grad_smooth < grad_thresh).astype(np.float32)
        bright_mask = (bright > bright_thresh).astype(np.float32)
        sky_mask = grad_mask * bright_mask

        # 5) 高斯羽化（可选）
        if feather_sigma > 0:
            sky_mask = cv2.GaussianBlur(sky_mask, (0, 0), feather_sigma)
            sky_mask = np.clip(sky_mask, 0.0, 1.0)

        return sky_mask
    
    @staticmethod
    def dehaze(image_atmos, image_original, t, t0 = 0.1):
        
        image_original_o = np.zeros(np.shape(image_original))
        
        image_original_o[:, :, 0] = (image_original[:, :, 0] - image_atmos[0]) / (np.maximum(t, t0)) + image_atmos[0]
        image_original_o[:, :, 1] = (image_original[:, :, 1] - image_atmos[1]) / (np.maximum(t, t0)) + image_atmos[1]
        image_original_o[:, :, 2] = (image_original[:, :, 2] - image_atmos[2]) / (np.maximum(t, t0)) + image_atmos[2]

        return image_original_o
    
    def __init__(self):
        self.debug = []
        # 注意类变量和实例变量的区别

    def forward(self, image_original: np.ndarray, omega = 0.95, sky_dark = 0.5):
        image_dark = self.dark_channel(image_original)
        image_atmos, atmos_indices, dark_mask = self.atmospheric_light(image_dark, image_original)
        image_atmos_dark = self.dark_channel(image_original / image_atmos)
        image_sky = self.sky_detection(image_original)
        image_sky_dark = (sky_dark * image_sky + image_atmos_dark * (1 - image_sky))
        image_trans = 1 - omega * image_sky_dark
        image_trans_filtered = self.transmission_filter(image_original, image_trans)
        image_dehazed = self.dehaze(image_atmos, image_original, image_trans_filtered)
        
        self.debug = [cv2.subtract(image_original, cv2.cvtColor(dark_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0),
                            image_sky_dark,
                            image_trans_filtered,
                            image_dehazed
        ]
        
        return image_dehazed


class Main:

    @staticmethod
    def stack_images(imgs, cols=3):
        # 将多图拼接为网格
        h_imgs = []
        row = []
        max_h = max(i.shape[0] for i in imgs)
        max_w = max(i.shape[1] for i in imgs)
        for idx, im in enumerate(imgs):
            if len(im.shape) == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
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
    def main(cls, name: str="city.jpg"):
        img = cv2.imread(f"{name}").astype(np.float32) / 255.0

        dehaze = Dehaze()
        if True:
            dehazed_img = dehaze.forward(img)
            forward_debug = dehaze.debug
        del dehaze

        # cv2.imwrite(f"{name}.dcp.jpg", (dehazed_img * 255).astype(np.uint8))
        cv2.imshow("Dehazing Process", cls.stack_images(forward_debug, cols=2))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

import typer
if __name__ == "__main__":
    typer.run(Main.main)
