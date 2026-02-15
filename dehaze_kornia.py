import cProfile
import cv2
import numpy as np
import torch
import kornia

class Dehaze:
    def __init__(self, use_debug=False):
        self.use_debug = use_debug
        self.debug = []
        # 预生成椭圆结构元素（15x15）
        self.kernel = self._create_ellipse_kernel(15)

    @staticmethod
    def _create_ellipse_kernel(size):
        """生成椭圆形结构元素（二值）"""
        import numpy as np
        center = (size - 1) / 2
        y, x = np.ogrid[:size, :size]
        mask = ((x - center) / center) ** 2 + ((y - center) / center) ** 2 <= 1
        kernel = mask.astype(np.float32)
        return torch.from_numpy(kernel)

    def dark_channel(self, image_original, iterations=2):
        """
        计算暗通道：先取RGB最小值，再执行腐蚀和膨胀。
        image_original: (1, 3, H, W) torch tensor, [0,1]
        返回: (1, 1, H, W)
        """
        # 取每个像素在通道维度的最小值
        image_dark = torch.min(image_original, dim=1, keepdim=True)[0]  # (1,1,H,W)
        # 腐蚀和膨胀（iterations次）
        for _ in range(iterations):
            image_dark = kornia.morphology.erosion(image_dark, self.kernel)
        for _ in range(iterations):
            image_dark = kornia.morphology.dilation(image_dark, self.kernel)
        return image_dark

    @staticmethod
    def atmospheric_light(dark_channel, image_original):
        """
        估计大气光：选取暗通道中最亮的0.1%像素，取对应原图RGB均值。
        dark_channel: (1,1,H,W)
        image_original: (1,3,H,W)
        返回: (大气光值(3,), 索引, 二值掩膜(H,W) uint8)
        """
        B, C, H, W = image_original.shape
        num_pixels = H * W
        num_pixels_needed = max(num_pixels // 1000, 1)

        dark_vec = dark_channel.view(-1)  # (N,)
        image_reshaped = image_original.permute(0, 2, 3, 1).reshape(-1, C)  # (N, C)

        # 升序排序，取最后num_pixels_needed个（最亮）
        sorted_indices = torch.argsort(dark_vec, descending=False)
        indices = sorted_indices[-num_pixels_needed:]

        atmospheric_light = torch.mean(image_reshaped[indices], dim=0)  # (C,)

        dark_mask = torch.zeros((H * W,), dtype=torch.uint8, device=dark_channel.device)
        dark_mask[indices] = 255
        dark_mask = dark_mask.view(H, W)  # (H,W)

        return atmospheric_light, indices, dark_mask

    @staticmethod
    def transmission_filter(image_guide, image_src, radius=60, eps=1e-6):
        """
        导向滤波优化透射率图。
        image_guide: (1,3,H,W) [0,1] 引导图像
        image_src: (1,1,H,W) [0,1] 待滤波图像
        返回: (1,1,H,W) [0,1]
        """
        # 转换为0-255范围，保持与OpenCV实现一致的eps尺度
        guide = image_guide * 255.0
        src = image_src * 255.0
        kernel_size = 2 * radius + 1  # 确保奇数
        filtered = kornia.filters.guided_blur(guide, src, kernel_size, eps)
        return filtered / 255.0

    @staticmethod
    def sky_detection(image_bgr,
                      grad_kernel_size=3,      # 保留参数，但当前实现固定使用3x3 Sobel
                      denoise_sigma=1.0,
                      grad_thresh=0.08,
                      bright_thresh=0.6,
                      feather_sigma=7.0) -> torch.Tensor:
        """
        天空检测（基于梯度与亮度）。
        image_bgr: (1,3,H,W) torch tensor, [0,1]
        返回: (1,1,H,W) [0,1] 天空概率掩膜
        """
        # 转灰度
        gray = kornia.color.rgb_to_grayscale(image_bgr)  # (1,1,H,W)

        # 计算Sobel梯度幅值（核大小固定为3）
        grad_mag = kornia.filters.sobel(gray)  # (1,1,H,W)

        # 高斯去噪（根据sigma自动计算核大小）
        def _gaussian_blur(img, sigma):
            ksize = int(2 * round(3 * sigma) + 1)
            if ksize % 2 == 0:
                ksize += 1
            return kornia.filters.gaussian_blur2d(img, (ksize, ksize), (sigma, sigma))

        grad_smooth = _gaussian_blur(grad_mag, denoise_sigma)
        bright_smooth = _gaussian_blur(gray, denoise_sigma)

        # 阈值区分
        grad_mask = (grad_smooth < grad_thresh).float()
        bright_mask = (bright_smooth > bright_thresh).float()
        sky_mask = grad_mask * bright_mask

        # 高斯羽化
        if feather_sigma > 0:
            sky_mask = _gaussian_blur(sky_mask, feather_sigma)
            sky_mask = torch.clamp(sky_mask, 0.0, 1.0)

        return sky_mask

    @staticmethod
    def dehaze(image_atmos, image_original, t, t0=0.1):
        """
        根据大气光和透射率恢复无雾图像。
        image_atmos: (3,) 大气光值
        image_original: (1,3,H,W) 有雾图像
        t: (1,1,H,W) 透射率
        返回: (1,3,H,W)
        """
        t0 = torch.tensor(t0, device=t.device)
        t_expanded = torch.maximum(t, t0)  # (1,1,H,W)
        atmos_expanded = image_atmos[None, :, None, None]  # (1,3,1,1)
        result = (image_original - atmos_expanded) / t_expanded + atmos_expanded
        return result

    def forward(self, image_original: np.ndarray, omega=0.95, sky_dark=0.5):
        """
        输入: image_original (H,W,3) np.float32 [0,1]
        返回: dehazed (H,W,3) np.float32 [0,1]
        """
        # 转换为torch张量 (1,3,H,W)
        img_t = torch.from_numpy(image_original).permute(2, 0, 1).unsqueeze(0).float()

        # 暗通道
        image_dark = self.dark_channel(img_t)

        # 大气光
        image_atmos, atmos_indices, dark_mask = self.atmospheric_light(image_dark, img_t)

        # 计算暗通道（大气光归一化后）
        image_atmos_dark = self.dark_channel(img_t / image_atmos.view(1, 3, 1, 1))

        # 天空检测
        image_sky = self.sky_detection(img_t)  # (1,1,H,W)

        # 融合天空暗通道
        image_sky_dark = sky_dark * image_sky + image_atmos_dark * (1 - image_sky)

        # 透射率
        image_trans = 1 - omega * image_sky_dark

        # 导向滤波
        image_trans_filtered = self.transmission_filter(img_t, image_trans)

        # 去雾
        image_dehazed = self.dehaze(image_atmos, img_t, image_trans_filtered)

        # 转回numpy (H,W,3)
        dehazed_np = image_dehazed.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        dehazed_np = np.clip(dehazed_np, 0, 1)

        if self.use_debug:
            # 准备调试图像（转为numpy方便显示）
            # debug0: 原图减去暗通道掩膜（掩膜高亮区域被减）
            mask_float = dark_mask.float().unsqueeze(0).unsqueeze(0) / 255.0  # (1,1,H,W)
            mask_float = mask_float.repeat(1, 3, 1, 1)  # (1,3,H,W)
            debug0 = img_t - mask_float
            debug0_np = debug0.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

            # debug1: 融合后的天空暗通道 (1,1,H,W) -> (H,W)
            debug1_np = image_sky_dark.squeeze().detach().cpu().numpy()

            # debug2: 滤波后的透射率 (1,1,H,W) -> (H,W)
            debug2_np = image_trans_filtered.squeeze().detach().cpu().numpy()

            # debug3: 去雾结果 (1,3,H,W) -> (H,W,3)
            debug3_np = image_dehazed.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

            self.debug = [debug0_np, debug1_np, debug2_np, debug3_np]

        return dehazed_np


class Main:
    @staticmethod
    def stack_images(imgs, cols=3):
        """将多图拼接为网格（与原始代码相同，输入为numpy图像）"""
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
            while len(row) < cols:
                row.append(np.zeros_like(row[0]))
            h_imgs.append(np.hstack(row))
        return np.vstack(h_imgs)

    @classmethod
    def main(cls, name: str = "assets/dehaze/dehaze_1.jpg"):
        img = cv2.imread(name).astype(np.float32) / 255.0

        dehaze = Dehaze(use_debug=True)
        dehazed_img = dehaze.forward(img, omega=0.8)
        forward_debug = dehaze.debug

        cv2.imwrite(f"{name}.dcp.jpg", (dehazed_img * 255).astype(np.uint8))
        cv2.imshow("Dehazing Process", cls.stack_images(forward_debug, cols=2))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @classmethod
    def main_video(cls, video_source='assets/dehaze/fire_output_x264.mp4'):
        """视频去雾（与原始代码相同，内部已适应numpy输入输出）"""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"无法打开视频源: {video_source}")
            return

        dehaze = Dehaze()
        print("开始视频去雾... 按 'q' 退出，按 's' 保存当前帧")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img = frame.astype(np.float32) / 255.0
            dehazed_img = dehaze.forward(img)
            dehazed_frame = (dehazed_img * 255).astype(np.uint8)

            comparison = np.hstack([frame, dehazed_frame])
            cv2.imshow("Original vs Dehazed", comparison)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("captured_frame.jpg", frame)
                cv2.imwrite("dehazed_frame.jpg", dehazed_frame)
                print("帧已保存")

        cap.release()
        cv2.destroyAllWindows()

import typer

if __name__ == "__main__":
    # Main.main()
    # cProfile.run('Main.main()')
    typer.run(Main.main_video)