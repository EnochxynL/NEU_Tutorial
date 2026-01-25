import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

class Detect:
    # 扩大后的红色范围（可根据样本继续放宽）
    # 红色在 HSV 中分布在低 H 和高 H 两段
    RED_LOWER1 = np.array([0, 50, 30])
    RED_UPPER1 = np.array([10, 255, 255])
    RED_LOWER2 = np.array([140, 50, 30])
    RED_UPPER2 = np.array([180, 255, 255])

    # 参数可调
    MIN_CONTOUR_AREA = 500  # 最小轮廓面积阈值（根据图片分辨率调整）
    KERNEL_SIZE0 = (7, 7)    # 形态学核大小
    KERNEL_SIZE1 = (3, 3)    # 形态学核大小
    KERNEL_SIZE2 = (15, 15)    # 形态学核大小

    @classmethod
    def forward(cls, img):
        """
        detect_red_vehicle
        输入：BGR 图像
        输出：(x,y,w,h) 或 None，如果找到则返回 bbox，以及二值掩码
        """
        mask = None

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 模糊处理以减少噪声
        hsv = cv2.GaussianBlur(hsv, cls.KERNEL_SIZE0, 0)
        if True:
            mask1 = cv2.inRange(hsv, cls.RED_LOWER1, cls.RED_UPPER1)
            mask2 = cv2.inRange(hsv, cls.RED_LOWER2, cls.RED_UPPER2)
            if True:
                mask = cv2.bitwise_or(mask1, mask2)
            del mask1, mask2

            # 可选：增强对显色区域的响应（利用饱和度通道）
            s = hsv[:, :, 1]
            _, s_mask = cv2.threshold(s, 30, 255, cv2.THRESH_BINARY)  # 适当降低阈值以覆盖更多颜色
            mask = cv2.bitwise_and(mask, s_mask)
        del hsv

        # 形态学去噪与填洞
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, cls.KERNEL_SIZE1)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, cls.KERNEL_SIZE2)
        if True:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2, iterations=2)
        del kernel1, kernel2

        # 找轮廓，取最大有效轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours is not None:
            # 选面积最大的轮廓（且超过最小面积）
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < cls.MIN_CONTOUR_AREA:
                    continue
                # 使用最小外接矩形（支持旋转）
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                # 转换为普通 bounding box (x,y,w,h) 方便绘制/使用
                x, y, w, h = cv2.boundingRect(cnt)
                return (x, y, w, h), mask, contours
        return None, mask, contours

class Main:

    @staticmethod
    def draw_bbox(img, bbox, color=(0, 255, 0), thickness=2):
        if bbox is None:
            return img
        x, y, w, h = bbox
        out = img.copy()
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        return out

    @staticmethod
    def save_hsv_hist(hsv, out_path):
        """
        保存 HSV 直方图（H, S, V 三个子图）到 out_path（PNG）
        """
        h, s, v = cv2.split(hsv)
        hist_h = cv2.calcHist([h], [0], None, [180], [0, 180]).flatten()
        hist_s = cv2.calcHist([s], [0], None, [256], [0, 256]).flatten()
        hist_v = cv2.calcHist([v], [0], None, [256], [0, 256]).flatten()

        fig, axes = plt.subplots(3, 1, figsize=(6, 8))
        axes[0].plot(hist_h, color='r'); axes[0].set_title('H histogram'); axes[0].set_xlim([0, 180])
        axes[1].plot(hist_s, color='g'); axes[1].set_title('S histogram'); axes[1].set_xlim([0, 255])
        axes[2].plot(hist_v, color='b'); axes[2].set_title('V histogram'); axes[2].set_xlim([0, 255])
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    @staticmethod
    # 新增：保存 bbox 内的 HSV 直方图
    def save_hsv_hist_bbox(hsv, bbox, out_path):
        """
        在检测到 bbox 时，保存 bbox 区域的 HSV 直方图。
        bbox: (x,y,w,h)
        """
        if bbox is None:
            return False
        x, y, w, h = bbox
        # 防止越界
        H, W = hsv.shape[:2]
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + w); y1 = min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return False
        roi = hsv[y0:y1, x0:x1]
        if roi.size == 0:
            return False

        h_ch, s_ch, v_ch = cv2.split(roi)
        hist_h = cv2.calcHist([h_ch], [0], None, [180], [0, 180]).flatten()
        hist_s = cv2.calcHist([s_ch], [0], None, [256], [0, 256]).flatten()
        hist_v = cv2.calcHist([v_ch], [0], None, [256], [0, 256]).flatten()

        fig, axes = plt.subplots(3, 1, figsize=(6, 8))
        axes[0].plot(hist_h, color='r'); axes[0].set_title('H histogram (bbox)'); axes[0].set_xlim([0, 180])
        axes[1].plot(hist_s, color='g'); axes[1].set_title('S histogram (bbox)'); axes[1].set_xlim([0, 255])
        axes[2].plot(hist_v, color='b'); axes[2].set_title('V histogram (bbox)'); axes[2].set_xlim([0, 255])
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return True

    @classmethod
    def main(cls, fp, output_dir):
        img = cv2.imread(fp)
        if img is None:
            # continue
            return
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 保存 HSV 直方图
        base = os.path.splitext(os.path.basename(fp))[0]
        hist_path = os.path.join(output_dir, base + "_hsv_hist.svg")
        cls.save_hsv_hist(hsv, hist_path)

        bbox, mask, contours = Detect.forward(img)
        out = cls.draw_bbox(img, bbox)
        # 保存结果图和 mask
        cv2.imwrite(os.path.join(output_dir, base + "_detected.png"), out)
        cv2.imwrite(os.path.join(output_dir, base + "_mask.png"), mask)
        # 可选：保存轮廓图
        contour_img = img.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, base + "_contours.png"), contour_img)
        print(f"Processed {fp} -> bbox={bbox}")

        # 新增：保存 bbox 内的 HSV 直方图（若存在）
        bbox_hist_path = os.path.join(output_dir, base + "_hsv_bbox_hist.svg")
        saved = cls.save_hsv_hist_bbox(hsv, bbox, bbox_hist_path)
        if not saved:
            # 若没有检测到 bbox 或保存失败，记录信息
            print(f"Processed {fp} -> bbox={bbox} (no bbox histogram saved)")
        else:
            print(f"Processed {fp} -> bbox={bbox} (bbox histogram saved)")

    @classmethod
    def process_folder(cls, input_glob="./*.jpg", output_dir="./results"): # 或 "./images/*.png" 根据实际路径改
        # 示例：在当前目录下处理所有 jpg/png，输出到 results 子目录
        # 在 Windows 终端中运行示例：
        # python car.py
        print("请务必在图片文件相同目录执行代码！")
        os.makedirs(output_dir, exist_ok=True)
        files = sorted(glob.glob(input_glob))
        for fp in files:
            cls.main(fp, output_dir)

import typer
if __name__ == "__main__":
    typer.run(Main.process_folder)