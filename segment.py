# import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, label

class LungSegment:

    @staticmethod
    def blocked_threshold(gray, block_size=3, threshold_ratio=1):
        h, w = gray.shape
        thresh = np.zeros_like(gray)
        for i in range(block_size):
            for j in range(block_size):
                block = gray[i * h // block_size:(i + 1) * h // block_size, j * w // block_size:(j + 1) * w // block_size]
                # _, block_thresh = cv2.threshold(block, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                otsu_thresh, _ = cv2.threshold(block, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                adjusted_thresh = otsu_thresh * threshold_ratio
                _, block_thresh = cv2.threshold(block, adjusted_thresh, 255, cv2.THRESH_BINARY)
                thresh[i * h // block_size:(i + 1) * h // block_size, j * w // block_size:(j + 1) * w // block_size] = block_thresh
        return thresh

    @staticmethod
    def remove_background_2d(binary):
        """去除与图像边缘连通的背景区域（2D）。"""
        h, w = binary.shape
        ex = cv2.copyMakeBorder(binary, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)
        mask_flood = ex.copy()
        cv2.floodFill(mask_flood, None, (0, 0), 255)
        # mask_flood 中被填充为255的是背景，恢复到原始大小并取反得到前景
        bg_mask = mask_flood[3:-3, 3:-3] == 255
        result = np.zeros_like(binary)
        result[~bg_mask] = 255
        return result

    @staticmethod
    def remove_small_objects(mask, min_area=1000):
        """移除小连通域 (2D)。"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                out[labels == i] = 255
        return out

    @staticmethod
    def select_lobes(mask, lobes_num=2, connectivity=4):
        """选择最大的几个连通域作为肺叶区域 (2D)。"""
        
        # 肺只有两片，选择最大的两个连通域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
        if num_labels <= 1:
            # 无前景
            selected = np.zeros_like(mask)
        else:
            # 忽略背景(标签0)，计算各连通域面积并取前两大
            areas = stats[1:, cv2.CC_STAT_AREA]
            sorted_idx = np.argsort(areas)[::-1]
            top = sorted_idx[:lobes_num]  # 若连通域不足2个则截断
            selected = np.zeros_like(mask)
            for i in top:
                lbl = int(i) + 1
                selected[labels == lbl] = 255
        return selected

    @staticmethod
    def fill_holes_2d(mask):
        return binary_fill_holes(mask > 0).astype(np.uint8) * 255

    @staticmethod
    def morph_close_2d(mask, kernel_size=5):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def keep_largest_objects_2d(mask, max_objects=2, area_ratio_thresh=5.0):
        labeled, num = label(mask > 0)
        if num == 0:
            return np.zeros_like(mask)
        counts = np.bincount(labeled.flat)
        areas = counts[1:]
        if len(areas) == 0:
            return np.zeros_like(mask)
        sorted_idx = np.argsort(areas)[::-1]
        if len(sorted_idx) == 1:
            return (labeled == (sorted_idx[0] + 1)).astype(np.uint8) * 255
        # 选择最多 max_objects 个连通域（考虑面积比）
        chosen = [sorted_idx[0] + 1]
        if len(sorted_idx) >= 2:
            a1, a2 = areas[sorted_idx[0]], areas[sorted_idx[1]]
            if a1 > area_ratio_thresh * a2:
                # 保留最大一个
                pass
            else:
                chosen.append(sorted_idx[1] + 1)
        mask_out = np.isin(labeled, chosen)
        return (mask_out.astype(np.uint8) * 255)

    def __init__(self):
        self.debug = []

    def forward(self, img, min_area=1000):

        # 大津法阈值分割二值化
        binary = self.blocked_threshold(img, block_size=3, threshold_ratio=1.2)
        no_bg = self.remove_background_2d(binary)
        cleaned = self.remove_small_objects(no_bg, min_area=min_area)
        # 选择最大的两个连通域作为肺叶区域
        selected = self.select_lobes(cleaned, lobes_num=2, connectivity=4)
        # 使用选出的两个连通域继续后续处理
        filled = self.fill_holes_2d(selected)
        closed = self.morph_close_2d(filled, kernel_size=5)
        lung_mask = self.keep_largest_objects_2d(closed)
        mask_closed = self.morph_close_2d(lung_mask, kernel_size=9)

        self.debug = [binary, no_bg, cleaned, selected, filled, closed, lung_mask, mask_closed]

        return mask_closed

from skimage.filters import threshold_otsu
from skimage.morphology import disk, opening
from scipy.ndimage import binary_fill_holes
from collections import deque

class NoduleSegment:

    @staticmethod
    def segment_nodule(img_lung):

        # 计算Otsu阈值（排除零值区域） - 替代MATLAB的graythresh。注意: 这里假设img_lung是8位图像(0-255)
        non_zero_values = img_lung[img_lung > 0]
        if len(non_zero_values) > 0:
            threshold = threshold_otsu(non_zero_values)
        
        # 二值化 - 替代MATLAB的imbinarize
        nodule_mask = img_lung > threshold
        # 形态学操作，创建结构元素 - 替代MATLAB的strel('disk', 3)
        se = disk(4)
        # 开运算 - 替代MATLAB的imopen
        nodule_mask = opening(nodule_mask, se)
        # 填充孔洞 - 替代MATLAB的imfill
        nodule_mask = binary_fill_holes(nodule_mask)

        return nodule_mask.astype(bool)

    @staticmethod
    def region_grow_from_seeds(img, seed_mask, tol=(15, 15), connectivity=8, max_size=None):
        """
        基于种子区域的区域生长（灰度范围相对 seed 均值 ± tol）。
        img: 灰度 uint8 图像
        seed_mask: bool/0-255 掩膜，非零像素作为初始种子
        tol: 容差（int），像素强度必须在 seed_mean ± tol 范围内才会被吸纳
        connectivity: 4 或 8
        max_size: 可选，区域生长的最大像素数量（防止无限扩张）
        返回 bool mask
        """
        h, w = img.shape
        seeds_idx = np.transpose(np.nonzero(seed_mask > 0))
        if seeds_idx.size == 0:
            return np.zeros_like(seed_mask, dtype=bool)

        visited = np.zeros((h, w), dtype=bool)
        out = np.zeros((h, w), dtype=bool)

        # 八/四连通邻居偏移
        if connectivity == 8:
            neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        else:
            neigh = [(-1,0),(0,-1),(0,1),(1,0)]

        # 对每个连通种子块按连通域处理（避免同一块重复生长）
        labeled, num = label(seed_mask > 0)
        for lbl in range(1, num+1):
            yx = np.transpose(np.nonzero(labeled == lbl))
            if yx.size == 0:
                continue
            # 种子块均值作为参考
            seed_vals = img[labeled == lbl]
            seed_mean = float(seed_vals.mean())
            low = max(0, seed_mean - tol[0])
            high = min(255, seed_mean + tol[1])

            # 初始化队列：把种子块所有像素加入
            q = deque()
            for (r,c) in yx:
                if not visited[r,c]:
                    visited[r,c] = True
                    out[r,c] = True
                    q.append((r,c))

            # BFS 生长
            while q:
                r,c = q.popleft()
                for dr,dc in neigh:
                    nr, nc = r+dr, c+dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc]:
                        continue
                    val = img[nr, nc]
                    if low <= val <= high:
                        visited[nr, nc] = True
                        out[nr, nc] = True
                        q.append((nr, nc))
                        if max_size is not None and out.sum() >= max_size:
                            return out
                    else:
                        visited[nr, nc] = True  # 标记已访问避免重复检查
        return out

    @staticmethod
    def get_center_of_mass(mask, radius=1):
        '''计算二值掩膜的质心列表，返回质心点的散点图'''
        labeled, num = label(mask > 0)
        center_map = np.zeros_like(mask, dtype=np.uint8)
        for lbl in range(1, num+1):
            yx = np.transpose(np.nonzero(labeled == lbl))
            if yx.size == 0:
                continue
            com = yx.mean(axis=0).astype(int)
            center_map[com[0], com[1]] = 255
        # 膨胀质心点为小圆
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        center_map = cv2.dilate(center_map, kernel)
        return center_map
    
    def __init__(self):
        self.debug = []

    def forward(self, img_lung):
        nodule_mask = self.segment_nodule(img_lung).astype(np.uint8) * 255
        nodule_mask_seed = self.get_center_of_mass(nodule_mask, radius=3)
        nodule_mask_detailed = self.region_grow_from_seeds(img_lung, nodule_mask_seed, tol=(75, 15), connectivity=4, max_size=5000).astype(np.uint8) * 255

        self.debug = [img_lung, nodule_mask, nodule_mask_seed, nodule_mask_detailed]

        return nodule_mask_detailed

class Main:

    @staticmethod
    def draw_contours_on_image(img, mask, color=(0, 0, 255)):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, color, 2)
        return img

    @staticmethod
    def stack_images(imgs, cols=3):
        # 将多图拼接为网格
        h_imgs = []
        row = []
        max_h = max(i.shape[0] for i in imgs)
        max_w = max(i.shape[1] for i in imgs)
        for idx, im in enumerate(imgs):
            # 将 float 图（如 label2rgb 的输出）从 [0,1] -> uint8 [0,255]
            if im.dtype == np.float32 or im.dtype == np.float64:
                im = np.clip(im, 0.0, 1.0)
                im = (im * 255).astype(np.uint8)
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
    def main(cls, name: str='cov_1.jpg'):
        # 默认从当前脚本所在文件夹读取 cov_1.jpg，并输出为 lung_mask_final.png
        img = cv2.imread(name, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        lungSegment = LungSegment()
        lung_mask = lungSegment.forward(img)
        lung_debug = lungSegment.debug
        # 使用肺区域掩膜
        img_lung = img * (lung_mask > 0)
        noduleSegment = NoduleSegment()
        nodule_mask = noduleSegment.forward(img_lung)
        nodule_debug = noduleSegment.debug

        # 绘制分割边缘
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        image_lung = cls.draw_contours_on_image(img_color, lung_mask, color=(255, 0, 0))
        image_lung_nodule = cls.draw_contours_on_image(image_lung, nodule_mask)

        # 可视化
        cv2.imshow('Lung Mask Pipeline', cls.stack_images(lung_debug, cols=4))
        cv2.imshow('Nodule Segmentation Pipeline', cls.stack_images(nodule_debug, cols=2))
        cv2.imshow('Nodule Contours', image_lung_nodule)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

import typer
if __name__ == "__main__":
    typer.run(Main.main)
