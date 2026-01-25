# import the necessary packages
from skimage.color import label2rgb
import numpy as np
import imutils
import cv2

from skimage.segmentation import watershed
from scipy import ndimage

class Count:

    @staticmethod
    def blocked_threshold(gray, block_size=3):
        h, w = gray.shape
        thresh = np.zeros_like(gray)
        for i in range(block_size):
            for j in range(block_size):
                block = gray[i * h // block_size:(i + 1) * h // block_size, j * w // block_size:(j + 1) * w // block_size]
                _, block_thresh = cv2.threshold(block, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                thresh[i * h // block_size:(i + 1) * h // block_size, j * w // block_size:(j + 1) * w // block_size] = block_thresh
        return thresh

    @staticmethod
    # 不指定文件路径，作为替代实现参考
    def get_foreground_mask_with_threshold(distance_transform, background_mask=None):
        _, sure_fg = cv2.threshold(distance_transform, 0.3 * distance_transform.max(), 255, 0)
        # 对fg开运算去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_ERODE, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_DILATE, kernel, iterations=1)
        return np.uint8(sure_fg)

    @staticmethod
    def get_segmentation(distance_transform, foreground_mask, not_background_mask):
        '''
        Docstring for get_segmentation_markers
        
        :param foreground_mask: Description
        '''
        # 对局部峰值执行连通组件分析，使用8连通性，然后应用分水岭算法
        markers, _ = ndimage.label(foreground_mask, structure=np.ones((3, 3)))
        labels = watershed(-distance_transform, markers, mask=not_background_mask.astype(bool))  # 已做的分割
        print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
        return labels

    def __init__(self):
        self.debug = []

    def forward(self, image):
        # 加载图像并执行金字塔均值漂移滤波以辅助阈值分割
        shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
        # 将均值漂移结果转换为灰度图，然后应用大津阈值分割
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        # 把图像分四块进行大津法阈值分割，然后合并结果

        # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # blockSize = int(min(gray.shape[:2]) * 0.6) * 2 + 1
        # thresh = cv2.adaptiveThreshold(gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, 1)
        thresh = self.blocked_threshold(gray, block_size=3)

        # 移除自适应阈值的背景

        # # 形态学开运算以去除小噪声
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # 通过膨胀获得确定的背景区域
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # 计算距离变换并通过阈值分割获得确定的前景区域
        dist_trans = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

        dist_trans_colored = cv2.applyColorMap(cv2.normalize(dist_trans, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)

        # 将峰值坐标转换为与 D 相同大小的布尔掩码
        sure_fg = self.get_foreground_mask_with_threshold(dist_trans, thresh)

        # 未知区域（既不是确定的背景也不是确定的前景）
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 
        labels = self.get_segmentation(dist_trans, sure_fg, thresh)
        
        labels_colored = label2rgb(labels, image=gray, bg_label=0)       # 标签->RGB，叠加原图可选

        self.debug = [
            thresh,
            # opening,
            sure_bg,
            dist_trans_colored,
            sure_fg,
            labels_colored,
            # image_counted
        ]

        return labels


class Main:

    @staticmethod
    def annotate_circle(image, labels, thickness=2, gray=None):

        if gray is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image_counted = image.copy()
        
        # 遍历分水岭算法返回的唯一标签
        for label in np.unique(labels):
            # 所有标签加一，使背景为1而不是0，并将未知区域标记为0
            if label == 0:
                continue

            # 否则，为标签区域分配内存并在掩码上绘制它
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255

            # 在掩码中检测轮廓并获取最大的一个
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)

            # 绘制包围对象的圆
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(image_counted, (int(x), int(y)), int(r), (0, 255, 0), thickness)
            cv2.putText(image_counted, "{}".format(int(label - 1)), (int(x) - 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), thickness)

        return image_counted

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
    def main(cls, name: str="Dowels.tif"):
        image = cv2.imread(name)

        count = Count()
        if True:
            count_labels = count.forward(image)
            count_debug = count.debug
        del count
        
        image_counted = cls.annotate_circle(image, count_labels, thickness=1)

        grid = cls.stack_images([*count_debug,image_counted])

        # 显示输出图像
        cv2.imshow("Output", grid)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

import typer
if __name__ == "__main__":
    typer.run(Main.main)
