import cv2
import sys

import numpy as np
from scipy import ndimage
import warnings

class KalmanFilter:

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """
        Convert top-left width-height format to center xy width-height format
        tlwh: [x, y, w, h] - top-left coordinates and dimensions
        returns: [cx, cy, w, h] - center coordinates and dimensions
        """
        x, y, w, h = tlwh
        cx = x + w / 2.0
        cy = y + h / 2.0
        return [cx, cy, w, h]

    @staticmethod
    def xywh_to_tlwh(xywh):
        """
        Convert center xy width-height format to top-left width-height format
        xywh: [cx, cy, w, h] - center coordinates and dimensions
        returns: [x, y, w, h] - top-left coordinates and dimensions
        """
        cx, cy, w, h = xywh
        x = cx - w / 2.0
        y = cy - h / 2.0
        return [x, y, w, h]

    @classmethod
    def kalman_filter_factory(cls, tlwh):
        """
        Initialize Kalman Filter with initial detection
        tlwh: [x, y, w, h] - top-left coordinates and dimensions
        """
        
        # Convert to xywh format for Kalman filter
        xywh = cls.tlwh_to_xywh(tlwh)
        
        # State: [cx, cy, w, h, d_cx, d_cy, d_w, d_h]
        x_ = np.concatenate((np.array(xywh), [0, 0, 0, 0]))
        P_ = np.eye(8)
        
        # State transition matrix (constant velocity model)
        F = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1]])
        
        # Process noise covariance
        Q = np.eye(8) * 0.1
        
        # Measurement matrix
        H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1]])
        
        # Measurement noise covariance
        R = np.eye(8) * 0.1
        
        kf = cls(x_, P_, F, Q, H, R)
        return kf

    def __init__(self, x_, P_, F, Q, H, R):
        # previous state
        self.x_ = x_
        # previous covariance matrix
        self.P_ = P_
        # state transition matrix
        self.F = F
        # process covariance matrix
        self.Q = Q
        # measurement matrix
        self.H = H
        # measurement covariance matrix
        self.R = R
        self.S = None
        # kalman gain
        self.K = None
        # predict internal status
        self.x_p = None
        self.P_p = None

    def predict(self):
        """
        x = F * x_                                (2)
        P = F * P_ * F^T + Q                      (3)
        """
        # predict state
        self.x_p = np.dot(self.F, self.x_)
        self.P_p = np.dot(np.dot(self.F, self.P_), self.F.T) + self.Q

    def update(self, xywh):
        """
        xywh: new detection info

        x' = H^-1 * 𝜇 = x + K' * (Z - H * x)    (10)
        P' = P - K' * H * P                     (11)
        S = H * P * H^T + R                       (4)
        K' = P * H^T * (H * P * H^T + S)^-1
        """
        # kalman gain
        self.S = np.dot(np.dot(self.H, self.P_p), self.H.T) + self.R
        self.K = np.dot(np.dot(self.P_p, self.H.T),
                        np.linalg.inv(np.dot(np.dot(self.H,
                                                    self.P_p),
                                             self.H.T)
                                      + self.S))

        # transition x to sensor reading
        # x = [x, y, a, h, d_x, d_y, d_a, d_h]
        # Z = H * x
        deriv = np.array(xywh) - self.x_[:4]
        Z = np.concatenate((xywh, deriv))

        # optimization
        self.x_ = self.x_p + np.dot(self.K, (Z - np.dot(self.H, self.x_p)))
        self.P_ = self.P_p - np.dot(self.K, np.dot(self.H, self.P_p))

        return self.x_


class AutoSelector:

    ENABLE_DEBUG = False

    @classmethod
    def auto_select(cls, img, min_area=800, morph_radius=4, aspect_ratio_min=1.2, 
                                aspect_ratio_max=4, y_ratio_max=0.8):

        img_h, img_w = img.shape[:2]
        
        # 2. 图像预处理（灰度化+对比度增强+降噪）
        if len(img.shape) == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img.copy()
        
        # 自适应直方图均衡化（CLAHE）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhance = clahe.apply(gray_img)
        
        # 高斯降噪
        gray_smooth = cv2.GaussianBlur(gray_enhance, (5, 5), 1.5)
        
        if cls.ENABLE_DEBUG: cv2.imshow("Gray Smooth", gray_smooth)
        
        # 3. 目标分割（自适应阈值+形态学处理）
        # Otsu阈值分割
        _, bw = cv2.threshold(gray_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = 255 - bw  # 车辆为暗区域则取反
        
        # 形态学处理：去除小噪声+填补车辆孔洞
        # 过滤小区域
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        
        min_size = min_area
        bw_filtered = np.zeros((output.shape), dtype=np.uint8)
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                bw_filtered[output == i + 1] = 255
        
        # 闭运算（矩形核）
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        bw_closed = cv2.morphologyEx(bw_filtered, cv2.MORPH_CLOSE, kernel_rect)
        
        # 填充孔洞
        bw_filled = ndimage.binary_fill_holes(bw_closed).astype(np.uint8) * 255
        
        if cls.ENABLE_DEBUG: cv2.imshow("BW Filled", bw_filled)
        
        # 4. 特征筛选：过滤树木，仅保留车辆目标
        # 寻找连通区域
        contours, _ = cv2.findContours(bw_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_vehicles = []
        valid_contours = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # 获取边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 计算中心点
            M = cv2.moments(contour)
            if M['m00'] != 0:
                centroid_y = int(M['m01'] / M['m00'])
            else:
                centroid_y = y + h // 2
            
            # 计算长宽比
            aspect_ratio = w / h if h > 0 else 0
            
            # 筛选规则
            rule1 = (aspect_ratio >= aspect_ratio_min) and (aspect_ratio <= aspect_ratio_max)
            rule2 = (centroid_y / img_h) <= y_ratio_max
            rule3 = h < img_h / 2
            
            # 满足所有规则则判定为车辆
            if rule1 and rule2 and rule3:
                valid_vehicles.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'centroid_y': centroid_y,
                    'aspect_ratio': aspect_ratio,
                    'contour': contour
                })
                valid_contours.append(contour)
        
        # 若筛选出多个车辆，取面积最大的（主目标）
        if len(valid_vehicles) > 0:
            if len(valid_vehicles) > 1:
                max_idx = np.argmax([v['area'] for v in valid_vehicles])
                selected_vehicle = valid_vehicles[max_idx]
            else:
                selected_vehicle = valid_vehicles[0]
            
            bbox = selected_vehicle['bbox']
            contour = selected_vehicle['contour']
        else:
            warnings.warn('未检测到符合条件的车辆目标，请调整筛选参数！')
            return None, None, None
        
        # 6. 裁剪并保存车辆模板
        x, y, w, h = bbox
        # 确保边界框在图像范围内
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        vehicle_template = img[y:y+h, x:x+w]

        return vehicle_template, bbox, contour


class AutoSelectorTester:
    @classmethod
    def main(cls, img_path='473.bmp', template_save_path='vehicle_template.bmp'):
        """
        车辆检测与模板裁剪函数
        
        参数:
        img_path: 输入图像路径
        template_save_path: 模板保存路径
        min_area: 最小连通区域面积
        morph_radius: 形态学处理核半径
        aspect_ratio_min: 车辆最小长宽比
        aspect_ratio_max: 车辆最大长宽比
        y_ratio_max: 目标y坐标上限
        display: 是否显示中间处理过程
        
        返回:
        vehicle_template: 裁剪后的车辆模板
        bbox: 边界框信息 [x, y, width, height]
        """
        
        # 1. 读取原始图像
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"无法读取图像: {img_path}")
        except Exception as e:
            raise ValueError(f"图像读取失败: {str(e)}")
        
        vehicle_template, bbox, contour = AutoSelector.auto_select(img)
        
        print(f'目标框选坐标：x={bbox[0]}, y={bbox[1]}, 宽度={bbox[2]}, 高度={bbox[3]}')

        # 5. 绘制框选结果
        if True:
            result_img = img.copy()
            # 绘制车辆轮廓
            cv2.drawContours(result_img, [contour], -1, (255, 0, 0), 2)
            # 绘制绿色外接框
            x, y, w, h = bbox
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('Selected car without trees', result_img)

        # 7. 显示模板
        cv2.imshow('The template', vehicle_template)
        
        # 保存模板
        if template_save_path:
            # 转换为BGR格式保存
            template_bgr = cv2.cvtColor(vehicle_template, cv2.COLOR_RGB2BGR)
            cv2.imwrite(template_save_path, template_bgr)
        
        # 8. 输出结果信息
        print(f'车辆模板已保存至：{template_save_path}')
        
        if vehicle_template is not None:
            print(f"成功检测到车辆，模板尺寸：{vehicle_template.shape}")
        else:
            print("未检测到车辆")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class Player:
    # 在__init__外定义的是类变量，而不是实例变量。这意味着所有 Player 的实例共享同一个 image、paused 和 frame_count！
    def __init__(self):
        self.image = None
        self.paused = True  # <-- 初始暂停，等待用户框选
        self.frame_count = 0


class MouseSelector:

    def __init__(self):
        # 在__init__外面定义的是类变量，而不是实例变量。！

        '''inner variable'''
        self._original_point = (0, 0)
        self._process_point = (0, 0)

        '''parameter'''

        '''signal'''
        self.selecting_view = None
        self.rect_image = None
        self.tlwh = None

    def hold(self, x, y):
        self._original_point = (x, y)
        self._process_point = self._original_point
        print(f"Mouse down at {self._original_point}")

    def drag(self, x, y, image):
        self.selecting_view = image.copy()
        self._process_point = (x, y)
        if self._original_point != self._process_point:
            cv2.rectangle(self.selecting_view, self._original_point, self._process_point, (0, 0, 255), 2)
    
    def release(self, image):
        # 归一化坐标，防止反向拖动或超出边界
        x0 = max(0, min(self._original_point[0], self._process_point[0]))
        x1 = min(image.shape[1], max(self._original_point[0], self._process_point[0]))
        y0 = max(0, min(self._original_point[1], self._process_point[1]))
        y1 = min(image.shape[0], max(self._original_point[1], self._process_point[1]))
        w = x1 - x0
        h = y1 - y0
        print(f"Selection: ({x0},{y0}) -> ({x1},{y1}), size: {w}x{h}")
        if w > 0 and h > 0:
            self.rect_image = image[y0:y1, x0:x1].copy()
            self.tlwh = [x0, y0, w, h]
            cv2.imshow("Sub Image", self.rect_image)
            print(f"Template size: {self.rect_image.shape}")
        else:
            self.rect_image = None
            print("Invalid selection (size/out of bounds).")
            # 如果需要自动选择，取消注释下面的代码
            self.rect_image, bbox, _ = AutoSelector.auto_select(image)
            cv2.imshow("Sub Image", self.rect_image)
            print(f"Template size: {self.rect_image.shape}")
            self.tlwh = list(bbox)
        

class Tracker:

    @staticmethod
    def image_to_edge(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edge = cv2.Canny(blur, 50, 150)
        return edge

    @staticmethod
    def detect(image, template, search_region=None):
        """
        Template matching with optional search region
        search_region: (x, y, w, h) defining region to search in
        即使image, template可以用成员变量，但是函数本身没有记忆性，因此用传入argument以区分
        """
        if template is None:
            return False, None, None, template
        
        th, tw = template.shape[:2]
        ih, iw = image.shape[:2]
        
        if not (ih >= th and iw >= tw and th > 0 and tw > 0):
            return False, None, None, template
        
        if search_region is not None:
            # Extract search region from image
            sx, sy, sw, sh = search_region
            search_image = image[sy:sy+sh, sx:sx+sw]
            
            # Perform template matching in search region
            res = cv2.matchTemplate(search_image, template, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            # Adjust coordinates back to full image
            min_loc = (min_loc[0] + sx, min_loc[1] + sy)
            
        else:
            # Full image search # 使用方差归一化，越小越好
            res = cv2.matchTemplate(image, template, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        top_left = min_loc # 按左上角查找位置
        bottom_right = (top_left[0] + tw, top_left[1] + th)
        detected = False

        if min_val < 0.3:  # 可加阈值避免错误匹配
            detected = True

        return detected, top_left, bottom_right
    
    @staticmethod
    def rescan(image, rect_image, top_left, bottom_right):
        updated_template = rect_image
        # 从原始帧中提取匹配区域（绘制前）
        matched_color = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].copy()
        # 仅从提取的（未绘制的）区域更新模板
        th, tw = rect_image.shape[:2]
        if matched_color.shape[0] == th and matched_color.shape[1] == tw:
            updated_template = matched_color.copy()
            print(f"Template updated")
        return updated_template
    
    def on_mouse_callback(self, event, x, y, flags, param):
        '''
        回调函数，是需要遵照特定的写法的，输入、输出服从其注册机制。
        因此，比起普通的函数：
        argument ->|        |->return
        parameter->|FUNCTION|->signal
        parameter和signal作为成员变量
        parameter方便调整
        signal方便追踪
        回调函数的形式是
        read     ->|        |->write
        argument ->|CALLBACK|->return
        parameter->|FUNCTION|->signal
        read/write和parameter本质都是类的属性，parameter可以作为dataclass成员，主动优化的parameter是在pytorch.module的__init__中注册为nn.parameter
        read/write放在__init__中，__init__用于绑定各个对象之间的关联
        以后哪怕来了别的什么东西，也可以放进__init__，例如我这里的parameter实际上是hyperparameter
        '''
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_button_down = True
            self.player.paused = True  # 绘制时暂停播放
            self.selector.hold(x,y)
        elif event == cv2.EVENT_MOUSEMOVE and self.left_button_down:
            if self.player.image is None:
                return
            self.selector.drag(x,y,self.player.image)
            cv2.imshow("Man", self.selector.selecting_view)
        elif event == cv2.EVENT_LBUTTONUP:
            self.left_button_down = False
            if self.player.image is None:
                return
            self.selector.release(self.player.image)
            self.kf = KalmanFilter.kalman_filter_factory(self.selector.tlwh)
            # self.player.paused = False  # 完成选取后自动继续播放并开始跟踪。这个代码注释掉才正常工作

    def __init__(self, fp):
        self.cap = cv2.VideoCapture(fp)
        if not self.cap.isOpened():
            print("Cannot open video:", fp)
            sys.exit(1)

        cv2.namedWindow("Man", cv2.WINDOW_NORMAL)

        '''parameter'''
        self.SEARCH_WINDOW_SIZE = 100  # Search window size around prediction

        '''Create/Bind Object'''
        self.player = Player()
        
        # 读取并显示第一帧（保持初始暂停）
        ret, self.player.image = self.cap.read()
        if not ret or self.player.image is None:
            print("Empty video or cannot read frame.")
            self.cap.release()
            sys.exit(1)
            
        self.player.frame_count = 1
        cv2.imshow("Man", self.player.image)

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps:  # 检查是否为nan
            fps = 30.0
        self.play_wait_ms = max(1, int(1000.0 / fps))
        self.idle_wait_ms = 30  # 暂停时的 waitKey 间隔，保证响应鼠标/键盘

        self.selector = MouseSelector()
        self.kf = KalmanFilter.kalman_filter_factory([0,0,*self.player.image.shape[:2]])

        self.left_button_down = False
        cv2.setMouseCallback("Man", self.on_mouse_callback)
        
        print("Instructions:")
        print("1. Drag mouse to select tracking target")
        print("2. Press SPACE to pause/play")
        print("3. Press Q or ESC to quit")

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        cv2.destroyAllWindows()

    def predict(self):
        th, tw = self.selector.rect_image.shape[:2]
        ih, iw = self.player.image.shape[:2]

        self.kf.predict()
        
        # Get predicted state and convert to top-left format
        predicted_xywh = self.kf.x_p[:4].tolist()
        predicted_tlwh = self.kf.xywh_to_tlwh(predicted_xywh)
        pred_x, pred_y, pred_w, pred_h = predicted_tlwh
        
        # Define search window around prediction
        search_margin = self.SEARCH_WINDOW_SIZE // 2
        search_x = max(0, int(pred_x - search_margin))
        search_y = max(0, int(pred_y - search_margin))
        search_w = min(iw - search_x, tw + self.SEARCH_WINDOW_SIZE)
        search_h = min(ih - search_y, th + self.SEARCH_WINDOW_SIZE)
        
        search_region = (0, 0, iw, ih)
        # Ensure search window is valid
        if search_w >= tw and search_h >= th:
            search_region = (search_x, search_y, search_w, search_h)
        
        return search_region

    def update(self, top_left):
        th, tw = self.selector.rect_image.shape[:2]
        ih, iw = self.player.image.shape[:2]
        # Convert detection to xywh format for Kalman update
        detected_tlwh = [top_left[0], top_left[1], tw, th]
        detected_xywh = self.kf.tlwh_to_xywh(detected_tlwh)
        
        # kf.update()
        self.kf.update(detected_xywh)

    def loop(self):
        # 确定等待时间
        if self.player.paused or self.left_button_down:
            wait_time = self.idle_wait_ms
        else:
            wait_time = self.play_wait_ms
        key = cv2.waitKey(wait_time) & 0xFF
        if key == 27 or key == ord('q'):
            return False
        elif key == ord(' '):  # 空格切换暂停/播放
            self.player.paused = not self.player.paused
            print(f"{'Paused' if self.player.paused else 'Playing'}")
        elif key == ord('r'):  # 重置选择
            self.selector.rect_image = None
            self.player.paused = True
            print("Template reset, please select new target")
        
        # 如果正在选择或暂停，不读取新帧
        if not (self.left_button_down or self.player.paused):
            ret, self.player.image = self.cap.read()
            if not ret or self.player.image is None:
                print("End of video or cannot read frame.")
                return False
            self.player.frame_count += 1
            
        if self.player.image is None:
            return False
        
        # 只有当有模板时才进行检测
        if self.selector.rect_image is not None:
            search_region = self.predict()
            detected, top_left, bottom_right = self.detect(self.player.image, self.selector.rect_image, search_region)
            if detected:
                self.update(top_left)
                self.selector.rect_image = self.rescan(self.player.image, self.selector.rect_image, top_left, bottom_right)  # 更新模板
        else:
            detected = False

        # 创建显示副本，不在原始图像上绘制
        display_frame = self.player.image.copy()
        if detected:
            cv2.rectangle(display_frame, top_left, bottom_right, (0, 0, 255), 2)
            search_x, search_y, search_w, search_h = search_region
            # Draw search window for visualization (in green)
            cv2.rectangle(display_frame, (search_x, search_y),
                        (search_x + search_w, search_y + search_h), (0, 255, 0), 1)

            
        cv2.putText(display_frame, f"Frame: {self.player.frame_count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, "Paused" if self.player.paused else "Playing", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 255, 0) if not self.player.paused else (0, 0, 255), 2)
        
        # 显示图像
        if self.left_button_down and self.selector.selecting_view is not None:
            cv2.imshow("Man", self.selector.selecting_view)
        else:
            cv2.imshow("Man", display_frame)

        return True


class TrackerTester:
    @classmethod
    def main(cls, video_file: str='assets/track/track_1.mp4'):
        """Usage: python template_tracking.py <video_file> (default: car.mp4)"""
        tracker = Tracker(video_file)
        try:
            while True:
                going = tracker.loop()
                if not going:
                    break
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
        finally:
            print("Program ended")


import typer
if __name__ == "__main__":
    typer.run(TrackerTester.main)
