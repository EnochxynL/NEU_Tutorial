import cv2
import sys
from icecream import ic

# class Player:
#     image = None
#     paused = True  # <-- 初始暂停，等待用户框选
#     frame_count = 0
#     定义的是类变量，而不是实例变量。这意味着所有 Player 的实例共享同一个 image、paused 和 frame_count！
class Player:
    def __init__(self):
        self.image = None
        self.paused = True
        self.frame_count = 0

class MouseSelector:
    # inner variable
    _original_point = (0, 0)
    _process_point = (0, 0)

    # parameter
    
    # signal
    left_button_down = False
    selecting_view = None
    rect_image = None

    # read and write
    def __init__(self, player: Player):
        self.player = player
        self.image = player.image

    # callbackmethod
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
            self._original_point = (x, y)
            self._process_point = self._original_point
        elif event == cv2.EVENT_MOUSEMOVE and self.left_button_down:
            if self.image is None:
                return
            self.selecting_view = self.image.copy()
            self._process_point = (x, y)
            if self._original_point != self._process_point:
                cv2.rectangle(self.selecting_view, self._original_point, self._process_point, (0, 0, 255), 2)
            cv2.imshow("Man", self.selecting_view)
        elif event == cv2.EVENT_LBUTTONUP:
            self.left_button_down = False
            if self.image is None:
                return
            # 归一化坐标，防止反向拖动或超出边界
            x0 = max(0, min(self._original_point[0], self._process_point[0]))
            x1 = min(self.image.shape[1], max(self._original_point[0], self._process_point[0]))
            y0 = max(0, min(self._original_point[1], self._process_point[1]))
            y1 = min(self.image.shape[0], max(self._original_point[1], self._process_point[1]))
            w = x1 - x0
            h = y1 - y0
            if w > 0 and h > 0:
                self.rect_image = self.image[y0:y1, x0:x1].copy()
                cv2.imshow("Sub Image", self.rect_image)
                # self.player.paused = False  # 完成选取后自动继续播放并开始跟踪
            else:
                self.rect_image = None
                print("Invalid selection (size/out of bounds). Select automatically.")
                from auto_select import auto_select
                self.rect_image, _, _ = auto_select(self.image)

class Tracker:

    @staticmethod
    def detect(image, rect_image):
        '''即使image, rect_image可以用成员变量，但是函数本身没有记忆性，因此用传入argument以区分'''
        # 仅在模板存在时运行边缘匹配
        if rect_image is None:
            return False, None, None
        th, tw = rect_image.shape[:2]
        ih, iw = image.shape[:2]
        if not (ih >= th and iw >= tw and th > 0 and tw > 0):
            return False, None, None

        # 使用方差归一化，越小越好
        res = cv2.matchTemplate(image, rect_image, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = min_loc # 按左上角查找位置
        bottom_right = (top_left[0] + tw, top_left[1] + th)
        detected = False

        if min_val < 0.3: # 可加阈值避免吃到错误匹配
            detected = True

        return detected, top_left, bottom_right

    def __init__(self, fp):
        cap = cv2.VideoCapture(fp)
        if not cap.isOpened():
            print("Cannot open video:", fp)
            return
        
        self.player = Player()
        self.selector = MouseSelector(self.player)

        cv2.namedWindow("Man", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Man", self.selector.on_mouse_callback)

        # 读取并显示第一帧（保持初始暂停）
        ret, image = cap.read()
        if not ret or image is None:
            print("Empty video or cannot read frame.")
            cap.release()
            return
        self.player.frame_count = 1
        cv2.imshow("Man", image)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps:
            fps = 30.0
        self.play_wait_ms = max(1, int(1000.0 / fps))
        self.idle_wait_ms = 30  # 暂停时的 waitKey 间隔，保证响应鼠标/键盘
        self.cap = cap
        # self.player.image = image # BUG: 这个代码忘记了，图片就无法显示。局部变量和实例变量要分清！

        # while True:
        #     pass
    
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def loop(self):
        ic(self.selector.left_button_down, self.player.paused)
        # BUG: 由于初始paused为True，不读取图片；而第一帧读取给了局部变量image，没给到self.player.image，因此图片无法显示。而又因为如此，直接return了，无法切换状态
        if not (self.selector.left_button_down or self.player.paused):
            print("read a frame")
            ret, self.player.image = self.cap.read()
            if not ret or self.player.image is None:
                print("End of video or cannot read frame.")
                return False
            self.player.frame_count += 1
        if self.player.image is None:
            print("Cannot read frame.")
            return False
        
        detected, top_left, bottom_right = self.detect(self.player.image, self.selector.rect_image)
        
        # create a display copy so we don't modify `image` used for matching
        display_frame = self.player.image.copy()
        if detected: # draw rectangle only on display copy
            cv2.rectangle(display_frame, top_left, bottom_right, (0, 0, 255), 2)

            # extract matched region FROM THE ORIGINAL FRAME BEFORE DRAWING
            matched_color = self.player.image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].copy()
            # update template only from the extracted (undrawn) patch
            th, tw = self.rect_image.shape[:2]
            if matched_color.shape[0] == th and matched_color.shape[1] == tw:
                self.rect_image = matched_color.copy()

        cv2.putText(display_frame, f"Current frame is: {self.player.frame_count}", (50, 80),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255), 2)

        if self.selector.left_button_down and self.selector.selecting_view is not None:
            cv2.imshow("Man", self.selector.selecting_view)
        else:
            # show the display copy if created, else show original
            try:
                cv2.imshow("Man", display_frame)
            except UnboundLocalError:
                cv2.imshow("Man", self.player.image)

        wait_time = self.idle_wait_ms if (self.player.paused or self.selector.left_button_down) else self.play_wait_ms
        key = cv2.waitKey(wait_time) & 0xFF
        if key == 27 or key == ord('q'):
            return False
        elif key == ord(' '):  # 空格切换暂停/播放
            self.player.paused = not self.player.paused
        
        return True

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("Usage: python template_tracking.py <video_file>")
    #     exit()
    tracker = Tracker('car.mp4') # sys.argv[1])
    while True:
        going = tracker.loop()
        # if not going:
        #     break
