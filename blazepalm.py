import torch
import torch.nn as nn
import torch.nn.functional as F

class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act='relu', skip_proj=False):
        '''在 https://github.com/vidursatija/BlazePalm/blob/master/ML/blazepalm.py 被称为 ResModule'''
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.kernel_size = kernel_size
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch 
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding, 
                      groups=in_channels, bias=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=True),
        )

        if skip_proj:
            self.skip_proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.skip_proj = None

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'prelu':
            self.act = nn.PReLU(out_channels)
        else:
            raise NotImplementedError("unknown activation %s"%act)

    def forward(self, x):
        if self.stride == 2:
            if self.kernel_size==3:
                h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            else:
                h = F.pad(x, (1, 2, 1, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.skip_proj is not None:
            x = self.skip_proj(x)
        elif self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)
        

        return self.act(self.convs(h) + x)


class BlazeBase(nn.Module):
    """ Base class for media pipe models. """

    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return self.classifier_8.weight.device
    
    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()        


class BlazeDetector(BlazeBase):
    """ Base class for detector models.

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/hollance/BlazeFace-PyTorch and
    https://github.com/google/mediapipe/
    """
    @staticmethod
    def resize_pad(img):
        """ resize and pad images to be input to the detectors

        The face and palm detector networks take 256x256 and 128x128 images
        as input. As such the input image is padded and resized to fit the
        size while maintaing the aspect ratio.

        Returns:
            img1: 256x256
            img2: 128x128
            scale: scale factor between original image and 256x256 image
            pad: pixels of padding in the original image
        """

        size0 = img.shape
        if size0[0]>=size0[1]:
            h1 = 256
            w1 = 256 * size0[1] // size0[0]
            padh = 0
            padw = 256 - w1
            scale = size0[1] / w1
        else:
            h1 = 256 * size0[0] // size0[1]
            w1 = 256
            padh = 256 - h1
            padw = 0
            scale = size0[0] / h1
        padh1 = padh//2
        padh2 = padh//2 + padh%2
        padw1 = padw//2
        padw2 = padw//2 + padw%2
        img1 = cv2.resize(img, (w1,h1))
        img1 = np.pad(img1, ((padh1, padh2), (padw1, padw2), (0,0)))
        pad = (int(padh1 * scale), int(padw1 * scale))
        img2 = cv2.resize(img1, (128,128))
        return img1, img2, scale, pad

    @staticmethod
    def denormalize_detections(detections, scale, pad):
        """ maps detection coordinates from [0,1] to image coordinates

        The face and palm detector networks take 256x256 and 128x128 images
        as input. As such the input image is padded and resized to fit the
        size while maintaing the aspect ratio. This function maps the
        normalized coordinates back to the original image coordinates.

        Inputs:
            detections: nxm tensor. n is the number of detections.
                m is 4+2*k where the first 4 valuse are the bounding
                box coordinates and k is the number of additional
                keypoints output by the detector.
            scale: scalar that was used to resize the image
            pad: padding in the x and y dimensions

        """
        detections[:, 0] = detections[:, 0] * scale * 256 - pad[0]
        detections[:, 1] = detections[:, 1] * scale * 256 - pad[1]
        detections[:, 2] = detections[:, 2] * scale * 256 - pad[0]
        detections[:, 3] = detections[:, 3] * scale * 256 - pad[1]

        detections[:, 4::2] = detections[:, 4::2] * scale * 256 - pad[1]
        detections[:, 5::2] = detections[:, 5::2] * scale * 256 - pad[0]
        return detections

    def load_anchors(self, path):
        self.anchors = torch.tensor(np.load(path), dtype=torch.float32, device=self._device())
        assert(self.anchors.ndimension() == 2)
        assert(self.anchors.shape[0] == self.num_anchors)
        assert(self.anchors.shape[1] == 4)

    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        return x.float() / 255.# 127.5 - 1.0

    def predict_on_image(self, img):
        """Makes a prediction on a single image.

        Arguments:
            img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                 shape (3, H, W). The image's height and width should be 
                 128 pixels.

        Returns:
            A tensor with face detections.
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1))

        return self.predict_on_batch(img.unsqueeze(0))[0]

    def predict_on_batch(self, x):
        """Makes a prediction on a batch of images.

        Arguments:
            x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
               shape (b, 3, H, W). The height and width should be 128 pixels.

        Returns:
            A list containing a tensor of face detections for each image in 
            the batch. If no faces are found for an image, returns a tensor
            of shape (0, 17).

        Each face detection is a PyTorch tensor consisting of 17 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 6 keypoints
            - confidence score
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))

        assert x.shape[1] == 3
        assert x.shape[2] == self.y_scale
        assert x.shape[3] == self.x_scale

        # 1. Preprocess the images into tensors:
        x = x.to(self._device())
        x = self._preprocess(x)

        # 2. Run the neural network:
        with torch.no_grad():
            out = self.__call__(x)

        # 3. Postprocess the raw predictions:
        detections = self._tensors_to_detections(out[0], out[1], self.anchors)

        # 4. Non-maximum suppression to remove overlapping detections:
        filtered_detections = []
        for i in range(len(detections)):
            faces = self._weighted_non_max_suppression(detections[i])
            faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, self.num_coords+1))
            filtered_detections.append(faces)

        return filtered_detections


    def detection2roi(self, detection):
        """ Convert detections from detector to an oriented bounding box.

        Adapted from:
        # mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt

        The center and size of the box is calculated from the center 
        of the detected box. Rotation is calcualted from the vector
        between kp1 and kp2 relative to theta0. The box is scaled
        and shifted by dscale and dy.

        """
        if self.detection2roi_method == 'box':
            # compute box center and scale
            # use mediapipe/calculators/util/detections_to_rects_calculator.cc
            xc = (detection[:,1] + detection[:,3]) / 2
            yc = (detection[:,0] + detection[:,2]) / 2
            scale = (detection[:,3] - detection[:,1]) # assumes square boxes

        elif self.detection2roi_method == 'alignment':
            # compute box center and scale
            # use mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
            xc = detection[:,4+2*self.kp1]
            yc = detection[:,4+2*self.kp1+1]
            x1 = detection[:,4+2*self.kp2]
            y1 = detection[:,4+2*self.kp2+1]
            scale = ((xc-x1)**2 + (yc-y1)**2).sqrt() * 2
        else:
            raise NotImplementedError(
                "detection2roi_method [%s] not supported"%self.detection2roi_method)

        yc += self.dy * scale
        scale *= self.dscale

        # compute box rotation
        x0 = detection[:,4+2*self.kp1]
        y0 = detection[:,4+2*self.kp1+1]
        x1 = detection[:,4+2*self.kp2]
        y1 = detection[:,4+2*self.kp2+1]
        #theta = np.arctan2(y0-y1, x0-x1) - self.theta0
        theta = torch.atan2(y0-y1, x0-x1) - self.theta0
        return xc, yc, scale, theta


    def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
        """The output of the neural network is a tensor of shape (b, 896, 16)
        containing the bounding box regressor predictions, as well as a tensor 
        of shape (b, 896, 1) with the classification confidences.

        This function converts these two "raw" tensors into proper detections.
        Returns a list of (num_detections, 17) tensors, one for each image in
        the batch.

        This is based on the source code from:
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
        """
        assert raw_box_tensor.ndimension() == 3
        assert raw_box_tensor.shape[1] == self.num_anchors
        assert raw_box_tensor.shape[2] == self.num_coords

        assert raw_score_tensor.ndimension() == 3
        assert raw_score_tensor.shape[1] == self.num_anchors
        assert raw_score_tensor.shape[2] == self.num_classes

        assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]
        
        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)
        
        thresh = self.score_clipping_thresh
        raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
        detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)
        
        # Note: we stripped off the last dimension from the scores tensor
        # because there is only has one class. Now we can simply use a mask
        # to filter out the boxes with too low confidence.
        mask = detection_scores >= self.min_score_thresh

        # Because each image from the batch can have a different number of
        # detections, process them one at a time using a loop.
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
            output_detections.append(torch.cat((boxes, scores), dim=-1))

        return output_detections

    def _decode_boxes(self, raw_boxes, anchors):
        """Converts the predictions into actual coordinates using
        the anchor boxes. Processes the entire batch at once.
        """
        boxes = torch.zeros_like(raw_boxes)

        x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax

        for k in range(self.num_keypoints):
            offset = 4 + k*2
            keypoint_x = raw_boxes[..., offset    ] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset    ] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def _weighted_non_max_suppression(self, detections):
        """The alternative NMS method as mentioned in the BlazeFace paper:

        "We replace the suppression algorithm with a blending strategy that
        estimates the regression parameters of a bounding box as a weighted
        mean between the overlapping predictions."

        The original MediaPipe code assigns the score of the most confident
        detection to the weighted detection, but we take the average score
        of the overlapping detections.

        The input detections should be a Tensor of shape (count, 17).

        Returns a list of PyTorch tensors, one for each detected face.
        
        This is based on the source code from:
        mediapipe/calculators/util/non_max_suppression_calculator.cc
        mediapipe/calculators/util/non_max_suppression_calculator.proto
        """
        if len(detections) == 0: return []

        output_detections = []

        # Sort the detections from highest to lowest score.
        remaining = torch.argsort(detections[:, self.num_coords], descending=True)

        while len(remaining) > 0:
            detection = detections[remaining[0]]

            # Compute the overlap between the first box and the other 
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(first_box, other_boxes)

            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            mask = ious > self.min_suppression_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            # Take an average of the coordinates from the overlapping
            # detections, weighted by their confidence scores.
            weighted_detection = detection.clone()
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :self.num_coords]
                scores = detections[overlapping, self.num_coords:self.num_coords+1]
                total_score = scores.sum()
                weighted = (coordinates * scores).sum(dim=0) / total_score
                weighted_detection[:self.num_coords] = weighted
                weighted_detection[self.num_coords] = total_score / len(overlapping)

            output_detections.append(weighted_detection)

        return output_detections    

class BlazePalm(BlazeDetector):
    """The palm detection model from MediaPipe. """
    def __init__(self):
        super(BlazePalm, self).__init__()

        # These are the settings from the MediaPipe example graph
        # mediapipe/graphs/hand_tracking/subgraphs/hand_detection_gpu.pbtxt
        self.num_classes = 1
        self.num_anchors = 2944
        self.num_coords = 18
        self.score_clipping_thresh = 100.0
        self.x_scale = 256.0
        self.y_scale = 256.0
        self.h_scale = 256.0
        self.w_scale = 256.0
        self.min_score_thresh = 0.5
        self.min_suppression_threshold = 0.3
        self.num_keypoints = 7

        # These settings are for converting detections to ROIs which can then
        # be extracted and feed into the landmark network
        # use mediapipe/calculators/util/detections_to_rects_calculator.cc
        self.detection2roi_method = 'box'
        # mediapipe/graphs/hand_tracking/subgraphs/hand_detection_cpu.pbtxt
        self.kp1 = 0
        self.kp2 = 2
        self.theta0 = np.pi/2
        self.dscale = 2.6
        self.dy = -0.5

        self._define_layers()

    def _define_layers(self):
        self.backbone1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),

            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            
            BlazeBlock(32, 64, stride=2),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),

            BlazeBlock(64, 128, stride=2),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),

        )
        
        self.backbone2 = nn.Sequential(
            BlazeBlock(128, 256, stride=2),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
        )

        self.backbone3 = nn.Sequential(
            BlazeBlock(256, 256, stride=2),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
        )

        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)
        self.blaze1 = BlazeBlock(256, 256)

        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)
        self.blaze2 = BlazeBlock(128, 128)

        self.classifier_32 = nn.Conv2d(128, 2, 1, bias=True)
        self.classifier_16 = nn.Conv2d(256, 2, 1, bias=True)
        self.classifier_8 = nn.Conv2d(256, 6, 1, bias=True)
        
        self.regressor_32 = nn.Conv2d(128, 36, 1, bias=True)
        self.regressor_16 = nn.Conv2d(256, 36, 1, bias=True)
        self.regressor_8 = nn.Conv2d(256, 108, 1, bias=True)
        
    def forward(self, x):
        b = x.shape[0]      # batch size, needed for reshaping later

        x = F.pad(x, (0, 1, 0, 1), "constant", 0)

        x = self.backbone1(x)           # (b, 128, 32, 32)        
        y = self.backbone2(x)           # (b, 256, 16, 16)
        z = self.backbone3(y)           # (b, 256, 8, 8)

        y = y + F.relu(self.conv_transpose1(z), True)
        y = self.blaze1(y)

        x = x + F.relu(self.conv_transpose2(y), True)
        x = self.blaze2(x)


        # Note: Because PyTorch is NCHW but TFLite is NHWC, we need to
        # permute the output from the conv layers before reshaping it.
        
        c1 = self.classifier_8(z)       # (b, 2, 16, 16)
        c1 = c1.permute(0, 2, 3, 1)     # (b, 16, 16, 2)
        c1 = c1.reshape(b, -1, 1)       # (b, 512, 1)

        c2 = self.classifier_16(y)      # (b, 6, 8, 8)
        c2 = c2.permute(0, 2, 3, 1)     # (b, 8, 8, 6)
        c2 = c2.reshape(b, -1, 1)       # (b, 384, 1)

        c3 = self.classifier_32(x)      # (b, 6, 8, 8)
        c3 = c3.permute(0, 2, 3, 1)     # (b, 8, 8, 6)
        c3 = c3.reshape(b, -1, 1)       # (b, 384, 1)

        c = torch.cat((c3, c2, c1), dim=1)  # (b, 896, 1)

        r1 = self.regressor_8(z)        # (b, 32, 16, 16)
        r1 = r1.permute(0, 2, 3, 1)     # (b, 16, 16, 32)
        r1 = r1.reshape(b, -1, 18)      # (b, 512, 16)

        r2 = self.regressor_16(y)       # (b, 96, 8, 8)
        r2 = r2.permute(0, 2, 3, 1)     # (b, 8, 8, 96)
        r2 = r2.reshape(b, -1, 18)      # (b, 384, 16)

        r3 = self.regressor_32(x)       # (b, 96, 8, 8)
        r3 = r3.permute(0, 2, 3, 1)     # (b, 8, 8, 96)
        r3 = r3.reshape(b, -1, 18)      # (b, 384, 16)

        r = torch.cat((r3, r2, r1), dim=1)  # (b, 896, 16)

        return [r, c]


class BlazeLandmark(BlazeBase):
    """ Base class for landmark models. """

    def extract_roi(self, frame, xc, yc, theta, scale):

        # take points on unit square and transform them according to the roi
        points = torch.tensor([[-1, -1, 1, 1],
                            [-1, 1, -1, 1]], device=scale.device).view(1,2,4)
        points = points * scale.view(-1,1,1)/2
        theta = theta.view(-1, 1, 1)
        R = torch.cat((
            torch.cat((torch.cos(theta), -torch.sin(theta)), 2),
            torch.cat((torch.sin(theta), torch.cos(theta)), 2),
            ), 1)
        center = torch.cat((xc.view(-1,1,1), yc.view(-1,1,1)), 1)
        points = R @ points + center

        # use the points to compute the affine transform that maps 
        # these points back to the output square
        res = self.resolution
        points1 = np.array([[0, 0, res-1],
                            [0, res-1, 0]], dtype=np.float32).T
        affines = []
        imgs = []
        for i in range(points.shape[0]):
            pts = points[i, :, :3].cpu().numpy().T
            M = cv2.getAffineTransform(pts, points1)
            img = cv2.warpAffine(frame, M, (res,res))#, borderValue=127.5)
            img = torch.tensor(img, device=scale.device)
            imgs.append(img)
            affine = cv2.invertAffineTransform(M).astype('float32')
            affine = torch.tensor(affine, device=scale.device)
            affines.append(affine)
        if imgs:
            imgs = torch.stack(imgs).permute(0,3,1,2).float() / 255.#/ 127.5 - 1.0
            affines = torch.stack(affines)
        else:
            imgs = torch.zeros((0, 3, res, res), device=scale.device)
            affines = torch.zeros((0, 2, 3), device=scale.device)

        return imgs, affines, points

    def denormalize_landmarks(self, landmarks, affines):
        landmarks[:,:,:2] *= self.resolution
        for i in range(len(landmarks)):
            landmark, affine = landmarks[i], affines[i]
            landmark = (affine[:,:2] @ landmark[:,:2].T + affine[:,2:]).T
            landmarks[i,:,:2] = landmark
        return landmarks

class BlazeHandLandmark(BlazeLandmark):
    """The hand landmark model from MediaPipe.
    
    """
    def __init__(self):
        super(BlazeHandLandmark, self).__init__()

        # size of ROIs used for input
        self.resolution = 256

        self._define_layers()

    def _define_layers(self):
        self.backbone1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=0, bias=True),
            nn.ReLU(inplace=True),

            BlazeBlock(24, 24, 5),
            BlazeBlock(24, 24, 5),
            BlazeBlock(24, 48, 5, 2),
        )

        self.backbone2 = nn.Sequential(
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 96, 5, 2),
        )

        self.backbone3 = nn.Sequential(
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5, 2),
        )

        self.backbone4 = nn.Sequential(
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5, 2),
        )

        self.blaze5 = BlazeBlock(96, 96, 5)
        self.blaze6 = BlazeBlock(96, 96, 5)
        self.conv7 = nn.Conv2d(96, 48, 1, bias=True)

        self.backbone8 = nn.Sequential(
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 48, 5),
            BlazeBlock(48, 96, 5, 2),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 96, 5),
            BlazeBlock(96, 288, 5, 2),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5, 2),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5, 2),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5, 2),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
            BlazeBlock(288, 288, 5),
        )

        self.hand_flag = nn.Conv2d(288, 1, 2, bias=True)
        self.handed = nn.Conv2d(288, 1, 2, bias=True)
        self.landmarks = nn.Conv2d(288, 63, 2, bias=True)


    def forward(self, x):
        if x.shape[0] == 0:
            return torch.zeros((0,)), torch.zeros((0,)), torch.zeros((0, 21, 3))

        x = F.pad(x, (0, 1, 0, 1), "constant", 0)

        x = self.backbone1(x)
        y = self.backbone2(x)
        z = self.backbone3(y)
        w = self.backbone4(z)

        z = z + F.interpolate(w, scale_factor=2, mode='bilinear')
        z = self.blaze5(z)

        y = y + F.interpolate(z, scale_factor=2, mode='bilinear')
        y = self.blaze6(y)
        y = self.conv7(y)

        x = x + F.interpolate(y, scale_factor=2, mode='bilinear')

        x = self.backbone8(x)

        hand_flag = self.hand_flag(x).view(-1).sigmoid()
        handed = self.handed(x).view(-1).sigmoid()
        landmarks = self.landmarks(x).view(-1, 21, 3) / 256

        return hand_flag, handed, landmarks


import numpy as np
import math
import cv2
import sys

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

class BlazeRunner:
    def __init__(self):
        self.palm_detector = BlazePalm().to(gpu)
        self.palm_detector.load_weights("blazepalm.pth")
        self.palm_detector.load_anchors("anchors_palm.npy")
        self.palm_detector.min_score_thresh = .75

        self.hand_regressor = BlazeHandLandmark().to(gpu)
        self.hand_regressor.load_weights("blazehand_landmark.pth")

        # empty_tensor = torch.tensor([], device='cpu', dtype=torch.float32)
        # self.rect_saved = empty_tensor, empty_tensor, empty_tensor, empty_tensor
        self.rect_saved = None

    def hand_detection(self, image):
        img256, img128, scale, pad = self.palm_detector.resize_pad(image) # 输入预处理：缩放和裁剪
        normalized_palm_detections = self.palm_detector.predict_on_image(img256) # Detection Model
        palm_detections = self.palm_detector.denormalize_detections(normalized_palm_detections, scale, pad) # 输出后处理：缩放还原
        return palm_detections
    
    def hand_regression(self, image, xc, yc, scale, theta):
        cropped_image, affine_trans, box_points = self.hand_regressor.extract_roi(image, xc, yc, theta, scale) # 输入预处理：变换和裁剪
        flags, handed, normalized_landmarks2 = self.hand_regressor(cropped_image.to(gpu)) # Regression Model
        landmarks = self.hand_regressor.denormalize_landmarks(normalized_landmarks2.cpu(), affine_trans) # 输出后处理：变换裁剪还原
        return landmarks, flags

    def detection_to_rectangle(self, detections):
        rect = self.palm_detector.detection2roi(detections.cpu())
        return rect
        
    def landmarks_to_rectangle(self, landmarks):
        """
        使用PyTorch实现的landmarks_to_rectangle
        完全支持GPU和批量处理
        """
        partial_landmark_indices = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]
        
        # 处理空输入
        if landmarks is None or landmarks.numel() == 0:
            device = torch.device('cpu')
            if isinstance(landmarks, torch.Tensor):
                device = landmarks.device
            
            # 返回四个空的tensor
            empty_tensor = torch.tensor([], device=device, dtype=torch.float32)
            return empty_tensor, empty_tensor, empty_tensor, empty_tensor
        
        # 获取设备信息
        device = landmarks.device
        dtype = landmarks.dtype
        
        # 处理批量数据
        if landmarks.dim() == 3:
            batch_size = landmarks.shape[0]
            landmarks = landmarks.view(-1, 21, 3)
        else:
            batch_size = 1
            landmarks = landmarks.unsqueeze(0)
        
        # 提取部分关键点
        partial_indices = torch.tensor(partial_landmark_indices, device=device)
        partial_landmarks = torch.index_select(landmarks, 1, partial_indices)
        
        # 提取坐标
        xs = partial_landmarks[:, :, 0] # * image_width
        ys = partial_landmarks[:, :, 1] # * image_height
        
        # 计算轴对齐边界框
        max_x, _ = torch.max(xs, dim=1)
        max_y, _ = torch.max(ys, dim=1)
        min_x, _ = torch.min(xs, dim=1)
        min_y, _ = torch.min(ys, dim=1)
        
        axis_aligned_center_x = (max_x + min_x) / 2.0
        axis_aligned_center_y = (max_y + min_y) / 2.0

        # 1. 计算旋转角度
        # 计算旋转角度（PyTorch版本）
        def compute_rotation(partial_landmarks):
            """
            PyTorch版本的计算手部旋转角度
            partial_landmarks: 部分关键点，形状为(batch_size, 12, 3)
            返回: rotation角度，形状为(batch_size,)
            """
            kTargetAngle = torch.tensor(math.pi * 0.5, device=partial_landmarks.device, dtype=partial_landmarks.dtype)
            
            # 获取手腕坐标（索引0）
            wrist = partial_landmarks[:, 0, :2]  # (batch_size, 2)
            x0 = wrist[:, 0]  # (batch_size,)
            y0 = wrist[:, 1]  # (batch_size,)
            
            # 获取食指、中指、无名指的PIP关节
            # 索引：手腕:0, 食指PIP:4, 中指PIP:6, 无名指PIP:8
            index_pip = partial_landmarks[:, 4, :2]  # (batch_size, 2)
            middle_pip = partial_landmarks[:, 6, :2]  # (batch_size, 2)
            ring_pip = partial_landmarks[:, 8, :2]   # (batch_size, 2)
            
            # 计算加权平均点（与C++代码一致）
            # x1 = (食指PIP.x + 无名指PIP.x) / 2
            # 然后 x1 = (x1 + 中指PIP.x) / 2
            x1 = (index_pip[:, 0] + ring_pip[:, 0]) / 2.0  # (batch_size,)
            y1 = (index_pip[:, 1] + ring_pip[:, 1]) / 2.0  # (batch_size,)
            
            x1 = (x1 + middle_pip[:, 0]) / 2.0  # (batch_size,)
            y1 = (y1 + middle_pip[:, 1]) / 2.0  # (batch_size,)
            
            # 计算向量角度
            # atan2(-(y1 - y0), x1 - x0)
            # 注意：y取负是因为图像坐标系y向下，数学坐标系y向上
            dx = x1 - x0
            dy = y1 - y0
            angle = torch.atan2(-dy, dx)  # (batch_size,)
            
            # 计算与目标角度的差值
            rotation = kTargetAngle - angle  # (batch_size,)
            
            # 标准化角度到[-π, π)范围内
            # 公式: angle - 2 * π * floor((angle - (-π)) / (2 * π))
            rotation = rotation - 2 * torch.pi * torch.floor((rotation - (-torch.pi)) / (2 * torch.pi))
            
            return rotation

        try:
            rotation = torch.tensor(compute_rotation(partial_landmarks), device=device)
        except Exception as e:
            print(f"Error computing rotation: {e}")
            rotation = torch.zeros(batch_size, device=device, dtype=dtype)
        
        # 以下为可选的完整旋转边界框计算（当前注释掉）
        if True:
            # 3. 计算旋转后的边界框
            reverse_angle = -rotation  # (batch_size,)
            cos_rev = torch.cos(reverse_angle)  # (batch_size,)
            sin_rev = torch.sin(reverse_angle)  # (batch_size,)
            cos_rot = torch.cos(rotation)  # (batch_size,)
            sin_rot = torch.sin(rotation)  # (batch_size,)
            
            # 将部分关键点平移到以中心为原点
            original_x = partial_landmarks[:, :, 0] - axis_aligned_center_x.unsqueeze(1)  # (batch_size, 12)
            original_y = partial_landmarks[:, :, 1] - axis_aligned_center_y.unsqueeze(1)  # (batch_size, 12)
            
            # 应用反向旋转矩阵（批量处理）
            projected_x = original_x * cos_rev.unsqueeze(1) - original_y * sin_rev.unsqueeze(1)  # (batch_size, 12)
            projected_y = original_x * sin_rev.unsqueeze(1) + original_y * cos_rev.unsqueeze(1)  # (batch_size, 12)
            
            # 找到投影后的极值
            proj_max_x, _ = torch.max(projected_x, dim=1)  # (batch_size,)
            proj_max_y, _ = torch.max(projected_y, dim=1)  # (batch_size,)
            proj_min_x, _ = torch.min(projected_x, dim=1)  # (batch_size,)
            proj_min_y, _ = torch.min(projected_y, dim=1)  # (batch_size,)
            
            # 计算旋转后坐标系的中心
            projected_center_x = (proj_max_x + proj_min_x) / 2.0  # (batch_size,)
            projected_center_y = (proj_max_y + proj_min_y) / 2.0  # (batch_size,)
            
            # 将中心旋转回原始方向
            center_x = (projected_center_x * cos_rot - projected_center_y * sin_rot + 
                    axis_aligned_center_x)  # (batch_size,)
            center_y = (projected_center_x * sin_rot + projected_center_y * cos_rot + 
                    axis_aligned_center_y)  # (batch_size,)
            
            # 计算宽度和高度
            width = proj_max_x - proj_min_x  # (batch_size,)
            height = proj_max_y - proj_min_y  # (batch_size,)
            
            # 计算质心
            xc = center_x  # (batch_size,)
            yc = center_y  # (batch_size,)
        else:
            # 简化版本：使用轴对齐边界框
            width = max_x - min_x  # (batch_size,)
            height = max_y - min_y  # (batch_size,)
            # 计算质心
            xc = axis_aligned_center_x  # (batch_size,)
            yc = axis_aligned_center_y  # (batch_size,)
        
        # 计算scale
        scale = torch.max(width, height) / 2.0  # (batch_size,)

        # 来自detection2roi方法的代码
        # yc += self.palm_detector.dy * scale # 这个位置偏移应该不必
        scale *= 4 # 4.5只是个尝试的比例系数

        # 旋转
        # theta = torch.zeros(batch_size, device=device, dtype=dtype)
        theta = rotation
        
        # 如果只有单个样本，去除batch维度
        if batch_size == 1:
            xc = xc.squeeze(0)
            yc = yc.squeeze(0)
            scale = scale.squeeze(0)
            theta = theta.squeeze(0)
        
        return xc, yc, scale, theta
    
    def loop(self, frame):
        if self.rect_saved is None or any(r.numel() == 0 for r in self.rect_saved) or (self.rect_saved[2].numel() > 0 and torch.all(self.rect_saved[2] < 64)):
            palm_detections = self.hand_detection(frame)
            rect_detected = self.detection_to_rectangle(palm_detections)
        else:
            rect_detected = self.rect_saved
        xc, yc, scale, theta = rect_detected

        landmarks, flags = self.hand_regression(frame, xc, yc, scale, theta)
        rect_predicted = self.landmarks_to_rectangle(landmarks)

        self.rect_saved = rect_predicted

        self.debug = {
            'palm': palm_detections,
            'hand': (landmarks, flags),
            'roi': rect_predicted
        }
        return landmarks

class Main:
    @staticmethod
    def draw_annotation(rendered_image, landmarks, flags):
        # https://github.com/metalwhale/hand_tracking/blob/b2a650d61b4ab917a2367a05b85765b81c0564f2/run.py
        #        8   12  16  20
        #        |   |   |   |
        #        7   11  15  19
        #    4   |   |   |   |
        #    |   6   10  14  18
        #    3   |   |   |   |
        #    |   5---9---13--17
        #    2    \         /
        #     \    \       /
        #      1    \     /
        #       \    \   /
        #        ------0-
        HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
            (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
        ]
        def draw_landmarks(img, points, connections=[], color=(0, 255, 0), size=2):
            points = points[:,:2]
            for point in points:
                x, y = point
                x, y = int(x), int(y)
                cv2.circle(img, (x, y), size, color, thickness=size)
            for connection in connections:
                x0, y0 = points[connection[0]]
                x1, y1 = points[connection[1]]
                x0, y0 = int(x0), int(y0)
                x1, y1 = int(x1), int(y1)
                cv2.line(img, (x0, y0), (x1, y1), (0,0,0), size)
        for i in range(len(flags)):
            landmark, flag = landmarks[i], flags[i]
            if flag>.5:
                draw_landmarks(rendered_image, landmark[:,:2], HAND_CONNECTIONS, size=2)

    @staticmethod
    def draw_detections(img, detections, with_keypoints=True):
        if isinstance(detections, torch.Tensor):
            detections = detections.cpu().numpy()

        if detections.ndim == 1:
            detections = np.expand_dims(detections, axis=0)

        n_keypoints = detections.shape[1] // 2 - 2

        for i in range(detections.shape[0]):
            ymin = detections[i, 0]
            xmin = detections[i, 1]
            ymax = detections[i, 2]
            xmax = detections[i, 3]
            
            start_point = (int(xmin), int(ymin))
            end_point = (int(xmax), int(ymax))
            img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 1) 

            if with_keypoints:
                for k in range(n_keypoints):
                    kp_x = int(detections[i, 4 + k*2    ])
                    kp_y = int(detections[i, 4 + k*2 + 1])
                    cv2.circle(img, (kp_x, kp_y), 2, (0, 0, 255), thickness=2)
        return img

    @staticmethod
    def draw_roi(img, xc, yc, theta, scale, front_color=(0,255,0)):
        # take points on unit square and transform them according to the roi
        points = torch.tensor([[-1, -1, 1, 1],
                            [-1, 1, -1, 1]], device=scale.device).view(1,2,4)
        points = points * scale.view(-1,1,1)/2
        theta = theta.view(-1, 1, 1)
        R = torch.cat((
            torch.cat((torch.cos(theta), -torch.sin(theta)), 2),
            torch.cat((torch.sin(theta), torch.cos(theta)), 2),
            ), 1)
        center = torch.cat((xc.view(-1,1,1), yc.view(-1,1,1)), 1)
        points = R @ points + center

        for i in range(points.shape[0]):
            (x1,x2,x3,x4), (y1,y2,y3,y4) = points[i]
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,0), 2)
            cv2.line(img, (int(x1), int(y1)), (int(x3), int(y3)), front_color, 2)
            cv2.line(img, (int(x2), int(y2)), (int(x4), int(y4)), (0,0,0), 2)
            cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0,0,0), 2)


if __name__ == "__main__":
    runner = BlazeRunner()
    
    WINDOW='test'
    cv2.namedWindow(WINDOW)
    if len(sys.argv) > 1:
        capture = cv2.VideoCapture(sys.argv[1])
        mirror_img = False
    else:
        capture = cv2.VideoCapture(0)
        mirror_img = True

    if capture.isOpened():
        hasFrame, frame = capture.read()
        frame_ct = 0
    else:
        hasFrame = False

    while hasFrame:
        frame_ct +=1

        if mirror_img:
            frame = np.ascontiguousarray(frame[:,::-1,::-1])
        else:
            frame = np.ascontiguousarray(frame[:,:,::-1])

        landmarks = runner.loop(frame)
        runnner_debug = runner.debug

        palm_detections = runnner_debug['palm']
        landmarks, flags = runnner_debug['hand']
        xc, yc, theta, scale = runnner_debug['roi']

        rendered_frame = frame.copy()
        Main.draw_detections(rendered_frame, palm_detections)
        Main.draw_annotation(rendered_frame, landmarks, flags)
        Main.draw_roi(rendered_frame, xc, yc, theta, scale)
        cv2.imshow(WINDOW, rendered_frame[:,:,::-1])
        # cv2.imwrite('sample/%04d.jpg'%frame_ct, frame[:,:,::-1])

        hasFrame, frame = capture.read()
        key = cv2.waitKey(1)
        if key == 27:
            break

    capture.release()
    cv2.destroyAllWindows()
