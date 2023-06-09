# main funtion to be called by MATLAB
# Zhuohe Liu, harry.liu@alumni.rice.edu
# St-Pierre Lab, May 2023

import torch

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import (check_img_size, non_max_suppression, cv2)
from utils.segment.general import process_mask

from utils.augmentations import letterbox
import numpy as np

class YOLOV7Segmenter(object):

    def __init__(self,
                 weights = '',
                 device = 'cpu'):
        '''
        data (str): dataset.yaml path. 
        weights (str): model weight *.pt file path. 
        device (str): cuda device, i.e. 0 or 0,1,2,3 or cpu. 
        '''

        # Load model
        self.device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
        self.model = model
        self.stride, names, pt = model.stride, model.names, model.pt
        
        # Run inference
        bs = 1  # batch_size
        # model.warmup(imgsz=(1 if pt else bs, 3, *self.imgsz))  # warmup
        

    def segment(self, I, 
                conf_thres = 0.3, 
                iou_thres = 0.45, 
                max_det = 1000):
        '''
        I (matarray): input matrix.
        conf_thres (float): confidence threshold. 
        iou_thres (float): NMS IOU threshold. 
        max_det (float): maximum detections per image
        '''

        # path, im, im0s, vid_cap, s = next(iter(dataset)) # assume only one item
        # im0 = cv2.imread(source) # im0 should be (x, y, 3)
        I = np.array(I)
        imgsz = I.shape[0:2]
        imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        im = letterbox(I, imgsz, stride=self.stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        pred, out = self.model(im, augment=False, visualize=False)
        proto = out[1]
        imshape = im.shape[2:]
        del im  # release memory

        classes = None # filter by class: --class 0, or --class 0 2 3 
        agnostic_nms=False,  # class-agnostic NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        det = pred[0]
        i = 0
        # p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        # if len(det):
        masks = process_mask(proto[i], det[:, 6:], det[:, :4], imshape, upsample=True)  # HWC
        
        maskout = masks.cpu().numpy() # convert to numpy
        del masks

        # compress mask
        # obtain overlapping region
        maskOverlap = np.sum(maskout, 0)
        ind = np.arange(1, maskout.shape[0] + 1, 1)
        maskout = np.sum(maskout * ind[:, None, None], 0)
        maskout[maskOverlap > 1] = 0

        return maskout

s = YOLOV7Segmenter(weights, device)