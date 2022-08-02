#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : b站@在下啊水
# detection: 基于yolov5+flask 网页端实现
import imp
import os
import cv2
from base_camera import BaseCamera
from models.experimental import attempt_load
import torch
import torch.nn as nn
import torchvision
import numpy as np
import argparse
import sys 
from models.common import DetectMultiBackend
sys.path.append('D:\Project\positionDetect\yolov5-master\yolov5-master\\') 
from utils.dataloaders import *
from utils.plots import *
from utils.torch_utils import *
from utils.general import (cv2,non_max_suppression,scale_coords)


class Camera(BaseCamera):
    video_source = '"D:\GoogleDowmload\pytorch\\2.mp4"'
    res='系统持续检测中。。。'
    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        out='inference/output'
        weights = 'D:\Project\positionDetect\yolov5-master\yolov5-master\\runs/train/exp3/weights/best.pt'
        imgsz = 640
        source = "0"
        data = 'D:\Project\positionDetect\yolov5-master\yolov5-master\data\\fall.yaml'
        device = select_device()
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

        # Load model
        model = attempt_load(weights, device=device)  # load FP32 model
        model.to(device).eval()

        # Half precision
        half = False and device.type != 'cpu'
        print('half = ' + str(half))

        if half:
            model.half()

        # Set Dataloader
        # vid_path, vid_writer = None, None
        modelss = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=half)
        stride, names, pt = modelss.stride, modelss.names, modelss.pt
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        # dataset = LoadStreams(source, img_size=imgsz)
        names = model.names if hasattr(model, 'names') else model.modules.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
        # 
        sum = 0
        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap,s in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_sync()
            pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, 0.8, 0.5,
                                       None, False,max_det=10)
            t2 = time_sync()
            
            for i, det in enumerate(pred):  # detections per image
                # p, s, im0 = path, '', im0s
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '   
                # save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    sum+=1
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # for c in det[:, -1].unique():  #probably error with torch 1.5
                    for c in det[:, -1].detach().unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %s, ' % (n, names[int(c)])  # add to string
                        if(names[int(c)]=='fall' and sum>5):
                            print('fall')
                            Camera.res = '监测到有人摔倒，请及时处理'
                    for *xyxy, conf, cls in det:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                # print('%sDone. (%.3fs)' % (s, t2 - t1))
                else:
                    sum=0
                    Camera.res = '系统持续检测中。。。'
            yield cv2.imencode('.jpg', im0)[1].tobytes()