#!/usr/bin/env python

import logging
import argparse
import torch
import os
import onnx
import onnxruntime
import cv2
import numpy as np
import yacs.config
from gaze_estimation import *
from gaze_estimation.gaze_estimator.common import (Face, FacePartsName,
                                                   Visualizer)
from demo import Demo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#导出模型的路径：
MODEL_PATH = R"../2_compile/fmodel/MPIIGaze_1x1x36x60_1x2.onnx"  #onnx model path
#保存量化集ftmp
QTSET_SAVE = False
ID=0 

def load_configs() -> yacs.config.CfgNode:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default="config/demo_mpiigaze_resnet.yaml", type=str)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--qtset_path', default="./qtset/MPIIGaze/", type=str)
    
    args = parser.parse_args()
    if QTSET_SAVE:
        if not os.path.exists(args.qtset_path):
            os.makedirs(args.qtset_path)
    config = get_default_config()
    if args.config is not None:
        config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.train_dataloader.pin_memory = False
        config.train.val_dataloader.pin_memory = False
        config.test.dataloader.pin_memory = False
    config.freeze()
    return config

def new_run(self) -> None:
    while True:
        if self.config.demo.display_on_screen:
            self._wait_key()
            if self.stop:
                break
        input_path = self.config.demo.video_path
        if input_path.split('.')[-1]=="jpg" or input_path.split('.')[-1]=="png":
            frame = cv2.imread(input_path)
        else:
            ok, frame = self.cap.read()
            if not ok:
                break

        undistorted = cv2.undistort(
            frame, self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients)

        self.visualizer.set_image(frame.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)
        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            self._draw_face_bbox(face)
            self._draw_head_pose(face)
            self._draw_landmarks(face)
            self._draw_face_template_model(face)
            self._draw_gaze_vector(face)
            self._display_normalized_image(face)

        if self.config.demo.use_camera:
            self.visualizer.image = self.visualizer.image[:, ::-1]
        if self.writer:
            self.writer.write(self.visualizer.image)
        if self.config.demo.display_on_screen:
            cv2.imshow('frame', self.visualizer.image)
            if input_path.split('.')[-1]=="jpg" or input_path.split('.')[-1]=="png":
                cv2.waitKey(0)  
    self.cap.release()
    if self.writer:
        self.writer.release()
Demo.run = new_run

def new_run_mpiigaze_model(self, face: Face) -> None:
    ## 修改前向为：左右眼分别跑两次前向
    predictions = []
    # 加载trace出的模型
    ort_session = onnxruntime.InferenceSession(MODEL_PATH)
    ort_inputs1 = ort_session.get_inputs()[0].name
    ort_inputs2 = ort_session.get_inputs()[1].name
    global ID
    ID += 1
    for key in self.EYE_KEYS:
        eye = getattr(face, key.name.lower())
        image = eye.normalized_image#(36,60)
        normalized_head_pose = eye.normalized_head_rot2d #(2,)欧拉角
        if key == FacePartsName.REYE:
            image = image[:, ::-1] #bgr->rgb
            normalized_head_pose *= np.array([1, -1]) #方向矫正
        image = self._transform(image).cpu().unsqueeze(0) #内部/255
        head_pose = torch.from_numpy(np.array(normalized_head_pose).astype(np.float32)).cpu().unsqueeze(0)
        if QTSET_SAVE:
            #导出ftmp，作为量化集
            name = str(key.name[0])
            image.numpy().astype(np.float32).tofile('./qtset/MPIIGaze/{}_eye_1x1x36x60_{}.ftmp'.format(name,ID))
            head_pose.numpy().astype(np.float32).tofile('./qtset/MPIIGaze/{}_head_poses_1x2_{}.ftmp'.format(name,ID))
        with torch.no_grad():
            image = image.to("cpu").numpy() # 单张灰度眼图（1,1,36,60）
            head_pose = head_pose.to("cpu").numpy() # head_pose:1x2
            prediction = ort_session.run(None, {ort_inputs1:image,ort_inputs2:head_pose}) #改成左右眼分别前向
            predictions.append(torch.from_numpy(prediction[0][0]))
    predictions = torch.stack(predictions)
    # predictions = predictions.cpu().numpy()
    # 后处理
    for i, key in enumerate(self.EYE_KEYS):
        eye = getattr(face, key.name.lower())
        eye.normalized_gaze_angles = predictions[i]
        if key == FacePartsName.REYE:
            eye.normalized_gaze_angles *= np.array([1, -1])
        eye.angle_to_vector()#角度转矢量
        eye.denormalize_gaze_vector()#从标准化坐标系转回实际坐标系，向量矩阵乘法

GazeEstimator._run_mpiigaze_model = new_run_mpiigaze_model

def main():
    config = load_configs()
    demo = Demo(config)
    demo.run()


if __name__ == '__main__':
    main()
