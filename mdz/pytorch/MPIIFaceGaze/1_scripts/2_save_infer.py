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
from gaze_estimation.utils import load_config
from demo import Demo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#导出模型的路径：
MODEL_PATH = R"../2_compile/fmodel/MPIIFaceGaze_1x3x224x224.onnx"  #onnx model path
#保存量化集ftmp
QTSET_SAVE = False
ID=0 

def load_configs() -> yacs.config.CfgNode:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default="config/demo_mpiifacegaze_resnet_simple_14.yaml", type=str)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--qtset_path', default="./qtset/MPIIFaceGaze/", type=str)
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

def new_run_mpiifacegaze_model(self, face: Face) -> None:
    global ID
    ID += 1
    if QTSET_SAVE:
        #导出ftmp，作为量化集
        face.normalized_image.reshape(1,224,224,3).astype(np.float32).tofile('./qtset/MPIIFaceGaze/img_1x224x224x3_{}.ftmp'.format(ID))
    image = self._transform(face.normalized_image).unsqueeze(0) #归一化[1,3,224,224]
    # 加载trace出的模型
    ort_session = onnxruntime.InferenceSession(MODEL_PATH)
    ort_inputs1 = ort_session.get_inputs()[0].name
    # global ID
    # ID += 1
    # if QTSET_SAVE:
    #     #导出ftmp，作为量化集
    #     image.numpy().astype(np.float32).tofile('./qtset/MPIIFaceGaze/face_1x3x224x224_{}.ftmp'.format(ID))
    with torch.no_grad():
        image = image.to("cpu").numpy()
        ort_outs = ort_session.run(None, {ort_inputs1:image})
        prediction = ort_outs[0]

    face.normalized_gaze_angles = prediction[0]
    face.angle_to_vector()
    face.denormalize_gaze_vector()

GazeEstimator._run_mpiifacegaze_model = new_run_mpiifacegaze_model


def main():
    config = load_configs()
    demo = Demo(config)
    demo.run()


if __name__ == '__main__':
    main()
