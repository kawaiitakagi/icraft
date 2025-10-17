#!/usr/bin/env python
import datetime
import logging
import pathlib
from typing import Optional
import argparse
import cv2
import torch
import numpy as np
import yacs.config
from gaze_estimation import get_default_config
from gaze_estimation.utils import load_config
from demo import Demo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_configs() -> yacs.config.CfgNode:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default="config/demo_mpiigaze_resnet.yaml", type=str)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
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

def main():
    config = load_configs()
    demo = Demo(config)
    demo.run()


if __name__ == '__main__':
    main()
