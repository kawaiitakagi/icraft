import sys
sys.path.append(R"../../../Deps/modelzoo")
from icraft.xir import *
from icraft.xrt import *
from icraft.host_backend import *
from icraft.buyibackend import *
import numpy as np
import cv2
import os
import yaml
from tqdm import tqdm
from datetime import timedelta
from process_mpiigaze import *
sys.path.insert(0,R"../../../../0_MPIIGaze")
from gaze_estimation import GazeEstimationMethod, GazeEstimator
from gaze_estimation.gaze_estimator.common import (Face, FacePartsName,
                                                   Visualizer)
from pyrtutils.icraft_utils import *
from pyrtutils.Netinfo import Netinfo
from pyrtutils.utils import *
from pyrtutils.et_device import *
from pyrtutils.calctime_utils import *


def new_run_mpiigaze_model(self, face: Face) -> None:
    predictions = []
    for key in self.EYE_KEYS:
        eye = getattr(face, key.name.lower())
        image = eye.normalized_image
        normalized_head_pose = eye.normalized_head_rot2d
        if key == FacePartsName.REYE:
            image = image[:, ::-1]
            normalized_head_pose *= np.array([1, -1])
        image = self._transform(image).cpu().unsqueeze(0)#前处理不要再次进行归一化
        eye_img = image.numpy().reshape(netinfo.i_shape[0]).astype(np.float32)
        head_pose = normalized_head_pose.reshape(netinfo.i_shape[1]).astype(np.float32)
        # 构造Icraft tensor
        inputs=[]
        inputs.append(Tensor(eye_img, Layout("***C")))
        inputs.append(Tensor(head_pose, Layout("*C")))
        # dma init(if use imk)
        dmaInit(run_sim, netinfo.ImageMake_on, netinfo.i_shape[0][1:], inputs[0], device)
        # 前向
        generated_output = session.forward(inputs)
        for tensor in generated_output:
            timeout = timedelta(milliseconds=1000)
            tensor.waitForReady(timeout)

        if not run_sim:
            device.reset(1)
            calctime_detail(session, network, name="./"+network.name+"_time.xlsx")

        # 后处理
        prediction = np.array(generated_output[0])
        predictions.append(prediction)


    for i, key in enumerate(self.EYE_KEYS):
        eye = getattr(face, key.name.lower())
        eye.normalized_gaze_angles = predictions[i][0]
        if key == FacePartsName.REYE:
            eye.normalized_gaze_angles *= np.array([1, -1])
        eye.angle_to_vector()
        eye.denormalize_gaze_vector()
GazeEstimator._run_mpiigaze_model = new_run_mpiigaze_model

if __name__ == "__main__":
    # 构造MPIIGaze相关配置
    config = load_config()
    gaze_estimator = GazeEstimator(config)
    visualizer = Visualizer(gaze_estimator.camera)
    # 获取yaml
    Yaml_Path = "../cfg/MPIIGaze.yaml"
    if len(sys.argv) < 2:
        print("Info:未传入yaml参数,读入默认yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) == 2:
        Yaml_Path = sys.argv[1]
        print("info:传入yaml文件:‌{}进行相关配置.".format(Yaml_Path), VERBOSE, 0)        
    if len(sys.argv) > 2:
        print("info:传入参数数量错误,请检查运行命令!", VERBOSE, 0)        
        sys.exit(1)
    # 从yaml里读入配置
    cfg = yaml.load(open(Yaml_Path, "r", encoding="utf-8"), Loader=yaml.FullLoader)   
    folderPath = cfg["imodel"]["net_dir"]
    stage = cfg["imodel"]["stage"]
    run_sim = cfg["imodel"]["sim"]
    JSON_PATH, RAW_PATH = getJrPath(folderPath,stage,run_sim)


    load_mmu = cfg["imodel"]["mmu"]
    load_speedmode = cfg["imodel"]["speedmode"]
    load_compressFtmp = cfg["imodel"]["compressFtmp"]
    ip = str(cfg["imodel"]["ip"])
    save = cfg["imodel"]["save"]
    show = cfg["imodel"]["show"]

    imgRoot = os.path.abspath(cfg["dataset"]["dir"])
    imgList = os.path.abspath(cfg["dataset"]["list"])
    resRoot = cfg["dataset"]["res"]

    # 加载network
    network = loadNetwork(JSON_PATH, RAW_PATH)
  	# 初始化netinfo
    netinfo = Netinfo(network)
    # 选择对网络进行切分
    network_view = network.view(0)
    # 打开device
    device = openDevice(run_sim, ip, netinfo.mmu or load_mmu)
    # 初始化session
    session = initSession(run_sim, network_view, device, netinfo.mmu or load_mmu, load_speedmode, load_compressFtmp)
    

	#开启计时功能
    session.enableTimeProfile(True)

	#session执行前必须进行apply部署操作
    session.apply()


    print(f'loading image from {imgRoot}')
    file_list = os.listdir(imgRoot)

    for id in tqdm(range(len(file_list))):
        img_path = imgRoot +"/"+ file_list[id] # Initial frame
        frame = cv2.imread(img_path)
        undistorted = cv2.undistort(
                frame, gaze_estimator.camera.camera_matrix,
                gaze_estimator.camera.dist_coefficients)
        visualizer.set_image(frame.copy())
        faces = gaze_estimator.detect_faces(undistorted)#dlib人脸关键点检测
        
        for i in range(len(faces)):
            face = faces[i]

            # 前向
            gaze_estimator.estimate_gaze(undistorted, face)

            # 可视化
            visualizer.draw_bbox(face.bbox)
            length = config.demo.head_pose_axis_length
            visualizer.draw_model_axes(face, length, lw=2)
            euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
            pitch, yaw, roll = face.change_coordinate_system(euler_angles)
            print(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, ' f'roll: {roll:.2f}, distance: {face.distance:.2f}')
            # visualizer.draw_points(face.landmarks,
            #                         color=(0, 255, 255),
            #                         size=1)
            # visualizer.draw_3d_points(face.model3d,
            #                            color=(255, 0, 525),
            #                            size=1)
            length = config.demo.gaze_visualization_length
            if config.mode == GazeEstimationMethod.MPIIGaze.name:
                for key in [FacePartsName.REYE, FacePartsName.LEYE]:
                    eye = getattr(face, key.name.lower())
                    visualizer.draw_3d_line(
                        eye.center, eye.center + length * eye.gaze_vector)
                    pitch, yaw = np.rad2deg(eye.vector_to_angle(eye.gaze_vector))
                    print(f'[{key.name.lower()}] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
            else:
                raise ValueError
            if config.mode == GazeEstimationMethod.MPIIGaze.name:
                reye = face.reye.normalized_image
                leye = face.leye.normalized_image
                normalized = np.hstack([reye, leye])
            else:
                raise ValueError
            if config.demo.use_camera:
                visualizer.image = visualizer.image[:, ::-1]
            if show:
                cv2.imshow('frame', visualizer.image)
                cv2.waitKey(0)
            if save:
                file_name = file_list[id].split(".")[0]
                save_path = resRoot +"/"+ file_name +"_res_"+str(i)+".jpg"
                cv2.imwrite(save_path, visualizer.image)
                print("save result in ", save_path)
            
    # 关闭设备
    Device.Close(device)
    
