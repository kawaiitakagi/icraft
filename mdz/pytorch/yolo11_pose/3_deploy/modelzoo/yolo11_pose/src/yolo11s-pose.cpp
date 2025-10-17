#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/buyi_device.h>
#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/cuda/device.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "postprocess_yolo11_pose.hpp"
#include "yolo11_pose_utils.hpp"
#include "icraft_utils.hpp"
#include "yaml-cpp/yaml.h"
using namespace icraft::xrt;
using namespace icraft::xir;


int main(int argc, char* argv[])
{
	YAML::Node config = YAML::LoadFile(argv[1]);
	// YAML::Node config = YAML::LoadFile("../cfg/yolov8s-pose.yaml");
	// icraft模型部署相关参数配置
	auto imodel = config["imodel"];
	// 仿真上板的jrpath配置
	std::string folderPath = imodel["dir"].as<std::string>();  
	bool run_sim = imodel["sim"].as<bool>();
	bool cudamode = imodel["cudamode"].as<bool>();
	std::string targetFileName;
	std::string JSON_PATH = getJrPath(run_sim,folderPath, imodel["stage"].as<std::string>());
	std::regex rgx3(".json");
	std::string RAW_PATH = std::regex_replace(JSON_PATH, rgx3, ".raw");
	// URL配置
	std::string ip = imodel["ip"].as<std::string>();
	// 可视化配置
	bool show = imodel["show"].as<bool>();
	bool save = imodel["save"].as<bool>();

	// 加载network
	Network network = loadNetwork(JSON_PATH, RAW_PATH);
	//初始化netinfo
	NetInfo netinfo = NetInfo(network);
	//netinfo.ouput_allinfo();
	// 选择对网络进行切分
	auto network_view = network.view(netinfo.inp_shape_opid + 1);
	// 打开device
	Device device = openDevice(run_sim, ip, netinfo.mmu || imodel["mmu"].as<bool>(), cudamode);
	// 初始化session
	Session session = initSession(run_sim, network_view, device, netinfo.mmu || imodel["mmu"].as<bool>(), imodel["speedmode"].as<bool>(), imodel["compressFtmp"].as<bool>());
	// 开启计时功能
	session.enableTimeProfile(true);
	// session执行前必须进行apply部署操作
	session.apply();

	// 数据集相关参数配置
	auto dataset = config["dataset"];
	std::string imgRoot = dataset["dir"].as<std::string>();
	std::string imgList = dataset["list"].as<std::string>();
	std::string names_path = dataset["names"].as<std::string>();
	std::string resRoot = dataset["res"].as<std::string>();
	checkDir(resRoot);
	auto LABELS = toVector(names_path);
	// 模型自身相关参数配置
	auto param = config["param"];
	float conf = param["conf"].as<float>();
	float iou_thresh = param["iou_thresh"].as<float>();
	bool fpga_nms = param["fpga_nms"].as<bool>();
	int NOH = param["number_of_head"].as<int>();
	int N_CLASS = param["number_of_class"].as<int>();
	int NOP = param["number_of_keypoint"].as<int>();

	std::vector<std::vector<std::vector<float>>> ANCHORS = 
		param["anchors"].as<std::vector<std::vector<std::vector<float>>>>();
	int NOA = 1;
	if (ANCHORS.size() != 0) {
		NOA = ANCHORS[0].size();
	}

	int bbox_info_channel = 64;
	std::vector<int> ori_out_channles = { N_CLASS,bbox_info_channel,NOP * 3 };
	int parts = ori_out_channles.size();

	// 统计图片数量
	int index = 0;
	auto namevector = toVector(imgList);
	int totalnum = namevector.size();
	for (auto name : namevector) {
		progress(index, totalnum);
		index++;
		std::string img_path = imgRoot + '/' + name;
		// 前处理
		PicPre img(img_path, cv::IMREAD_COLOR);

		img.Resize({ netinfo.i_cubic[0].h, netinfo.i_cubic[0].w}, PicPre::LONG_SIDE).rPad();
		Tensor img_tensor = CvMat2Tensor(img.dst_img, network);

		dmaInit(run_sim, netinfo.ImageMake_on,img_tensor, device);
		
		std::vector<Tensor> outputs = session.forward({ img_tensor });
		std::cout << outputs[0].dtype()->shape << std::endl;
		std::cout << outputs[1].dtype()->shape << std::endl;
		std::cout << outputs[2].dtype()->shape << std::endl;
		if(!run_sim) device.reset(1);
		//calctime_detail(session);
		// 计时
		 #ifdef __linux__
		 device.reset(1);
		 calctime_detail(session);
		 #endif
		 auto post_start = std::chrono::system_clock::now();
		 if (netinfo.DetPost_on) {
			std::pair<int, std::vector<int>> anchor_length_real_out_channles =
				_getAnchorLength(ori_out_channles, netinfo.detpost_bit, NOA);
			int anchor_length = anchor_length_real_out_channles.first;
			std::vector<int> real_out_channles = anchor_length_real_out_channles.second;

		 	post_detpost_hard(outputs, img,
		 		netinfo, conf, iou_thresh, fpga_nms, N_CLASS, NOP, ANCHORS,LABELS,
		 		show,save,resRoot,name,device,run_sim, real_out_channles, bbox_info_channel);
		 }
		 else {
		  	post_detpost_soft(outputs, img,
				netinfo, conf, iou_thresh, fpga_nms, N_CLASS, NOP, ANCHORS, LABELS,
				show, save, resRoot, name, device, run_sim, bbox_info_channel);
		 }
		auto post_end = std::chrono::system_clock::now();
		auto post_time = std::chrono::duration_cast<std::chrono::microseconds>(post_end - post_start);
		std::cout << "post_time = " << double(post_time.count() / 1000.f) << '\n';

	}
	//关闭设备
	Device::Close(device);
    return 0;
}
