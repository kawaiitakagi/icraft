
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/buyi_device.h>
#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/cuda/device.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "postprocess_bytetrack.hpp"
#include "icraft_utils.hpp"
#include "yaml-cpp/yaml.h"
#include "det_post.hpp"
using namespace icraft::xrt;
using namespace icraft::xir;

int main(int argc, char* argv[])
{
	YAML::Node config = YAML::LoadFile(argv[1]);
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
	// 初始化netinfo
	NetInfo netinfo = NetInfo(network);
	// netinfo.ouput_allinfo();

	
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
	#ifdef _WIN32
		std::string videoRoot = dataset["video_dir"].as<std::string>();
		std::string videoList = dataset["video_list"].as<std::string>();
		// 统计视频数量
		auto video_namevector = toVector(videoList);
		int total_video_num = video_namevector.size();
	#elif __linux__
		//板子上无法从视频提取视频帧，因此axi模式下只能输入图片
		std::string imgRoot = dataset["img_dir"].as<std::string>();
		std::string imgList = dataset["img_list"].as<std::string>();
		// 统计图片数量
		auto img_namevector = toVector(imgList);
		int total_img_num = img_namevector.size();
	#endif
	std::string names_path = dataset["names"].as<std::string>();
	auto LABELS = toVector(names_path);
	std::string resRoot = dataset["res"].as<std::string>();
	checkDir(resRoot);


	// 模型自身相关参数配置
	auto param = config["param"];
	float conf = param["conf"].as<float>();
	float iou_thresh = param["iou_thresh"].as<float>();
	bool MULTILABEL = param["multilabel"].as<bool>();
	bool fpga_nms = param["fpga_nms"].as<bool>();
	int N_CLASS = param["number_of_class"].as<int>();
	int N_HEAD = param["number_of_head"].as<int>();
	std::vector<std::vector<std::vector<float>>> ANCHORS = 
		param["anchors"].as<std::vector<std::vector<std::vector<float>>>>();

	// 计算real_out_channels
	int NOA = 1; //Anchor为空，NOA=1
	std::vector<int> ori_out_channels = { 1, 4, N_CLASS };
	int parts = ori_out_channels.size();
	

	

	int index = 0;
	#ifdef _WIN32
		for (auto name : video_namevector) {
			progress(index, total_video_num);
			index++;
			//------------- PRE PROCESS -----------------------//
			std::string video_path = videoRoot + '/' + name;
			cv::VideoCapture cap(video_path); //读取视频数据
			cv::Mat one_frame;
			int fps = cap.get(cv::CAP_PROP_FPS);
			BYTETracker tracker(fps, 30);
			std::vector<std::array<float, 10>> res_export = std::vector<std::array<float, 10>>();
			auto total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
			for (int frame_id = 0; frame_id < total_frames; ++frame_id) {
				progress(frame_id, total_frames);
				cap >> one_frame;//获取一帧图片
				PicPre_bytetrack img(one_frame);
				img.Resize({ netinfo.i_cubic[0].h, netinfo.i_cubic[0].w }, PicPre_bytetrack::LONG_SIDE).rPad();

				//------------- ICRAFT RUN ------------------//
				Tensor img_tensor = CvMat2Tensor(img.dst_img, network);
				dmaInit(run_sim, netinfo.ImageMake_on, img_tensor, device);
				//std::cout << img_tensor.dtype()->shape << std::endl;

				std::vector<Tensor> outputs = session.forward({ img_tensor });
				//std::cout << outputs[0].dtype()->shape << std::endl;
				//std::cout << outputs[1].dtype()->shape << std::endl;
				//std::cout << outputs[2].dtype()->shape << std::endl;
				//std::cout << outputs[3].dtype()->shape << std::endl;
				//std::cout << outputs[4].dtype()->shape << std::endl;
				//std::cout << outputs[5].dtype()->shape << std::endl;
				//std::cout << outputs[6].dtype()->shape << std::endl;
				//std::cout << outputs[7].dtype()->shape << std::endl;
				//std::cout << outputs[8].dtype()->shape << std::endl;
				if (!run_sim) device.reset(1);
				// 计时
				#ifdef __linux__
				device.reset(1);
				calctime_detail(session);
				#endif

				//------------- POST PROCESS ------------------//
				if (netinfo.DetPost_on) {
					std::pair<int, std::vector<int>> anchor_length_real_out_channels =
						_getAnchorLength(ori_out_channels, netinfo.detpost_bit, NOA);
					std::vector<int> real_out_channels = anchor_length_real_out_channels.second;
					// normratio分组
					std::vector<float> normalratio = netinfo.o_scale;
					std::vector<std::vector<float>> norm = set_norm_by_head(N_HEAD, parts, normalratio);

					post_detpost_hard(real_out_channels, outputs, img, netinfo, norm,
						conf, iou_thresh, MULTILABEL, fpga_nms, N_CLASS, ANCHORS, LABELS,
						show, save, resRoot, name, device, run_sim, tracker, res_export, frame_id);
				}
				else {
					post_detpost_soft(outputs, img, LABELS, ANCHORS, netinfo,
						N_CLASS, conf, iou_thresh, MULTILABEL, tracker,  res_export, frame_id, show, save, resRoot, name);
				}
			}
		}

	#elif __linux__
		BYTETracker tracker(30, 30);
		std::vector<std::array<float, 10>> res_export = std::vector<std::array<float, 10>>();
		for (int frame_id = 0; frame_id < total_img_num; ++frame_id) {
			progress(frame_id, total_img_num);
			std::string name = img_namevector[frame_id];
			std::string img_path = imgRoot + '/' + name;

			PicPre_bytetrack img(img_path, cv::IMREAD_COLOR);
			img.Resize({ netinfo.i_cubic[0].h, netinfo.i_cubic[0].w }, PicPre_bytetrack::LONG_SIDE).rPad();
			//------------- ICRAFT RUN ------------------//
			Tensor img_tensor = CvMat2Tensor(img.dst_img, network);
			dmaInit(run_sim, netinfo.ImageMake_on, img_tensor, device);
			//std::cout << img_tensor.dtype()->shape << std::endl;

			std::vector<Tensor> outputs = session.forward({ img_tensor });
			//std::cout << outputs[0].dtype()->shape << std::endl;
			//std::cout << outputs[1].dtype()->shape << std::endl;
			//std::cout << outputs[2].dtype()->shape << std::endl;
			if (!run_sim) device.reset(1);
			device.reset(1);
			calctime_detail(session);

			std::pair<int, std::vector<int>> anchor_length_real_out_channels =
				_getAnchorLength(ori_out_channels, netinfo.detpost_bit, NOA);
			std::vector<int> real_out_channels = anchor_length_real_out_channels.second;
			// normratio分组
			std::vector<float> normalratio = netinfo.o_scale;
			std::vector<std::vector<float>> norm = set_norm_by_head(N_HEAD, parts, normalratio);

			//------------- POST PROCESS ------------------//
			post_detpost_hard(real_out_channels, outputs, img, netinfo, norm,
					conf, iou_thresh, MULTILABEL, fpga_nms, N_CLASS, ANCHORS, LABELS,
					show, save, resRoot, name, device, run_sim, tracker, res_export, frame_id);
		}

	#endif
	//关闭设备
	Device::Close(device);
	return 0;
}



