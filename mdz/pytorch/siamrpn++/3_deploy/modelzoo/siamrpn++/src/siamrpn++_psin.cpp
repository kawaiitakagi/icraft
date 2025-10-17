#include <iostream>
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/buyi_device.h>
#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <random>
#include "postprocess_siamrpn++.hpp"
#include "icraft_utils.hpp"
#include "yaml-cpp/yaml.h"


using namespace icraft::xrt;
using namespace icraft::xir;
namespace fs = std::filesystem;


int main(int argc, char* argv[]) {
	try {
		YAML::Node config = YAML::LoadFile(argv[1]);
		// icraft模型部署相关参数配置
		auto imodel = config["imodel"];
		// 仿真上板的jrpath配置
		bool run_sim = imodel["sim"].as<bool>();
		std::string folderPath_net1 = imodel["net1_dir"].as<std::string>();
		std::string JSON_PATH_net1 = getJrPath(run_sim, folderPath_net1, imodel["stage"].as<std::string>());
		std::regex rgx3(".json");
		std::string RAW_PATH_net1 = std::regex_replace(JSON_PATH_net1, rgx3, ".raw");
		std::string folderPath_net2 = imodel["net2_dir"].as<std::string>();
		std::string JSON_PATH_net2 = getJrPath(run_sim, folderPath_net2, imodel["stage"].as<std::string>());
		std::string RAW_PATH_net2 = std::regex_replace(JSON_PATH_net2, rgx3, ".raw");
		std::string targetFileName;
		// URL配置
		std::string ip = imodel["ip"].as<std::string>();
		// 可视化配置
		bool show = imodel["show"].as<bool>();
		bool save = imodel["save"].as<bool>();
		// 加载network
		Network network_1 = loadNetwork(JSON_PATH_net1, RAW_PATH_net1);
		Network network_2 = loadNetwork(JSON_PATH_net2, RAW_PATH_net2);
		//初始化netinfo
		NetInfo netinfo_1 = NetInfo(network_1);
		NetInfo netinfo_2 = NetInfo(network_2);
		// 打开device
		Device device = openDevice(run_sim, ip, netinfo_1.mmu || imodel["mmu"].as<bool>());
		// 初始化session
		Session session_1 = initSession(run_sim, network_1, device, netinfo_1.mmu || imodel["mmu"].as<bool>(), imodel["speedmode"].as<bool>(), imodel["compressFtmp"].as<bool>());
		Session session_2 = initSession(run_sim, network_2, device, netinfo_2.mmu || imodel["mmu"].as<bool>(), imodel["speedmode"].as<bool>(), imodel["compressFtmp"].as<bool>());
		auto buyi_backend_1 = session_1->backends[0].cast<BuyiBackend>();
		auto buyi_backend_2 = session_2->backends[0].cast<BuyiBackend>();
		// 开启计时功能
		session_1.enableTimeProfile(true);
		session_2.enableTimeProfile(true);
		// session执行前必须进行apply部署操作
		session_2.apply();
		session_1.apply();

		//siamrpn++模型参数配置
		float context_amount = 0.5;
		float z_size = 127;
		float s_size = 255;
		// 生成anchor
		cv::Mat anchor_ = cv::Mat(3125, 4, CV_32F);
		std::ifstream anchorFile("../io/anchors.ftmp", std::ios::binary);
		anchorFile.read(reinterpret_cast<char*>(anchor_.data), 4 * 3125 * sizeof(float));
		anchorFile.close();
		cv::Mat anchor = anchor_.t();

		// 创建汉宁窗
		float window_influence = 0.6;
		cv::Mat window = cv::Mat(625 * 5, 1, CV_32F);
		std::ifstream windowFile("../io/window.ftmp", std::ios::binary);
		windowFile.read(reinterpret_cast<char*>(window.data), 3125 * sizeof(float));
		windowFile.close();
		/*float window_influence = 0.21;
		cv::Mat hanning = CreatHannWindow(17, 17);
		cv::Mat window = cv::Mat(289, 1, CV_32F, hanning.data);*/

		// 数据集相关参数配置
		auto dataset = config["dataset"];
		std::string filesRoot = dataset["dir"].as<std::string>();
		std::string filesList = dataset["list"].as<std::string>();
		std::string resRoot = dataset["res"].as<std::string>();
		std::string txt_root = resRoot + "/res_txt/"; //txt文件保存路径
		std::string img_root = resRoot + "/res_img/"; //img文件保存路径
		checkDir(resRoot);
		if (save) {
			checkDir(txt_root);
			checkDir(img_root);
		}

		// 统计数据集文件数量
		int index = 0;
		auto file_namevector = toVector(filesList);
		int total_file_num = file_namevector.size();

		//逐文件进行单目标追踪
		for (auto filename : file_namevector){
			progress(index, total_file_num);
			index++;
			std::cout << filename << std::endl;
			std::string imgpath = filesRoot + "/" + filename+ "/";
			std::string imgList = imgpath + filename + ".txt";
			auto img_namevector = toVector(imgList);
			int total_img_num = img_namevector.size();

			//从GT中获取target框的信息
			std::string gt = imgpath +  "groundtruth.txt";
			std::cout << gt << std::endl;

			std::vector<float> init_rect;//从gt读取target初始框
			std::ifstream file(gt);
			if (!file.is_open()) {
				// 处理文件打开失败的情况
				std::cout << gt + " open failed!" << std::endl;
				return 0;
			}
			std::string line;
			if (std::getline(file, line)) {
				std::stringstream ss(line);
				std::string token;
				while (std::getline(ss, token, ',')) {
					init_rect.push_back(std::stof(token));
				}
				// 输出目标初始位置
				for (float num : init_rect) {
					std::cout << num << " ";
				}
			}
			file.close();

			//std::vector<float> init_rect = { 160, 100, 110, 53 };//给定初始框；若不给定初始值，则从gt读取
			// 从GT获取初始帧目标的target_pos&size
			std::vector<float> target_pos = { (init_rect[0] + (init_rect[2] - 1) / 2),(init_rect[1] + (init_rect[3] - 1) / 2) }; //中心点位置
			std::vector<float> target_sz = { init_rect[2], init_rect[3] };
		
			// 获取输入图片尺寸
			cv::Mat frame = cv::imread(imgpath + "/img/" + img_namevector[0]);
			float input_h = frame.rows;
			float input_w = frame.cols;
			std::vector<float> im_sz = { input_h, input_w, 3 };

			// 按数据集中序列名称来命名结果保存的路径
			std::string img_path = img_root + filename + "/";
			std::string txt_file = txt_root + filename + ".txt";
			std::fstream outFile(txt_file, std::ios::out | std::ios::trunc);
			if (save) {
				checkDir(img_path);
				std::cout << "save txt in:" << txt_file << std::endl;
				//将第一帧结果保存至txt
				outFile << init_rect[0] << " " << init_rect[1] << " " << init_rect[2] << " " << init_rect[3] << std::endl;
			}

			//序列文件的全局变量
			cv::Mat im_patch;
			cv::Mat im_patch_template;
			icraft::xrt::Tensor data_ptr_1;
			icraft::xrt::Tensor data_ptr_2;
			icraft::xrt::Tensor data_ptr_3;
			icraft::xrt::Tensor data_ptr_4;
			icraft::xrt::Tensor data_ptr_5;
			icraft::xrt::Tensor data_ptr_6;
			float scale = 1.0;
			//std::vector<std::vector<float>> M_inversed = {    //仿射变换逆矩阵
			//	{1.0f, 0.0f, 0.0f},
			//	{0.0f, 1.0f, 0.0f}
			//};

			//逐帧进行单目标追踪
			for (int frame_id = 0; frame_id < total_img_num; ++frame_id) {
				progress(frame_id, total_img_num);
				std::string imgname = img_namevector[frame_id];

				if (frame_id == 0) {
					cv::Mat one_frame = cv::imread(imgpath + "/img/" + imgname);
				
					//siamrpn++ net1 前处理
					siamrpn_preprocess(target_pos, target_sz, im_sz, im_patch_template, one_frame, context_amount, scale, z_size, z_size);

					Tensor net1_input_tensor = CvMat2Tensor(im_patch_template, network_1);
					//初始化ImageMake
					if (netinfo_1.ImageMake_on) {
						Operation ImageMake_net1 = netinfo_1.ImageMake_;
						buyi_backend_1.initOp(ImageMake_net1);
					}
					dmaInit(run_sim, netinfo_1.ImageMake_on, net1_input_tensor, device);

					//net1前向推理
					auto net1_output_tensors = session_1.forward({ net1_input_tensor });
					//手动同步
					for (auto&& tensor : net1_output_tensors) {
						tensor.waitForReady(1000ms);
					}
					data_ptr_1 = net1_output_tensors[0];
					data_ptr_2 = net1_output_tensors[1];
					data_ptr_3 = net1_output_tensors[2];
					data_ptr_4 = net1_output_tensors[3];
					data_ptr_5 = net1_output_tensors[4];
					data_ptr_6 = net1_output_tensors[5];

					if (!run_sim) device.reset(1);
					// 计时
					#ifdef __linux__
						device.reset(1);
						calctime_detail(session_1);
					#endif
				}
				else {
					cv::Mat img = cv::imread(imgpath + "/img/" + imgname);
					//siamrpn++ net2 前处理
					siamrpn_preprocess(target_pos, target_sz, im_sz, im_patch, img, context_amount, scale, z_size, s_size);
					// 构造Icraft Tensor
					Tensor net2_input_tensor = CvMat2Tensor(im_patch, network_2);
					//初始化ImageMake
					if (frame_id == 1 && netinfo_2.ImageMake_on) {
						Operation ImageMake_net2 = netinfo_2.ImageMake_;
						buyi_backend_2.initOp(ImageMake_net2);
					}
					dmaInit(run_sim, netinfo_2.ImageMake_on, net2_input_tensor, device);
				

					//网络2：前向推理
					auto net2_output_tensors = session_2.forward({ net2_input_tensor, data_ptr_1, data_ptr_2, data_ptr_3, data_ptr_4, data_ptr_5, data_ptr_6 });
					//手动同步
					for (auto&& tensor : net2_output_tensors) {
						tensor.waitForReady(1000ms);
					}
					if (!run_sim) device.reset(1);
					// 计时
					#ifdef __linux__
						device.reset(1);
						calctime_detail(session_2);
					#endif
					//net2 后处理
					net2_postprocess_withcast(net2_output_tensors, target_pos, target_sz, anchor, window, window_influence, scale, im_sz);
				
					// cxywh2xywh 
					float pred_x = target_pos[0] - (target_sz[0] - 1) / 2;
					float pred_y = target_pos[1] - (target_sz[1] - 1) / 2;
					cv::rectangle(img, cv::Rect(pred_x, pred_y, target_sz[0], target_sz[1]), cv::Scalar(0, 255, 0), 2);
					//std::cout << "Pred: " << pred_x << " " << pred_y << " " << target_sz[0] << " " << target_sz[1] << std::endl;
					#ifdef _WIN32
						if (show) {
							cv::imshow("results", img);
							cv::waitKey(1);
						}
						if (save) {
							//保存bbox结果到txt文件中，xmin,ymin,width,height
							outFile << pred_x << " " << pred_y << " " << target_sz[0] << " " << target_sz[1] << std::endl;
						}
					#elif __linux__
						if (save) {
							//保存图片
							std::string save_path = img_path + imgname;
							std::regex rgx("\\.(?!.*\\.)"); // 匹配最后一个点号（.）之前的位置，且该点号后面没有其他点号
							std::string RES_PATH = std::regex_replace(save_path, rgx, "_result.");
							cv::imwrite(RES_PATH, img);
							//保存bbox结果到txt文件中，xmin,ymin,width,height
							outFile << pred_x << " " << pred_y << " " << target_sz[0] << " " << target_sz[1] << std::endl;
						}
					#endif

				}	
			}
			outFile.close();
		}
		//关闭设备
		Device::Close(device);
		}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
	return 0;

}
