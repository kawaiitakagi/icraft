
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/buyi_device.h>
#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "postprocess_yolov5s+siamfc++.hpp"
#include "icraft_utils.hpp"
#include "yaml-cpp/yaml.h"
using namespace icraft::xrt;
using namespace icraft::xir;


int main(int argc, char* argv[])
{
	YAML::Node config = YAML::LoadFile(argv[1]);
	// icraft模型部署相关参数配置
	auto imodel = config["imodel"];
	// 仿真上板的jrpath配置
	bool run_sim = imodel["sim"].as<bool>();
	std::string folderPath_net1 = imodel["net1_dir"].as<std::string>();
	std::string JSON_PATH_net1 = getJrPath(run_sim, folderPath_net1, imodel["stage"].as<std::string>());
	std::regex rgx3(".json");
	std::string RAW_PATH_net1 = std::regex_replace(JSON_PATH_net1, rgx3, ".raw");
	std::cout << "as" << std::endl;

	std::string folderPath_net2 = imodel["net2_dir"].as<std::string>();
	std::string JSON_PATH_net2 = getJrPath(run_sim, folderPath_net2, imodel["stage"].as<std::string>());
	std::string RAW_PATH_net2 = std::regex_replace(JSON_PATH_net2, rgx3, ".raw");
	std::cout << "as" << std::endl;
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
	//netinfo.ouput_allinfo();

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


	// 数据集相关参数配置
	auto dataset = config["dataset"];
	std::string filesRoot = dataset["dir"].as<std::string>();
	std::string filesList = dataset["list"].as<std::string>();
	//std::string names_path = dataset["names"].as<std::string>();
	std::string resRoot = dataset["res"].as<std::string>();
	std::string txt_root = resRoot + "/res_txt/"; //txt文件保存路径
	std::string img_root = resRoot + "/res_img/"; //img文件保存路径
	checkDir(resRoot);
	if (save) {
		checkDir(txt_root);
		checkDir(img_root);
	}

	// siamfc++模型参数配置
	float context_amount = 0.5;
	float z_size = 127;
	float x_size = 303;
	// 创建汉宁窗
	float window_influence = 0.21;
	cv::Mat hanning = CreatHannWindow(17, 17);
	cv::Mat window = cv::Mat(289, 1, CV_32F, hanning.data);


	// 统计数据集文件数量
	int index = 0;
	auto file_namevector = toVector(filesList);
	int total_file_num = file_namevector.size();


	//逐文件进行单目标追踪
	for (auto filename : file_namevector) {
		progress(index, total_file_num);
		index++;
		std::string imgpath = filesRoot + "/" + filename + "/";
		std::string imgList = imgpath + filename + ".txt";
		auto img_namevector = toVector(imgList);
		int total_img_num = img_namevector.size();

		//从GT中获取target框的信息
		std::string gt = imgpath + "groundtruth.txt";
		std::cout <<"read groundtruth from " << gt << std::endl;

		//从gt读取target初始框
		std::vector<float> init_rect;
		std::ifstream file(gt);
		if (!file.is_open()) {
			// 处理文件打开失败的情况
			std::cout << gt + " open failed" << std::endl;
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
		// 获取初始帧目标的target_pos&size
		std::vector<float> target_pos = { (init_rect[0] + (init_rect[2] - 1) / 2),(init_rect[1] + (init_rect[3] - 1) / 2) };//中心点位置
		std::vector<float> target_sz = { init_rect[2],init_rect[3] };

		//输入图片尺寸
		cv::Mat frame = cv::imread(imgpath + "/img/" + img_namevector[0]);
		float im_h = frame.rows;
		float im_w = frame.cols;

		//按数据集中序列名称来命名结果保存的路径
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
		float scale = 1.0;
		std::vector<std::vector<float>> M_inversed = {    //仿射变换逆矩阵
			{1.0f, 0.0f, 0.0f},
			{0.0f, 1.0f, 0.0f}
		};
		cv::Mat xy_ctr = cv::Mat(289, 2, CV_32F);
		for (int i = 0; i < 17; i++) {
			for (int j = 0; j < 17; j++) {
				xy_ctr.at<float>(17 * i + j, 0) = 87 + 8 * j;
				xy_ctr.at<float>(17 * i + j, 1) = 87 + 8 * i;
			}
		}

		//逐帧进行单目标追踪
		for (int frame_id = 0; frame_id < total_img_num; ++frame_id) {
			progress(frame_id, total_img_num);
			std::string imgname = img_namevector[frame_id];
			if (frame_id == 0) {
				cv::Mat one_frame = cv::imread(imgpath + "/img/" + imgname);
				//siamfc++ net1 前处理
				siamfc_preprocess(target_pos, target_sz, M_inversed, scale, context_amount, z_size, z_size);
				// 仿射变换
				cv::Mat mat2x3 = (cv::Mat_<float>(2, 3) << M_inversed[0][0], 0, M_inversed[0][2], 0, M_inversed[1][1], M_inversed[1][2]);
				cv::warpAffine(one_frame, im_patch_template, mat2x3, cv::Size(z_size, z_size), (cv::INTER_LINEAR | cv::WARP_INVERSE_MAP), cv::BORDER_CONSTANT, (0, 0, 0));
				Tensor net_1_input_tensor = CvMat2Tensor(im_patch_template, network_1);

				//初始化ImageMake
				if (netinfo_1.ImageMake_on) {
					Operation ImageMake_net1 = netinfo_1.ImageMake_;
					buyi_backend_1.initOp(ImageMake_net1);
				}
				dmaInit(run_sim, netinfo_1.ImageMake_on, net_1_input_tensor, device);

				//net1前向推理
				auto net_1_output_tensors = session_1.forward({ net_1_input_tensor });
				
				data_ptr_1 = net_1_output_tensors[0];
				data_ptr_2 = net_1_output_tensors[1];

				//手动同步
				for (auto&& tensor : net_1_output_tensors) {
					tensor.waitForReady(1000ms);
				}

				if (!run_sim) device.reset(1);
				// 计时
				#ifdef __linux__
					device.reset(1);
					calctime_detail(session_1);
				#endif
			}
			else {
				cv::Mat one_frame = cv::imread(imgpath + "/img/" + imgname);

				//siamfc++ net2前处理
				siamfc_preprocess(target_pos, target_sz, M_inversed, scale, context_amount, z_size, x_size);

				// 仿射变换
				cv::Mat mat2x3 = (cv::Mat_<float>(2, 3) << M_inversed[0][0], 0, M_inversed[0][2], 0, M_inversed[1][1], M_inversed[1][2]);
				cv::warpAffine(one_frame, im_patch, mat2x3, cv::Size(x_size, x_size), (cv::INTER_LINEAR | cv::WARP_INVERSE_MAP), cv::BORDER_CONSTANT, (0, 0, 0));
				Tensor net_2_input_tensor = CvMat2Tensor(im_patch, network_2);

				//初始化ImageMake
				if (netinfo_2.ImageMake_on && frame_id == 1) {
					Operation ImageMake_net2 = netinfo_2.ImageMake_;
					buyi_backend_2.initOp(ImageMake_net2);
				}
				dmaInit(run_sim, netinfo_2.ImageMake_on, net_2_input_tensor, device);
				//net2 前向推理
				auto output_tensors = session_2.forward({ net_2_input_tensor, data_ptr_1, data_ptr_2 });

				// 手动同步
				for (auto&& tensor : output_tensors) {
					tensor.waitForReady(1000ms);
				}
				if (!run_sim) device.reset(1);
				// 计时
#ifdef __linux__
				device.reset(1);
				calctime_detail(session_2);
#endif

				//net2 后处理
				net2_postprocess_withcast(output_tensors, target_pos, target_sz, xy_ctr, window, window_influence, x_size, scale, im_w, im_h);
				//std::cout << "target: " << target_pos[0] << " " << target_pos[1] << " " << target_sz[0] << " " << target_sz[1] << std::endl;

				// cxywh2xywh 
				float pred_x = target_pos[0] - (target_sz[0] - 1) / 2;
				float pred_y = target_pos[1] - (target_sz[1] - 1) / 2;
				cv::rectangle(one_frame, cv::Rect(pred_x, pred_y, target_sz[0], target_sz[1]), cv::Scalar(0, 255, 0), 2);

				#ifdef _WIN32
					if (show) {
						cv::imshow("results", one_frame);
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
						cv::imwrite(RES_PATH, one_frame);
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
	return 0;
}
