
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/buyi_device.h>
#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/cuda/device.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <icraft-xrt/core/tensor.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <modelzoo_utils.hpp>
#include "icraft_utils.hpp"
#include "utils.hpp"
#include "yaml-cpp/yaml.h"
#include "yolov8_utils.hpp"

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
	std::vector<float> _norm = netinfo.o_scale;


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

	//	std::cout << "img.dst_img shape: (" << img.dst_img.rows << ", "
	//		<< img.dst_img.cols << ", " << img.dst_img.channels() << ")" << std::endl;

	//	img.Resize({ netinfo.i_cubic[0].h, netinfo.i_cubic[0].w}, PicPre::LONG_SIDE).rPad();
	//	img.Resize({ netinfo.i_cubic[0].h, netinfo.i_cubic[0].w}, PicPre::BOTH_SIDE);
		img.Resize({ netinfo.i_cubic[0].h, netinfo.i_cubic[0].w }, PicPre::SHORT_SIDE).rCenterCrop({ netinfo.i_cubic[0].h, netinfo.i_cubic[0].w });

		Tensor img_tensor = CvMat2Tensor(img.dst_img, network);

		dmaInit(run_sim, netinfo.ImageMake_on,img_tensor, device);
		
		std::vector<Tensor> outputs = session.forward({ img_tensor });

		if(!run_sim) device.reset(1);
		// 计时
		#ifdef __linux__
		device.reset(1);
		calctime_detail(session);
		#endif

		// 后处理
		auto host_tensor = outputs[0].to(HostDevice::MemRegion());
		int output_tensors_bits = outputs[0].dtype()->element_dtype.getStorageType().bits();
		//std::cout << "output_tensors_bits: " << output_tensors_bits << std::endl;
		int obj_num = outputs[0].dtype()->shape[1];
		//std::cout << "obj_num: " << obj_num << std::endl;
		//std::cout << "Output Data: " << host_tensor.data() << std::endl;
		auto tensor_data = (float*)host_tensor.data().cptr();

		struct LabelConfidence {
			int label;
			float confidence;
		};
		std::vector<LabelConfidence> label_confidences;
		// 将标签和置信度打包
		for (int i = 0; i < obj_num; i++) {
			label_confidences.push_back({ i, tensor_data[i] });
		}

		// 基于置信度降序排列
		std::sort(label_confidences.begin(), label_confidences.end(), [](const LabelConfidence& a, const LabelConfidence& b) {
			return a.confidence > b.confidence;
		});

		std::vector<int> topk_res;
		// 只选取排名前五的数据
		for (int i = 0; i < 5 && i < label_confidences.size(); i++) {
			auto c = label_confidences[i].confidence;
			int label_index = label_confidences[i].label;

			std::ostringstream text;
			text << LABELS[label_index] << ":" << std::fixed << std::setprecision(6) << c;
			cv::putText(img.ori_img, text.str(), cv::Point2f(10, 20 + i * 15), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(128, 0, 0), 2);
			topk_res.push_back(label_index);
		}

		//saveRes
        #ifdef _WIN32
		if (show) {
			cv::imshow("results", img.ori_img);
			cv::waitKey(0);

		}
		if (save) {
			auto res_path = resRoot + "//" + name;
			std::regex reg(R"(\.(\w*)$)");
			res_path = std::regex_replace(res_path, reg, ".txt");
			std::ofstream outputFile(res_path);
			if (!outputFile.is_open()) {
				std::cout << "Create txt file fail." << std::endl;
			}
			for (auto&& res : topk_res) {
				outputFile << res << std::endl;
		    }
			outputFile.close();
			cv::imwrite(resRoot + '/' + name, img.ori_img);
		}

		#elif __linux__
        if (save) {
            cv::imwrite(resRoot + '/' + name, img.ori_img);
        }
        #endif
	}
	//关闭设备
	Device::Close(device);
    return 0;
}
