
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/buyi_device.h>
#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/cuda/device.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <fstream>
#include <opencv2/opencv.hpp>
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
	// 打开device
	Device device = openDevice(run_sim, ip,netinfo.mmu || imodel["mmu"].as<bool>());
	// 初始化session
	Session session = initSession(run_sim, network, device, netinfo.mmu || imodel["mmu"].as<bool>(), imodel["speedmode"].as<bool>(), imodel["compressFtmp"].as<bool>());
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
	int N_CLASS = param["number_of_class"].as<int>();
	// 统计输入数量
	int index = 0;
	auto namevector = toVector(imgList);
	int totalnum = namevector.size();
	for (auto name : namevector) {
		progress(index, totalnum);
		index++;
		std::string img_path = imgRoot + '/' + name;
		// read FTMP and convert to Tensor
		auto img_tensor = hostbackend::utils::Ftmp2Tensor(img_path, network.inputs()[0].tensorType());
		
		dmaInit(run_sim, netinfo.ImageMake_on,img_tensor, device);
		
		std::vector<Tensor> outputs = session.forward({ img_tensor });
		// check outputs output1 = [1,40],output2 = [1,3,3],output3 = [1,64,64]
		// for(auto output : outputs){
		// 	std::cout << output.dtype()->shape << std::endl;
		// }
		if(!run_sim) device.reset(1);
		// 计时
		#ifdef __linux__
		device.reset(1);
		calctime_detail(session);
		#endif
		
		// post process 寻找最大值对应下标
		auto host_tensor = outputs[0].to(HostDevice::MemRegion());	
		auto pred = (float*)host_tensor.data().cptr();
        auto max_prob_ptr = std::max_element(pred, pred + N_CLASS);
		int max_index = std::distance(pred, max_prob_ptr);
		std::cout <<"\nPRED ="<<LABELS[max_index] << std::endl;
		if (save) {
        std::string save_path = resRoot + '/' + name;
        std::regex rgx("\\.(?!.*\\.)");
        std::string RES_PATH = std::regex_replace(save_path, rgx, "_result.");
		// save results
        #ifdef _WIN32
            std::regex reg(R"(\.(\w*)$)");
            save_path = std::regex_replace(save_path, reg, ".txt");
            std::ofstream outputFile(save_path);
            if (!outputFile.is_open()) {
                std::cout << "Create txt file fail." << std::endl;
            }
			outputFile << max_index << "\n";
            outputFile.close();
        #elif __linux__
            std::regex reg(R"(\.(\w*)$)");
            save_path = std::regex_replace(save_path, reg, "_result.txt");
            std::ofstream outputFile(save_path);
            if (!outputFile.is_open()) {
                std::cout << "Create txt file fail." << std::endl;
            }
			outputFile << max_index << "\n";
            outputFile.close();
        #endif
    }
		
	}
	//关闭设备
	Device::Close(device);
    return 0;
}
