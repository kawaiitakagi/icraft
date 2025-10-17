
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/buyi_device.h>
#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/cuda/device.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "postprocess_a2net.hpp"
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
	// 初始化netinfo
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
	std::string resRoot = dataset["res"].as<std::string>();
	checkDir(resRoot);



	// 统计图片数量
	int index = 0;
	auto namevector = toVector(imgList);
	int totalnum = namevector.size();

	for (auto name : namevector) {
		progress(index, totalnum);
		index++;
		std::string pic1_name = split(name, ";")[0];
		std::string pic2_name = split(name, ";")[1];
		std::string img1_path = imgRoot + '/' + pic1_name;
		std::string img2_path = imgRoot + '/' + pic2_name;
		// 前处理
		PicPre img1(img1_path, cv::IMREAD_COLOR);
		img1.Resize({ netinfo.i_cubic[0].h, netinfo.i_cubic[0].w}, PicPre::LONG_SIDE).rPad();
		//std::cout << img1.src_img.size << std::endl;
		Tensor img1_tensor = CvMat2Tensor(img1.dst_img, network);

		PicPre img2(img2_path, cv::IMREAD_COLOR);
		img2.Resize({ netinfo.i_cubic[0].h, netinfo.i_cubic[0].w }, PicPre::LONG_SIDE).rPad();
		Tensor img2_tensor = CvMat2Tensor(img2.dst_img, network);
		//std::cout << "IMK" << netinfo.ImageMake_on << std::endl;
		//dmaInit(run_sim, netinfo.ImageMake_on, img1_tensor, device);
		//dmaInit(run_sim, netinfo.ImageMake_on, img2_tensor, device);
		
		auto outputs = session.forward({ img1_tensor, img2_tensor });
		//std::cout << outputs[0].dtype()->shape << std::endl;
		//std::cout << outputs[1].dtype()->shape << std::endl;
		//std::cout << outputs[2].dtype()->shape << std::endl;
		//std::cout << outputs[3].dtype()->shape << std::endl;

		if(!run_sim) device.reset(1);
		
		// 计时
		#ifdef __linux__
		device.reset(1);
		calctime_detail(session);
		#endif
		//后处理
		post_process_a2net(outputs, netinfo, show, save, resRoot, pic1_name);

	}
	//关闭设备
	Device::Close(device);
    return 0;
}
