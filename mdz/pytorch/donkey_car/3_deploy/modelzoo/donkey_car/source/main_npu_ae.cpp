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


float cosineSimilarity(const float* a, const float* b, int size) {
    float dot = 0.0;
    float magA = 0.0;
    float magB = 0.0;

    for (int i = 0; i < size; i++) {
        dot += a[i] * b[i];
        magA += a[i] * a[i];
        magB += b[i] * b[i];
    }

    magA = std::sqrt(magA);
    magB = std::sqrt(magB);

    return dot / (magA * magB);
}

int main()
{

	YAML::Node config = YAML::LoadFile("../cfg/donkey_car.yaml");
	// icraft模型部署相关参数配置
	auto imodel = config["imodel"];
	// 仿真上板的jrpath配置
	std::string folderPath = imodel["dir"].as<std::string>();
	bool run_sim = imodel["sim"].as<bool>();
    bool cudamode = imodel["cudamode"].as<bool>();
	std::string targetFileName;
	std::string JSON_PATH = getJrPath(run_sim, folderPath, imodel["stage"].as<std::string>());
	std::regex rgx3(".json");
	std::string RAW_PATH = std::regex_replace(JSON_PATH, rgx3, ".raw");
	// URL配置
	std::string ip = imodel["ip"].as<std::string>();
	// save output
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
	// std::string resRoot = dataset["res"].as<std::string>();
	// checkDir(resRoot);

	// 统计输入ftmp数量
	int totalnum = 100;
	for (int i = 0; i < totalnum; i++) {
		progress(i, totalnum);
		// std::string temp_path = '/' + std::to_string(i) + ".ftmp";
		std::string encoder_input = imgRoot + '/' + "encoder_input.ftmp";
		// 前处理
		std::vector<Tensor> input_tensors;
		input_tensors.push_back(hostbackend::utils::Ftmp2Tensor(encoder_input, network.inputs()[0].tensorType()));
		// forward
		auto start = std::chrono::high_resolution_clock::now();
		std::vector<Tensor> output_tensors = session.forward({ input_tensors });
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		std::cout<< "num_"<< i << " : inference time: " << duration.count() << " μs." <<std::endl;
		// 后处理
		auto host_tensor = output_tensors[0].to(HostDevice::MemRegion());
		auto tensor_data = (float*)host_tensor.data().cptr();

		auto output_base = new float[64];
		const std::string output_path = imgRoot + '/' + "ae_output.ftmp";
		std::ifstream output_file(output_path, std::ios::binary);
		output_file.read(reinterpret_cast<char*>(output_base), 129 * sizeof(float));
		output_file.close();
		float cosine_sim = cosineSimilarity(output_base, tensor_data, 64);
        std::cout << " cosine_similarity: " << cosine_sim << std::endl;

		// for (int j = 0; j < 64; j++) {
		// 	std::cout << "output: " << tensor_data[j] * 1 << std::endl;
		// }
		if (!run_sim) device.reset(1);
		// 计时
		#ifdef __linux__
				device.reset(1);
				calctime_detail(session);
		#endif
	}
	//关闭设备
	Device::Close(device);
	return 0;
}