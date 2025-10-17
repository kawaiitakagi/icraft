#include<onnxruntime_cxx_api.h>
#include<onnxruntime_c_api.h>
#include<chrono>
#include<dlfcn.h>
#include<filesystem>
#include<iostream>
#include<fstream>
#include<numeric>

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


void ftmp2tensor(const float* tensor_ptr, int64_t fp32_size, const std::string& ftmp_path)
{
	std::ifstream ftmp_stream(ftmp_path, std::ios::binary);

	if (!ftmp_stream) {
		throw std::runtime_error("[Error in Ftmp2XrtTensor] FtmpFile is invalid");
	}

	ftmp_stream.read((char*)tensor_ptr, fp32_size * 4);
};


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

	// 加载network
	Network network = loadNetwork(JSON_PATH, RAW_PATH);
	//初始化netinfo
	NetInfo netinfo = NetInfo(network);
	// 选择对网络进行切分
	auto network_view = network.view(netinfo.inp_shape_opid + 1);
	// 打开device
	Device device = openDevice(run_sim, ip, netinfo.mmu || imodel["mmu"].as<bool>(), cudamode);
	// 初始化session
	Session session = initSession(run_sim, network_view, device, netinfo.mmu || imodel["mmu"].as<bool>(), imodel["speedmode"].as<bool>(), imodel["compressFtmp"].as<bool>());

	// 开启计时功能
	// session.enableTimeProfile(true);
	// session执行前必须进行apply部署操作
	session.apply();

	// 数据集相关参数配置
	auto dataset = config["dataset"];
	std::string imgRoot = dataset["dir"].as<std::string>();

	// onnxruntime model 配置
	const std::string model_dir = "../imodel";
	const std::string model_name = "sac_donkey_car.onnx";
	const std::string model_path = model_dir + "/" + model_name;
	std::cout<< "load model from " << model_path <<std::endl;

	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "sac");
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(4);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	Ort::AllocatorWithDefaultOptions allocator;
	Ort::Session session_o(env, model_path.c_str(), session_options);
	Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	
	//Ort::Value::CreateTensor<float>(encoder_input, input_lstm_states[0], input_lstm_states[1]);
	std::vector<Ort::Value> inputs;
	std::vector<Ort::Value> sac_outputs;
	//input_path
	const std::string input_dir = "../io/inputs/sac";
	const std::string input_name = "obs_onnx.ftmp";
	const std::string input_path = input_dir + "/" + input_name;
	std::cout<< "load input from " << input_dir <<std::endl;
	//encoder_input
	std::vector<int64_t> encoder_input_dims = {1, 132};
	size_t encoder_input_numel = std::accumulate(encoder_input_dims.begin() + 1, encoder_input_dims.end(), (int)!encoder_input_dims.empty(), std::multiplies<int>());
	std::vector<float> encoder_input_data(encoder_input_numel);
	ftmp2tensor(encoder_input_data.data(), encoder_input_numel, input_path);
	Ort::Value encoder_input_tensor = Ort::Value::CreateTensor<float>(memory_info, encoder_input_data.data(), encoder_input_numel, encoder_input_dims.data(), encoder_input_dims.size());
	inputs.push_back(std::move(encoder_input_tensor));
	
	std::vector<const char*> input_names = {"obs"};
	std::vector<const char*> output_names = {"57"};

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
		std::cout << '\n' <<"ae-npu inference time: " << duration.count() << " μs." <<std::endl;
		// 后处理
		auto host_tensor = output_tensors[0].to(HostDevice::MemRegion());
		auto tensor_data = (float*)host_tensor.data().cptr();

		auto output_base = new float[64];
		const std::string output_path = imgRoot + '/' + "ae_output.ftmp";
		std::ifstream output_file(output_path, std::ios::binary);
		output_file.read(reinterpret_cast<char*>(output_base), 129 * sizeof(float));
		output_file.close();
		float cosine_sim = cosineSimilarity(output_base, tensor_data, 64);
        std::cout <<"Npu infer cosine_similarity: " << cosine_sim << std::endl;
		
		// onnxruntime 
		auto start_o = std::chrono::high_resolution_clock::now();
		sac_outputs = session_o.Run(
			Ort::RunOptions{nullptr}, 
			input_names.data(), 
			inputs.data(), 
			inputs.size(),
			output_names.data(),
			output_names.size()
		);
		auto end_o = std::chrono::high_resolution_clock::now();
		auto duration_o = std::chrono::duration_cast<std::chrono::microseconds>(end_o - start_o);
		std::cout<< "sac-onnxruntime inference time: " << duration_o.count() << " μs." <<std::endl;

		if (!run_sim) device.reset(1);
		// 计时
		#ifdef __linux__
				device.reset(1);
				// calctime_detail(session);
		#endif
	}
	// print output 
	float* output1 = sac_outputs[0].GetTensorMutableData<float>();
	std::cout<< "output1: " <<std::endl;
	for (size_t j = 0; j < 3; j++)
	{
		std::cout<< output1[j] <<" ";
	}
	//关闭设备
	Device::Close(device);
	return 0;
}