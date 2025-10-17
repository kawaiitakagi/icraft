#include<onnxruntime_cxx_api.h>
#include<onnxruntime_c_api.h>
#include<chrono>
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

#include <unordered_map>
#include <vector>
#include <sstream>

using namespace icraft::xrt;
using namespace icraft::xir;


// 加载词表
std::unordered_map<std::wstring, int64_t> buildVocabMap(const std::string& filename) {
    std::unordered_map<std::wstring, int64_t> vocabMap;
    std::wifstream file(filename);
    std::wstring line;
    int64_t id = 0;

    if (!file.is_open()) {
        std::wcerr << L"无法打开文件: " << filename.c_str() << std::endl;
        return vocabMap;
    }

    file.imbue(std::locale(file.getloc(), new std::codecvt_utf8<wchar_t>));

    while (std::getline(file, line)) {
        // 去除行末的换行符
        line.erase(line.find_last_not_of(L"\r\n") + 1);
        vocabMap[line] = id++;
    }

    file.close();
    return vocabMap;
}



int main(int argc, char* argv[])
{
    try{
        std::locale::global(std::locale("en_US.UTF-8"));
        std::wcout.imbue(std::locale("en_US.UTF-8"));
        
        // ----------------------------------------icraft runtime 参数----------------------------------------------
        YAML::Node config = YAML::LoadFile(argv[1]);
        // YAML::Node config = YAML::LoadFile("../cfg/bert_cls.yaml");
        // icraft模型部署相关参数配置
        auto imodel = config["imodel"];
        // 仿真上板的jrpath配置
        std::string folderPath = imodel["dir"].as<std::string>();
        bool run_sim = imodel["sim"].as<bool>();
        bool show = imodel["show"].as<bool>();
        bool save = imodel["save"].as<bool>();

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
        auto network_view = network.view(netinfo.inp_shape_opid + 0);
        // 打开device
        Device device = openDevice(run_sim, ip, netinfo.mmu || imodel["mmu"].as<bool>());
        // 初始化session
        Session session = initSession(run_sim, network_view, device, netinfo.mmu || imodel["mmu"].as<bool>(), imodel["speedmode"].as<bool>(), imodel["compressFtmp"].as<bool>());

        // 开启计时功能
        session.enableTimeProfile(true);
        // session执行前必须进行apply部署操作
        session.apply();

        // 输入文件相关参数
        auto dataset = config["dataset"];
        std::string model_path = dataset["onnx_model_path"].as<std::string>(); // 输入onnx模型地址
        std::string inputFilePath = dataset["dir"].as<std::string>(); // 输入文本文件地址
        std::string output_path = dataset["res"].as<std::string>();  // 输出文本文件地址
        std::cout<< "input text from " << inputFilePath <<std::endl;
        std::cout<< "load model from " << model_path <<std::endl;
        


        // ----------------------------------------onnx model 参数----------------------------------------------
        
        // 初始化ONNX Runtime环境
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "embedding");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 加载onnx模型
        Ort::Session session_o(env, model_path.c_str(), session_options);

        // 设置编码
        std::locale::global(std::locale("en_US.UTF-8"));
        std::wcout.imbue(std::locale("en_US.UTF-8"));

        std::wifstream inputFile(inputFilePath);

        // 设置文件流为 UTF-8 编码
        inputFile.imbue(std::locale(inputFile.getloc(), new std::codecvt_utf8<wchar_t>));

        auto start_vocab = std::chrono::high_resolution_clock::now();
        std::unordered_map<std::wstring, int64_t> vocabMap = buildVocabMap("../io/input/vocab.txt");
        auto end_vocab = std::chrono::high_resolution_clock::now();

        if (!inputFile.is_open()) {
            std::wcerr << L"无法打开文件: " << inputFilePath.c_str() << std::endl;
            return 1;
        }

        std::wstring text;
        
        std::ofstream outputFile(output_path, std::ios::out);
        
        int round = 0;
        while (std::getline(inputFile, text)) {
            round++;
            std::cout << "Processing round  " << round << ",  ";

            auto start_getid = std::chrono::high_resolution_clock::now();

            std::vector<int64_t> input_ids;
            size_t pos = text.find_last_not_of(L"0123456789");
            std::wstring cleanedText = (pos != std::wstring::npos && pos < text.size() - 1) ? text.substr(0, pos + 1) : text;

            input_ids.push_back(101);  // 首字符填充特殊字符：[UNK]->101
            // 遍历 cleanedText 中的每一个字符
            for (wchar_t ch : cleanedText) {
                // 将字符转换为字符串并输出
                std::wstring charAsString(1, ch);
                if (vocabMap.find(charAsString) != vocabMap.end()) {
                    input_ids.push_back(vocabMap.at(charAsString));
                }
            }
            // 调整 input_ids 到 32 个字符
            while (input_ids.size() < 32) {
                input_ids.push_back(0);  // 补 0
            }
            if (input_ids.size() > 32) {
                input_ids.resize(32);  // 删除多余的
            }

            auto end_getid = std::chrono::high_resolution_clock::now();

            std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};

            // 获取输入和输出张量的名称
            std::vector<const char*> input_name = {"input_ids"};
            std::vector<const char*> output_name = {"embedding_output"};

            // 创建输入张量
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<Ort::Value> input_tensors_o;
            input_tensors_o.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size()));

            // 运行模型
            auto start_onnx = std::chrono::high_resolution_clock::now();

            auto output_tensors_o = session_o.Run(
                Ort::RunOptions{nullptr}, 
                input_name.data(), 
                input_tensors_o.data(), 
                input_tensors_o.size(),
                output_name.data(),
                output_name.size()
            );
            auto end_onnx = std::chrono::high_resolution_clock::now();
            
            // 获取输出
            float* embedding_out = output_tensors_o[0].GetTensorMutableData<float>();
            int64_t output_size = output_tensors_o[0].GetTensorTypeAndShapeInfo().GetElementCount();


            // -----------------------------icraft forward ------------------------------
            // 前处理：输入类型都调整为float
            std::vector<float> float_input_ids(input_ids.begin(), input_ids.end());
            // 生成输入： attention_mask
            std::vector<float> attention_mask(32, 0.0f);  // 初始化 attention_mask 为全 0.0f
            for (size_t i = 0; i < float_input_ids.size(); ++i) {
                if (float_input_ids[i] != 0.0f) {
                    attention_mask[i] = 1.0f;  // 如果 input_ids 中的值不是 0.0f，则 attention_mask 对应位置为 1.0f
                }
            }
            
            // 构建icraft输入tensor
            std::vector<icraft::xrt::Tensor> input_tensors;
            
            input_tensors.push_back(data2Tensor(embedding_out, network.inputs()[0]));
            input_tensors.push_back(data2Tensor(float_input_ids.data(), network.inputs()[1]));
            input_tensors.push_back(data2Tensor(attention_mask.data(), network.inputs()[2]));

            // forward
            auto start_icore = std::chrono::high_resolution_clock::now();
            std::vector<icraft::xrt::Tensor> output_tensors = session.forward(input_tensors);
            auto end_icore = std::chrono::high_resolution_clock::now();
            if (!run_sim) device.reset(1);
            // 后处理
            auto host_tensor = output_tensors[0].to(HostDevice::MemRegion());
            auto tensor_data = (float*)host_tensor.data().cptr();

            // 找到最大值的索引
            float* max_element = std::max_element(tensor_data, tensor_data + 10);
            size_t max_index = std::distance(tensor_data, max_element);
            // 打印类别
            if(show) {
                std::unordered_map<int, std::string> class_map = {
                    {0, "finance"},
                    {1, "realty"},
                    {2, "stocks"},
                    {3, "education"},
                    {4, "science"},
                    {5, "society"},
                    {6, "politics"},
                    {7, "sports"},
                    {8, "game"},
                    {9, "entertainment"}
                };
                std::cout << "Top1 class: " << class_map.at(max_index) << std::endl;          
            }
            if(save){
                outputFile << max_index << std::endl;
            }
            // 各阶段记时
            // auto duration_vocab = std::chrono::duration_cast<std::chrono::microseconds>(end_vocab - start_vocab);
            // auto duration_getid = std::chrono::duration_cast<std::chrono::microseconds>(end_getid - start_getid);
            // auto duration_onnx = std::chrono::duration_cast<std::chrono::microseconds>(end_onnx - start_onnx);
            // auto duration_icore = std::chrono::duration_cast<std::chrono::microseconds>(end_icore - start_icore);
            // std::cout<< "get_vocab time: " << duration_vocab.count() << " μs." <<std::endl;
            // std::cout<< "get_id time: " << duration_getid.count() << " μs." <<std::endl;
            // std::cout<< "onnx time: " << duration_onnx.count() << " μs." <<std::endl;
            // std::cout<< "icore time: " << duration_icore.count() << " μs." <<std::endl;
            
        }
        // 关闭文件
        outputFile.close();
        inputFile.close();
        //关闭设备
        calctime_detail(session);
        Device::Close(device);
    }
    catch(std::exception& e){
		std::cout<< e.what()<<std::endl;
	}
	return 0;
}
