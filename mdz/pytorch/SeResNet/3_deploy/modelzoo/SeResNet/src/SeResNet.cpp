#include <iostream>
#include <fstream>
#include <regex>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/buyi_device.h>
#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/cuda/device.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>

#include <random>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include "yaml-cpp/yaml.h"
#include "utils.hpp"
#include "et_device.hpp"
#include "icraft_utils.hpp"

using namespace icraft::xrt;
using namespace icraft::xir;
namespace fs = std::filesystem;

int main(int argc, char* argv[])
{
    YAML::Node config = YAML::LoadFile(argv[1]);


    // icraft模型部署相关参数配置
    auto imodel = config["imodel"];
    // 仿真上板的jrpath配置
    std::string folderPath = imodel["dir"].as<std::string>();
    bool run_sim = imodel["sim"].as<bool>();
    bool cudamode = imodel["cudamode"].as<bool>();
    std::string JSON_PATH = getJrPath(run_sim, folderPath, imodel["stage"].as<std::string>());
    std::regex rgx3(".json");
    std::string RAW_PATH = std::regex_replace(JSON_PATH, rgx3, ".raw");
    // URL配置
    std::string ip = imodel["ip"].as<std::string>();
    // 可视化配置
    bool show = imodel["show"].as<bool>();
    bool save = imodel["save"].as<bool>();
    bool eval = imodel["eval"].as<bool>();


    // 加载network
    Network network = loadNetwork(JSON_PATH, RAW_PATH);
    //初始化netinfo
    NetInfo netinfo = NetInfo(network);
    // 打开device
    Device device = openDevice(run_sim, ip, netinfo.mmu || imodel["mmu"].as<bool>(), cudamode);
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
    std::pair<int, int> resize_hw = param["resize_hw"].as<std::pair<int, int>>();
    std::pair<int, int> crop_hw = param["crop_hw"].as<std::pair<int, int>>();

    // 统计图片数量
    int index = 0;
    auto namevector = toVector(imgList);
    int totalnum = namevector.size();
    std::vector<int> pred_lables;

    for (auto name : namevector) {


        progress(index, totalnum);
        index++;
        //-------------PRE PROCESS-----------------------//
        std::string img_path = imgRoot + '/' + name;
        if (eval) {
            name = name.substr(0, name.length() - 2);
            img_path = imgRoot + "//" + name;
        }
        PicPre img(img_path, cv::IMREAD_COLOR);
        img.Resize(resize_hw, PicPre::SHORT_SIDE).rCenterCrop(crop_hw);
        Tensor img_tensor = CvMat2Tensor(img.dst_img, network);
        dmaInit(run_sim, netinfo.ImageMake_on, img_tensor, device);
        auto output_tensors = session.forward({ img_tensor});
        if (!run_sim) device.reset(1);
        // 计时
        #ifdef __linux__
        device.reset(1);
        calctime_detail(session);
        #endif

        auto host_tensor_class = output_tensors[0].to(HostDevice::MemRegion());
        auto outputs_class = (float*)host_tensor_class.data().cptr();
        auto softmax_class = softmax(outputs_class, 1, 0, 5);
        //std::cout << output_tensors[0].dtype()->shape << std::endl;
        for (int i = 0; i < output_tensors[0].dtype()->shape[1]; i++) {
            auto c = softmax_class[i];
            std::ostringstream text;
            text << LABELS[i] << " : " << std::fixed << std::setprecision(6) << c;
            cv::putText(img.ori_img, text.str(), cv::Point2f(10, 20 + i * 15), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(128, 0, 0), 2);
        }
        //saveRes
        std::regex rgx("\\.(?!.*\\.)"); // 匹配最后一个点号（.）之前的位置，且该点号后面没有其他点号
        std::string RES_NAME = std::regex_replace(name, rgx, "_result.");
        #ifdef _WIN32
        if (show) {
            cv::imshow("results", img.ori_img);
            cv::waitKey(0);

        }
        if (save) {
            cv::imwrite(resRoot + RES_NAME, img.ori_img);
        }
        if (eval) {
            int pred_lable = std::distance(softmax_class.begin(), std::max_element(softmax_class.begin(), softmax_class.end()));
            pred_lables.emplace_back(pred_lable);


        }
        #elif __linux__
        if (save) {
            cv::imwrite(resRoot + RES_NAME, img.ori_img);
        }
        #endif

    }
    //关闭设备
    if (!run_sim) Device::Close(device);

    if (eval) {
        std::string save_path = resRoot + "/all_pred_res.txt" ;
  
        std::ofstream outputFile(save_path);
        if (!outputFile.is_open()) {
            std::cout << "Create txt file fail." << std::endl;
        }

        for (auto i : pred_lables) {
            
             outputFile << i << "\n";

        }
        outputFile.close();
    }
    return 0;
}
