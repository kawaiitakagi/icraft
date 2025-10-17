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
#include "post_process_detr.hpp"
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
    float conf = param["conf"].as<float>();
    int N_CLASS = param["number_of_class"].as<int>();

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
        img.Resize({ netinfo.i_cubic[0].h, netinfo.i_cubic[0].w }, PicPre::BOTH_SIDE);
        Tensor img_tensor = CvMat2Tensor(img.dst_img, network);
        dmaInit(run_sim, netinfo.ImageMake_on, img_tensor, device);
        auto output_tensors = session.forward({ img_tensor});
        
        if (!run_sim) device.reset(1);

        // 计时
        #ifdef __linux__
        device.reset(1);
        calctime_detail(session);
        #endif

        post_process(output_tensors, img, conf, N_CLASS, LABELS,show, save, resRoot, name);


    }

    //关闭设备
    Device::Close(device);
    return 0;
}
