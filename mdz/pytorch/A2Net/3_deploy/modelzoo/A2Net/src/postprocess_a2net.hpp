#pragma once
#include <opencv2/opencv.hpp>
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>
#include <modelzoo_utils.hpp>

static std::vector<std::string> split(std::string& str, const std::string& delim) {
    std::string::size_type pos;
    std::vector<std::string> result;
    str += delim;
    int size = str.size();
    for (int i = 0; i < size; i++) {
        pos = str.find(delim, i);
        if (pos < size) {
            std::string s = str.substr(i, pos - i);
            result.push_back(s);
            i = pos + delim.size() - 1;
        }
    }
    return result;
}



void post_process_a2net(const std::vector<Tensor>& result_tensor, NetInfo& netinfo, bool& show, bool& save, std::string& resRoot, std::string& name
) {
    //std::cout << result_tensor[0].dtype()->shape << std::endl;//[1,256,256,1]
    auto host_tensor = result_tensor[0].to(HostDevice::MemRegion());
    auto tensor_data = (float*)host_tensor.data().cptr();
    cv::Mat net_result_mat = cv::Mat(result_tensor[0].dtype()->shape[1], result_tensor[0].dtype()->shape[2], CV_32FC1, tensor_data);
    float threshold = 0.5;
    cv::Mat pred;
    cv::threshold(net_result_mat, pred, threshold, 1.0, cv::THRESH_BINARY);
    
    pred = pred * 255;
    #ifdef _WIN32
        if (show) {
            cv::imshow("results", pred);
            cv::waitKey(0);
        }
        if (save) {
            std::string save_path = resRoot + '/' + name;
            std::regex rgx("\\.(?!.*\\.)"); // 匹配最后一个点号（.）之前的位置，且该点号后面没有其他点号
            std::string RES_PATH = std::regex_replace(save_path, rgx, "_result.");
            cv::imwrite(RES_PATH, pred);
}
    #elif __linux__
        if (save) {
            std::string save_path = resRoot + '/' + name;
            std::regex rgx("\\.(?!.*\\.)"); // 匹配最后一个点号（.）之前的位置，且该点号后面没有其他点号
            std::string RES_PATH = std::regex_replace(save_path, rgx, "_result.");
            cv::imwrite(RES_PATH, pred);
        }
    #endif
}

