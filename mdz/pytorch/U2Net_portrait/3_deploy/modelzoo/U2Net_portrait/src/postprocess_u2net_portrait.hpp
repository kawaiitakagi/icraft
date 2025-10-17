#pragma once
#include <opencv2/opencv.hpp>
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>
#include <modelzoo_utils.hpp>

void post_process_u2net(const std::vector<Tensor>& result_tensor, PicPre& img, NetInfo& netinfo, bool& show, bool& save, std::string& resRoot, std::string& name
) {
    auto host_tensor = result_tensor[0].to(HostDevice::MemRegion());
    auto tensor_data = (float*)host_tensor.data().cptr();//获取结果的指针
    cv::Mat net_result_mat = cv::Mat(result_tensor[0].dtype()->shape[1], result_tensor[0].dtype()->shape[2], CV_32FC(result_tensor[0].dtype()->shape[3]), tensor_data);
    cv::Mat res = (1 - net_result_mat);
    double minValue, maxValue;
    cv::minMaxLoc(res, &minValue, &maxValue);
    cv::Mat out = (res - minValue) / (maxValue - minValue) * 255; 
    auto d = img.src_dims;
    auto h = std::get<1>(d);
    auto w = std::get<2>(d);
    cv::Mat resultMat;
    cv::resize(out, resultMat, cv::Size(w, h));
    // 转换为8位无符号整数类型
    cv::Mat resultMat_8u;
    resultMat.convertTo(resultMat_8u, CV_8U);

    #ifdef _WIN32
        if (show) {
            cv::imshow("results", resultMat_8u);
            cv::waitKey(0);
        }
        if (save) {
            std::string save_path = resRoot + '/' + name;
            std::regex rgx("\\.(?!.*\\.)"); // 匹配最后一个点号（.）之前的位置，且该点号后面没有其他点号
            std::string RES_PATH = std::regex_replace(save_path, rgx, "_result.");
            cv::imwrite(RES_PATH, resultMat_8u);
        }
    #elif __linux__
        if (save) {
            std::string save_path = resRoot + '/' + name;
            std::regex rgx("\\.(?!.*\\.)"); // 匹配最后一个点号（.）之前的位置，且该点号后面没有其他点号
            std::string RES_PATH = std::regex_replace(save_path, rgx, "_result.");
            cv::imwrite(RES_PATH, resultMat_8u);
        }
    #endif
}

