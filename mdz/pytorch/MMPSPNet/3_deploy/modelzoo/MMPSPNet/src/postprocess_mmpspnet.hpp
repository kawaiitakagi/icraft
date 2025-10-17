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

// Palette for dataset Cityscapes
static std::vector<std::vector<uint8_t>> PALETTE_CITYSCAPES = { {128, 64, 128}, {244, 35, 232}, {70, 70, 70}, {102, 102, 156},
            {190, 153, 153}, {153, 153, 153}, {250, 170, 30}, {220, 220, 0},
            {107, 142, 35}, {152, 251, 152}, {70, 130, 180}, {220, 20, 60},
            {255, 0, 0}, {0, 0, 142}, {0, 0, 70}, {0, 60, 100}, {0, 80, 100},
            {0, 0, 230}, {119, 11, 32} };

void map_plot(const cv::Mat& src_mat, cv::Mat& img_id_mat, cv::Mat& img_color_mat, const cv::Mat& dst_img, cv::Mat& res_color_mat) {
    //------------ Finds id of every pixels ----------------//
    std::vector<std::vector<uint8_t>> PALETTE = PALETTE_CITYSCAPES;
    const int channels = src_mat.channels();
    int index;
    std::vector<int> img_id;
    for (int row = 0; row < src_mat.rows; row++) {
        const float* pointer = src_mat.ptr<float>(row);
        for (int col = 0; col < src_mat.cols; col++) {
            index = std::distance(pointer + col * channels, std::max_element(pointer + col * channels, pointer + (col + 1) * channels));
            img_id.emplace_back(index);
        }
    }
    img_id_mat = cv::Mat(img_id).clone();
    img_id_mat = img_id_mat.reshape(1, src_mat.rows);

    //------------- uses palette for coloring  ------------//
    std::vector<uint8_t> img_color;
    for (auto i : img_id) {
        img_color.emplace_back(PALETTE[i][2]);
        img_color.emplace_back(PALETTE[i][1]);
        img_color.emplace_back(PALETTE[i][0]);
    }
    img_color_mat = cv::Mat(img_color).clone();
    img_color_mat = img_color_mat.reshape(3, src_mat.rows);

    cv::Mat src;
    dst_img.convertTo(src, CV_8UC3);
    cv::scaleAdd(img_color_mat, 0.8, src, res_color_mat);
}


void post_process_seg(const std::vector<Tensor>& result_tensor, PicPre& img, NetInfo& netinfo, bool& show, bool& save, std::string& resRoot, std::string& name
) {
    auto host_tensor = result_tensor[0].to(HostDevice::MemRegion());
    auto tensor_data = (float*)host_tensor.data().cptr();
    cv::Mat net_result_mat = cv::Mat(result_tensor[0].dtype()->shape[1], result_tensor[0].dtype()->shape[2], CV_32FC(result_tensor[0].dtype()->shape[3]), tensor_data);
    auto h = netinfo.i_cubic[0].h;
    auto w = netinfo.i_cubic[0].w;
    cv::Mat resultMat;
    cv::resize(net_result_mat, resultMat, cv::Size(w, h), cv::INTER_LINEAR);

    cv::Mat img_id_mat;
    cv::Mat img_color_mat;
    cv::Mat res_color_mat;
    map_plot(resultMat, img_id_mat, img_color_mat, img.dst_img, res_color_mat);

    #ifdef _WIN32
        if (show) {
            cv::imshow("results", img_color_mat);
            cv::waitKey(0);
        }
        if (save) {
            //测试精度时保存结果
            std::vector<std::string> img_name_list = split(name, "/");
            std::string img_name = img_name_list[3];
            std::string dataset_name = img_name_list[2];
            checkDir(resRoot + '/' + dataset_name);
            cv::imwrite(resRoot + '/' + dataset_name + '/' + img_name, img_id_mat);
            cv::waitKey(0);
        }
    #elif __linux__
        std::string save_path = resRoot + '/' + name;
        std::regex rgx("\\.(?!.*\\.)"); // 匹配最后一个点号（.）之前的位置，且该点号后面没有其他点号
        std::string RES_PATH = std::regex_replace(save_path, rgx, "_result.");
        cv::imwrite(RES_PATH, img_color_mat);
        
    #endif
}

