#pragma once
#include <opencv2/opencv.hpp>
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>
#include <modelzoo_utils.hpp>
bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2) {
    if (mat1.empty() && mat2.empty()) {
        return true;
    }
    if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims ||
        mat1.channels() != mat2.channels()) {
        return false;
    }
    if (mat1.size() != mat2.size() || mat1.channels() != mat2.channels() || mat1.type() != mat2.type()) {
        return false;
    }
    int nrOfElements1 = mat1.total() * mat1.elemSize();
    if (nrOfElements1 != mat2.total() * mat2.elemSize()) return false;
    bool lvRet = memcmp(mat1.data, mat2.data, nrOfElements1) == 0;
    return lvRet;
}

// Palette for dataset VOC2012
static std::vector<std::vector<uint8_t>> PALETTE_VOC2012 = { {0, 0, 0}, {128, 0, 0}, {0, 128, 0}, {128, 128, 0}, {0, 0, 128},
            {128, 0, 128}, {0, 128, 128}, {128, 128, 128}, {64, 0, 0},
            {192, 0, 0}, {64, 128, 0}, {192, 128, 0}, {64, 0, 128},
            {192, 0, 128}, {64, 128, 128}, {192, 128, 128}, {0, 64, 0},
            {128, 64, 0}, {0, 192, 0}, {128, 192, 0}, {0, 64, 128} };

void map_plot(const cv::Mat& src_mat, cv::Mat& img_id_mat, cv::Mat& img_color_mat, const cv::Mat& dst_img) {
    //------------ Finds id of every pixels ----------------//
    std::vector<std::vector<uint8_t>> PALETTE = PALETTE_VOC2012;
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

    //可视化及保存图片
    cv::Mat src;
    dst_img.convertTo(src, CV_8UC3);
    cv::scaleAdd(img_color_mat, 0.8, src, img_color_mat);
}


void post_process_seg(const std::vector<Tensor>& result_tensor, PicPre& img, NetInfo& netinfo, bool& show, bool& save, std::string& resRoot, std::string& name
) {
    auto host_tensor = result_tensor[0].to(HostDevice::MemRegion());
    auto tensor_data = (float*)host_tensor.data().cptr();//获取结果的指针
    cv::Mat net_result_mat = cv::Mat(result_tensor[0].dtype()->shape[1], result_tensor[0].dtype()->shape[2], CV_32FC(result_tensor[0].dtype()->shape[3]), tensor_data);
    auto h = netinfo.i_cubic[0].h;
    auto w = netinfo.i_cubic[0].w;
    cv::resize(net_result_mat, net_result_mat, cv::Size(w, h), cv::INTER_LINEAR);

    cv::Mat img_id_mat;
    cv::Mat img_color_mat;
    map_plot(net_result_mat, img_id_mat, img_color_mat, img.dst_img);

    #ifdef _WIN32
        if (show) {
            cv::imshow("results", img_color_mat);
            cv::waitKey(0);
        }
        if (save) {
            std::string save_path = resRoot + '/' + name;
            //cv::imwrite(save_path, img_id_mat); //保存图片
            auto os0 = std::ofstream(save_path+".ftmp", std::ios::binary);
            result_tensor[0].dump(os0, "SFB"); //保存结果
        }
    #elif __linux__
        std::string save_path = resRoot + '/' + name;
        std::regex rgx("\\.(?!.*\\.)"); // 匹配最后一个点号（.）之前的位置，且该点号后面没有其他点号
        std::string RES_PATH = std::regex_replace(save_path, rgx, "_result.");
        cv::imwrite(RES_PATH, img_color_mat);
    #endif

}

