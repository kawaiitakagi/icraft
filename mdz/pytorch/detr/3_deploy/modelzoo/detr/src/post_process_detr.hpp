#pragma once
#include <opencv2/opencv.hpp>
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>

#include <modelzoo_utils.hpp>
using namespace icraft::xrt;
void post_process(const std::vector<Tensor>& output_tensors, PicPre& img, 
    float conf, int N_CLASS, std::vector<std::string>& LABELS,
    bool& show, bool& save, std::string& resRoot, std::string& name) {
    auto host_tensor_class = output_tensors[0].to(HostDevice::MemRegion());
    auto outputs_class = (float*)host_tensor_class.data().cptr();
    auto host_tensor_bbox = output_tensors[1].to(HostDevice::MemRegion());
    auto outputs_bbox = (float*)host_tensor_bbox.data().cptr();

    int ori_h = std::get<1>(img.src_dims);
    int ori_w = std::get<2>(img.src_dims);

    std::vector<int> id_list;
    std::vector<float> socre_list;
    std::vector<cv::Rect2f> box_list;
    std::vector<std::vector<float>> output_res;

    for (int i = 0; i < 100; i++) {

        auto classPtr = outputs_class + i * (N_CLASS+1);
        auto bboxPtr = outputs_bbox + i * 4;
        auto max_prob_ptr_softmax = softmax(classPtr, 0, (N_CLASS + 1));
        auto maxElementIndex = std::distance(max_prob_ptr_softmax.begin(), std::max_element(max_prob_ptr_softmax.begin(), max_prob_ptr_softmax.end() - 1));
        int max_index = std::distance(max_prob_ptr_softmax.begin(), std::max_element(max_prob_ptr_softmax.begin(), max_prob_ptr_softmax.end() - 1));
        if (max_prob_ptr_softmax[max_index] > conf) {
            float x = (*bboxPtr) * ori_w;
            float y = (*(bboxPtr + 1)) * ori_h;
            float w = (*(bboxPtr + 2)) * ori_w;
            float h = (*(bboxPtr + 3)) * ori_h;
            id_list.emplace_back(max_index);
            socre_list.emplace_back(max_prob_ptr_softmax[max_index]);
            box_list.emplace_back(cv::Rect2f((x - w / 2),
                (y - h / 2), w, h));
        }

    }
    //std::cout << "The num of res: " << id_list.size() << std::endl;
    for (int i = 0; i < id_list.size(); ++i) {
        float class_id = id_list[i];
        float score = socre_list[i];
        auto box = box_list[i];
        float x1 = box.tl().x;
        float y1 = box.tl().y;
        float x2 = box.br().x;
        float y2 = box.br().y;
        if (1) {
            x1 = checkBorder(x1, 0.f, (float)img.src_img.cols);
            y1 = checkBorder(y1, 0.f, (float)img.src_img.rows);
            x2 = checkBorder(x2, 0.f, (float)img.src_img.cols);
            y2 = checkBorder(y2, 0.f, (float)img.src_img.rows);
        }
        float w = x2 - x1;
        float h = y2 - y1;
        //bbox�����Ͻǵ��wh
        output_res.emplace_back(std::vector<float>({ class_id, x1, y1, w, h, score }));
    }
    #ifdef _WIN32
        if (show) {
            visualize(output_res, img.ori_img, resRoot, name, LABELS);
        }
    #endif
    if (save) {
        #ifdef _WIN32
        saveRes(output_res, resRoot, name);
        #elif __linux__
        visualize(output_res, img.ori_img, resRoot, name, LABELS);
        #endif
    }
    //visualize(output_data, img.ori_img, resRoot, "res", LABELS);
}