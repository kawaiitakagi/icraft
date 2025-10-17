#pragma once
#include <opencv2/opencv.hpp>
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>

#include <modelzoo_utils.hpp>


using namespace icraft::xrt;
struct Grid {
    uint16_t location_x = 0;
    uint16_t location_y = 0;
    uint16_t anchor_index = 0;
};


std::vector<float> get_stride(NetInfo& netinfo) {
    std::vector<float> stride;
    for (auto i : netinfo.o_cubic) {
        stride.emplace_back(netinfo.i_cubic[0].h / i.h);
    }
    return stride;
};

template <typename T>
Grid get_grid(int bits, T* tensor_data, int base_addr, int anchor_length) {
    Grid grid;
    uint16_t anchor_index;
    uint16_t location_y;
    uint16_t location_x;
    if (bits == 8)
    {
        anchor_index = (((uint16_t)tensor_data[base_addr + anchor_length - 1]) << 8) + (uint8_t)tensor_data[base_addr + anchor_length - 2];
        location_y = (((uint16_t)tensor_data[base_addr + anchor_length - 3]) << 8) + (uint8_t)tensor_data[base_addr + anchor_length - 4];
        location_x = (((uint16_t)tensor_data[base_addr + anchor_length - 5]) << 8) + (uint8_t)tensor_data[base_addr + anchor_length - 6];

    }
    else if (bits == 16)
    {
        anchor_index = (uint16_t)tensor_data[base_addr + anchor_length - 1];
        location_y = (uint16_t)tensor_data[base_addr + anchor_length - 2];
        location_x = (uint16_t)tensor_data[base_addr + anchor_length - 3];
    }
    grid.location_x = location_x;
    grid.location_y = location_y;
    grid.anchor_index = anchor_index;
    return grid;
}

template <typename T>
void get_cls_bbox(std::vector<int>& id_list, std::vector<float>& socre_list, std::vector<cv::Rect2f>& box_list, T* tensor_data, int base_addr,
    Grid& grid, float & SCALE, int stride,
    std::vector<float> anchor, int N_CLASS, float THR_F, bool MULTILABEL) {
    if (!MULTILABEL) {
        auto _score_ = sigmoid(tensor_data[base_addr + 4] * SCALE);
        auto class_ptr_start = tensor_data + base_addr + 5;
        auto max_prob_ptr = std::max_element(class_ptr_start, class_ptr_start + N_CLASS);
        int max_index = std::distance(class_ptr_start, max_prob_ptr);
        auto _prob_ = sigmoid(*max_prob_ptr * SCALE);
        auto realscore = _score_ * _prob_;
        if (realscore > THR_F) {
            std::vector<float> xywh = sigmoid(tensor_data, SCALE, base_addr, 4);


            xywh[0] = (2 * xywh[0] + grid.location_x - 0.5) * stride;
            xywh[1] = (2 * xywh[1] + grid.location_y - 0.5) * stride;

            xywh[2] = 4 * powf(xywh[2], 2) * anchor[0];
            xywh[3] = 4 * powf(xywh[3], 2) * anchor[1];
            id_list.emplace_back(max_index);
            socre_list.emplace_back(realscore);
            box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),
                (xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
        }
    }
    else {
        for (size_t cls_idx = 0; cls_idx < N_CLASS; cls_idx++) {
            //auto realscore = this->getRealScore(tensor_data, obj_ptr_start, norm, i);

            auto _score_ = sigmoid(tensor_data[base_addr + 4] * SCALE);
            auto _prob_ = sigmoid(tensor_data[base_addr + 5 + cls_idx] * SCALE);
            auto realscore = _score_ * _prob_;
            if (realscore > THR_F) {
                //auto bbox = this->getBbox(tensor_data, norm, obj_ptr_start, location_x, location_y, stride, anchor);
                std::vector<float> xywh = sigmoid(tensor_data, SCALE, base_addr, 4);

                xywh[0] = (2 * xywh[0] + grid.location_x - 0.5) * stride;
                xywh[1] = (2 * xywh[1] + grid.location_y - 0.5) * stride;

                xywh[2] = 4 * powf(xywh[2], 2) * anchor[0];
                xywh[3] = 4 * powf(xywh[3], 2) * anchor[1];

                id_list.emplace_back(cls_idx);
                socre_list.emplace_back(realscore);
                box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),
                    (xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
            }
        }
    }
}


void post_detpost_hard(const std::vector<Tensor>& output_tensors, PicPre& img, NetInfo& netinfo,
    float conf, float iou_thresh, bool MULTILABEL, bool fpga_nms, int N_CLASS, 
    std::vector<std::vector<std::vector<float>>> &ANCHORS,std::vector<std::string>& LABELS, 
    bool & show, bool & save , std::string &resRoot, std::string & name,icraft::xrt::Device device,bool & run_sim) {

    std::vector<int> id_list;
    std::vector<float> socre_list;
    std::vector<cv::Rect2f> box_list;
    std::vector<float> stride = get_stride(netinfo);
    for (size_t i = 0; i < output_tensors.size(); i++) {
        
        auto host_tensor = output_tensors[i].to(HostDevice::MemRegion());
        int output_tensors_bits = output_tensors[i].dtype()->element_dtype.getStorageType().bits();
        int obj_num = output_tensors[i].dtype()->shape[2];
        int anchor_length = output_tensors[i].dtype()->shape[3];
        if (output_tensors_bits == 16) {
            auto tensor_data = (int16_t*)host_tensor.data().cptr();
            for (size_t obj = 0; obj < obj_num; obj++) {
                int base_addr = obj * anchor_length; 
                Grid grid = get_grid(output_tensors_bits, tensor_data, base_addr, anchor_length);
                get_cls_bbox(id_list, socre_list, box_list, tensor_data, base_addr,grid, netinfo.o_scale[i], stride[i],ANCHORS[i][grid.anchor_index], N_CLASS, conf, MULTILABEL);
            }
        }
        else {
            auto tensor_data = (int8_t*)host_tensor.data().cptr();
            for (size_t obj = 0; obj < obj_num; obj++) {
                int base_addr = obj * anchor_length; 
                Grid grid = get_grid(output_tensors_bits, tensor_data, base_addr, anchor_length);
                get_cls_bbox(id_list, socre_list, box_list, tensor_data, base_addr,grid, netinfo.o_scale[i], stride[i], ANCHORS[i][grid.anchor_index], N_CLASS, conf, MULTILABEL);

            }
        }
    }
    std::vector<std::tuple<int, float, cv::Rect2f>> nms_res;
    //std::cout << "number of results before nms = " << id_list.size() << '\n';
    //auto post_start = std::chrono::system_clock::now();
    if (fpga_nms&& !run_sim) {
        nms_res = nms_hard(box_list, socre_list, id_list, iou_thresh, device);
    }
    else {
        nms_res = nms_soft(id_list, socre_list, box_list, iou_thresh);   // 后处理 之 NMS

    }
    //auto post_end = std::chrono::system_clock::now();
    //auto post_time = std::chrono::duration_cast<std::chrono::microseconds>(post_end - post_start);
    //std::cout << "nms_time = " << double(post_time.count() / 1000.f) << '\n';
    //std::cout << "number of results after nms = " << nms_res.size() << '\n';
    std::vector<std::vector<float>> output_res = coordTrans(nms_res, img);

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


}

void post_detpost_soft(const std::vector<Tensor>& output_tensors, PicPre& img, std::vector<std::string>& LABELS,
    std::vector<std::vector<std::vector<float>>>& ANCHORS, NetInfo& netinfo, int N_CLASS, float conf, float iou_thresh,
    bool& show, bool& save, std::string& resRoot, std::string& name) {

    std::vector<int> id_list;
    std::vector<float> socre_list;
    std::vector<cv::Rect2f> box_list;
    std::vector<float> stride = get_stride(netinfo);
    for (int yolo = 0; yolo < output_tensors.size(); yolo++) {
        int _H = output_tensors[yolo].dtype()->shape[1];
        int _W = output_tensors[yolo].dtype()->shape[2];
        auto host_tensor = output_tensors[yolo].to(HostDevice::MemRegion());
        auto tensor_data = (float*)host_tensor.data().cptr();

        for (size_t h = 0; h < _H; h++) {
            int _h = h;
            for (size_t w = 0; w < _W; w++) {
                int _w = w;
                for (size_t anchor_index = 0; anchor_index < ANCHORS[yolo].size(); anchor_index++) {
                    int _anchor_id = anchor_index;
                    auto one_head_stride = stride[yolo];
                    std::vector<float> one_head_anchor = {};
                    if (ANCHORS.size() != 0) {
                        one_head_anchor = ANCHORS[yolo][anchor_index];
                    }

                    auto boxPtr = tensor_data + h * _W * (N_CLASS + 5) * ANCHORS[yolo].size()
                        + w * (N_CLASS + 5) * ANCHORS[yolo].size() + anchor_index * (N_CLASS + 5);
                    auto scorePtr = boxPtr + 4;
                    auto classPtr = boxPtr + 5;
                    auto _score_ = sigmoid(*scorePtr);
                    float* max_prob_ptr = std::max_element(classPtr, classPtr + N_CLASS);
                    int max_index = std::distance(classPtr, max_prob_ptr);
                    auto _prob_ = sigmoid(*max_prob_ptr);
                    auto realscore = _score_ * _prob_;

                    auto xywh = sigmoid(boxPtr, 4);
                    xywh[0] = (2 * xywh[0] + w - 0.5) * one_head_stride;
                    xywh[1] = (2 * xywh[1] + h - 0.5) * one_head_stride;
                    xywh[2] = 4 * powf(xywh[2], 2) * one_head_anchor[0];
                    xywh[3] = 4 * powf(xywh[3], 2) * one_head_anchor[1];
                    if(realscore >conf){
                        id_list.emplace_back(max_index);
                        socre_list.emplace_back(realscore);
                        box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),
                            (xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
                    }
                }
            }
        }
    }

    std::vector<std::tuple<int, float, cv::Rect2f>> nms_res = nms_soft(id_list, socre_list, box_list, iou_thresh);   // 后处理 之 NMS

    //std::cout << "number of results = " << nms_res.size() << '\n';
    std::vector<std::vector<float>> output_res = coordTrans(nms_res, img);


    #ifdef _WIN32
    if (show) {
        visualize(output_res, img.ori_img, resRoot, name, LABELS);
    }
    if (save) {
        saveRes(output_res, resRoot, name);
    }
    #endif
    

}

// smooth
const float ALPHA = 0.5f;
const float SMOOTH_IOU = 0.80f;

using YoloPostResult = std::tuple<std::vector<int>, std::vector<float>, std::vector<cv::Rect2f>  >; // box_list, id_list, score_list

YoloPostResult post_detpost_plin(const std::vector<Tensor>& output_tensors, YoloPostResult& last_frame_result,NetInfo& netinfo,
    float conf, float iou_thresh, bool MULTILABEL, bool fpga_nms, int N_CLASS,
    std::vector<std::vector<std::vector<float>>>& ANCHORS, icraft::xrt::Device device) {

    std::vector<int> id_list;
    std::vector<float> socre_list;
    std::vector<cv::Rect2f> box_list;
    std::vector<float> stride = get_stride(netinfo);
    for (size_t i = 0; i < output_tensors.size(); i++) {

        auto host_tensor = output_tensors[i].to(HostDevice::MemRegion());
        int output_tensors_bits = output_tensors[i].dtype()->element_dtype.getStorageType().bits();
        int obj_num = output_tensors[i].dtype()->shape[2];
        int anchor_length = output_tensors[i].dtype()->shape[3];
        if (output_tensors_bits == 16) {
            auto tensor_data = (int16_t*)host_tensor.data().cptr();
            for (size_t obj = 0; obj < obj_num; obj++) {
                int base_addr = obj * anchor_length;
                Grid grid = get_grid(output_tensors_bits, tensor_data, base_addr, anchor_length);
                get_cls_bbox(id_list, socre_list, box_list, tensor_data, base_addr, grid, netinfo.o_scale[i], stride[i], ANCHORS[i][grid.anchor_index], N_CLASS, conf, MULTILABEL);
            }
        }
        else {
            auto tensor_data = (int8_t*)host_tensor.data().cptr();
            for (size_t obj = 0; obj < obj_num; obj++) {
                int base_addr = obj * anchor_length;
                Grid grid = get_grid(output_tensors_bits, tensor_data, base_addr, anchor_length);
                get_cls_bbox(id_list, socre_list, box_list, tensor_data, base_addr, grid, netinfo.o_scale[i], stride[i], ANCHORS[i][grid.anchor_index], N_CLASS, conf, MULTILABEL);

            }
        }
    }
    std::vector<std::tuple<int, float, cv::Rect2f>> nms_res;
    if (fpga_nms) {
        nms_res = nms_hard(box_list, socre_list, id_list, iou_thresh, device);
    }
    else {
        nms_res = nms_soft(id_list, socre_list, box_list, iou_thresh);   // 后处理 之 NMS

    }


    // // 对前后帧的结果求平均
    auto id_list_last_frame = std::get<0>(last_frame_result);
    auto score_list_last_frame = std::get<1>(last_frame_result);
    auto box_list_last_frame = std::get<2>(last_frame_result);

    for (auto  idx_score_bbox : nms_res) {
        for (size_t i = 0; i < box_list_last_frame.size(); ++i) {
            if ((1.f - cv::jaccardDistance(std::get<2>(idx_score_bbox), box_list_last_frame[i])) > SMOOTH_IOU && (std::get<0>(idx_score_bbox) == id_list_last_frame[i])) {

                std::get<2>(idx_score_bbox).x = box_list_last_frame[i].x * ALPHA + std::get<2>(idx_score_bbox).x * (1.0f - ALPHA);
                std::get<2>(idx_score_bbox).y = box_list_last_frame[i].y * ALPHA + std::get<2>(idx_score_bbox).y * (1.0f - ALPHA);
                std::get<2>(idx_score_bbox).width = box_list_last_frame[i].width * ALPHA + std::get<2>(idx_score_bbox).width * (1.0f - ALPHA);
                std::get<2>(idx_score_bbox).height = box_list_last_frame[i].height * ALPHA + std::get<2>(idx_score_bbox).height * (1.0f - ALPHA);
                std::get<1>(idx_score_bbox) = score_list_last_frame[i] * ALPHA + std::get<1>(idx_score_bbox) * (1.0f - ALPHA);
                break;
            }
        }
    }
    std::vector<int> id_list_ret;   //id_list_ret.reserve(nms_indices.size());
    std::vector<float> score_list_ret; //score_list_ret.reserve(nms_indices.size());
    std::vector<cv::Rect2f> box_list_ret; //box_list_ret.reserve(nms_indices.size());
    for (auto  idx_score_bbox : nms_res) {
        // store 
        id_list_ret.emplace_back(std::get<0>(idx_score_bbox));
        score_list_ret.emplace_back(std::get<1>(idx_score_bbox));
        box_list_ret.emplace_back(std::get<2>(idx_score_bbox));
    }
    return YoloPostResult { id_list_ret, score_list_ret, box_list_ret};

}
