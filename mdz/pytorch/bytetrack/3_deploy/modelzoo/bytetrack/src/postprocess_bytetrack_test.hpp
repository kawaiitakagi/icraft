#pragma once
#include <opencv2/opencv.hpp>
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>
#include <modelzoo_utils.hpp>
#include "bt_include/BYTETracker.h"
#include "PicPre_bytetrack.hpp"

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

// 获取指定文件夹下的所有文件名
std::vector<std::string> getFileNames(const std::string& folderPath) {
    std::vector<std::string> fileNames;
    try {
        // 遍历文件夹
        for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
            if (entry.is_regular_file()) {  // 只处理普通文件
                fileNames.push_back(entry.path().filename().string());
            }
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error accessing folder: " << e.what() << std::endl;
    }
    return fileNames;
}

//根据每个head数，将原本1维的norm分组
std::vector<std::vector<float>> set_norm_by_head(int NOH, int parts, std::vector<float>& normalratio) {
    std::vector<std::vector<float>> _norm;
    for (size_t i = 0; i < NOH; i++)
    {
        std::vector<float> _norm_;
        for (size_t j = 0; j < parts; j++)
        {
            _norm_.push_back(normalratio[i * parts + j]);

        }
        _norm.push_back(_norm_);
    }
    return _norm;
}

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
void get_cls_bbox(std::vector<int> real_out_channels, std::vector<int>& id_list, std::vector<float>& socre_list, std::vector<cv::Rect2f>& box_list, T* tensor_data, int base_addr,
    Grid& grid, std::vector<float>& normratio_list, int stride,
    std::vector<float> anchor, int N_CLASS, float THR_F, bool MULTILABEL) {
    //yolox
    if (!MULTILABEL) {
        auto _score_ = sigmoid(tensor_data[base_addr] * normratio_list[0]);
        auto class_ptr_start = tensor_data + base_addr + real_out_channels[0] + real_out_channels[1];
        auto max_prob_ptr = std::max_element(class_ptr_start, class_ptr_start + N_CLASS);
        int max_index = std::distance(class_ptr_start, max_prob_ptr);
        auto _prob_ = sigmoid(*max_prob_ptr * normratio_list[2]);
        auto realscore = _score_ * _prob_;

        if (realscore > THR_F) {
            std::vector<float> xywh = {};
            for (size_t i = 0; i < 4; i++)
            {
                auto box = tensor_data[base_addr + real_out_channels[0] + i] * normratio_list[1];
                xywh.push_back(box);
            }
            
            xywh[0] = (xywh[0] + grid.location_x) * stride;
            xywh[1] = (xywh[1] + grid.location_y) * stride;
            xywh[2] = expf(xywh[2]) * stride;
            xywh[3] = expf(xywh[3]) * stride;

            id_list.emplace_back(max_index);
            socre_list.emplace_back(realscore);
            box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),
                (xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
        }
    }
    else {
        for (size_t cls_idx = 0; cls_idx < N_CLASS; cls_idx++) {
            //auto realscore = this->getRealScore(tensor_data, base_addr, norm, i);

            auto _score_ = sigmoid(tensor_data[base_addr] * normratio_list[0]);
            auto _prob_ = sigmoid(tensor_data[base_addr + real_out_channels[0] + real_out_channels[1] + cls_idx] * normratio_list[2]);
            auto realscore = _score_ * _prob_;
            if (realscore > THR_F) {
                std::vector<float> xywh = {};
                for (size_t i = 0; i < 4; i++)
                {
                    auto box = tensor_data[base_addr + real_out_channels[0] + i] * normratio_list[1];
                    xywh.push_back(box);
                }
                xywh[0] = (xywh[0] + grid.location_x) * stride;
                xywh[1] = (xywh[1] + grid.location_y) * stride;
                xywh[2] = expf(xywh[2]) * stride;
                xywh[3] = expf(xywh[3]) * stride;

                id_list.emplace_back(cls_idx);
                socre_list.emplace_back(realscore);
                box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),
                    (xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
            }
        }
    }
}

void obj_tracking(std::vector<std::vector<float>> output_res, PicPre_bytetrack& img, BYTETracker& tracker, std::vector<std::array<float, 10>>& res_export, cv::Mat& res_img, int frame_id)
{
    std::vector<byteTracker::Object> objects;
    for (auto i : output_res) {
        byteTracker::Object obj;
        obj.rect.x = i[1];
        obj.rect.y = i[2];
        obj.rect.width = i[3];
        obj.rect.height = i[4];
        obj.label = 0;
        obj.prob = i[5];
        objects.push_back(obj);
    }
    std::vector<STrack> output_stracks = tracker.update(objects);
    for (int i = 0; i < output_stracks.size(); i++) {
        auto frame = (float)(frame_id)+1;
        auto id = (float)(output_stracks[i].track_id);
        std::vector<float> tlwh = output_stracks[i].tlwh;
        bool vertical = tlwh[2] / tlwh[3] > 1.6;
        if (tlwh[2] * tlwh[3] > 20 && !vertical) {
            auto bb_left = tlwh[0];
            auto bb_top = tlwh[1];
            auto bb_width = tlwh[2];
            auto bb_height = tlwh[3];
            auto conf = output_stracks[i].score;
            std::array<float, 10> each_line = { frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1.f, -1.f, -1.f };
            for (size_t i = 2; i < 6; ++i) { 
                each_line[i] = std::round(each_line[i] * 10) / 10; // 四舍五入保留一位小数
            }
            res_export.push_back(each_line);
            Scalar s = tracker.get_color(output_stracks[i].track_id);
            putText(res_img, format("%.0f", id), Point(bb_left, bb_top), 0, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
            cv::rectangle(res_img, Rect(bb_left, bb_top, bb_width, bb_height), s, 2);
        }
    }
}
void post_detpost_hard(std::vector<int> real_out_channels, const std::vector<Tensor>& output_tensors, PicPre_bytetrack& img, NetInfo& netinfo, std::vector<std::vector<float>>norm,
    float conf, float iou_thresh, bool MULTILABEL, bool fpga_nms, int N_CLASS, 
    std::vector<std::vector<std::vector<float>>> &ANCHORS,std::vector<std::string>& LABELS, 
    bool& show, bool& save , std::string &resRoot, std::string & name,icraft::xrt::Device device,bool& run_sim, BYTETracker& tracker, std::vector<std::array<float, 10>>& res_export,int frame_id) {
    //-------------STEP1: DETECT POST PROCESS ------------------//
    std::vector<int> id_list;
    std::vector<float> socre_list;
    std::vector<cv::Rect2f> box_list;
    std::vector<float> stride = get_stride(netinfo);
    for (size_t i = 0; i < output_tensors.size(); i++) {
        auto host_tensor = output_tensors[i].to(HostDevice::MemRegion());
        int output_tensors_bits = netinfo.detpost_bit;
        int obj_num = output_tensors[i].dtype()->shape[2];
        int anchor_length = output_tensors[i].dtype()->shape[3];
        std::vector<float> _anchor_ = {};
        if (output_tensors_bits == 16) {
            auto tensor_data = (int16_t*)host_tensor.data().cptr();
            for (size_t obj = 0; obj < obj_num; obj++) {
                int base_addr = obj * anchor_length; 
                Grid grid = get_grid(output_tensors_bits, tensor_data, base_addr, anchor_length);
                if (ANCHORS.size() != 0) {
                    _anchor_ = ANCHORS[i][grid.anchor_index];
                }
                get_cls_bbox(real_out_channels,id_list, socre_list, box_list, tensor_data, base_addr,grid, norm[i], stride[i], _anchor_, N_CLASS, conf, MULTILABEL);
            }
        }
        else {
            auto tensor_data = (int8_t*)host_tensor.data().cptr();
            for (size_t obj = 0; obj < obj_num; obj++) {
                int base_addr = obj * anchor_length; 
                Grid grid = get_grid(output_tensors_bits, tensor_data, base_addr, anchor_length);
                if (ANCHORS.size() != 0) {
                    _anchor_ = ANCHORS[i][grid.anchor_index];
                }
                get_cls_bbox(real_out_channels,id_list, socre_list, box_list, tensor_data, base_addr,grid, norm[i], stride[i], _anchor_, N_CLASS, conf, MULTILABEL);

            }
        }
    }
    std::vector<std::tuple<int, float, cv::Rect2f>> nms_res;

    // 后处理 之 NMS
    if (fpga_nms&& !run_sim) {
        nms_res = nms_hard(box_list, socre_list, id_list, iou_thresh, device);
    }
    else {
        nms_res = nms_soft(id_list, socre_list, box_list, iou_thresh);   
    }
    //std::cout << "number of results after nms = " << nms_res.size() << '\n';
    std::vector<std::vector<float>> output_res = coordTrans(nms_res, img, false);// 测试精度时，check_border必须为false,避免将边界值优化


    //-------------STEP2: TRACK POST PROCESS ------------------//
    cv::Mat res_img = img.ori_img;
    obj_tracking(output_res, img, tracker, res_export, res_img, frame_id);

    //-------------VIS OR SAVE-----------------------//
    if (show) {
        cv::imshow("results", res_img);
        cv::waitKey(1);
    }
    if (save) {
        std::string save_path = resRoot + '/' + name + ".txt";;
        std::fstream outFile(save_path, std::ios::out | std::ios::trunc);
        for (size_t line = 0; line < res_export.size(); ++line) {
            for (size_t i = 0; i < 10; ++i) {
                outFile << res_export[line][i];
                if (i != 9) {
                    outFile << ",";
                }
            }
            outFile << "\n";
        }
        outFile.close();
    }
}

void post_detpost_soft(const std::vector<Tensor>& output_tensors, PicPre_bytetrack& img, std::vector<std::string>& LABELS,
    std::vector<std::vector<std::vector<float>>>& ANCHORS, NetInfo& netinfo, int N_CLASS, float conf, float iou_thresh,
    bool& MULTILABEL, BYTETracker& tracker, std::vector<std::array<float, 10>>& res_export, int frame_id, bool& show, bool& save, std::string& resRoot, std::string& name) {
    //-------------STEP1: DETECT POST PROCESS ------------------//
    std::vector<int> id_list;
    std::vector<float> socre_list;
    std::vector<cv::Rect2f> box_list;
    std::vector<float> stride = get_stride(netinfo);
    for (int yolo = 0; yolo < output_tensors.size() - 1; yolo = yolo + 3) {
        int _H = output_tensors[yolo].dtype()->shape[1];
        int _W = output_tensors[yolo].dtype()->shape[2];

        auto host_tensor_prob = output_tensors[yolo + 0].to(HostDevice::MemRegion());
        auto host_tensor_box = output_tensors[yolo + 1].to(HostDevice::MemRegion());
        auto host_tensor_cls = output_tensors[yolo + 2].to(HostDevice::MemRegion());
        auto tensor_data_prob = (float*)host_tensor_prob.data().cptr();
        auto tensor_data_box = (float*)host_tensor_box.data().cptr();
        auto tensor_data_cls = (float*)host_tensor_cls.data().cptr();

        for (size_t h = 0; h < _H; h++) {
            int _h = h;
            for (size_t w = 0; w < _W; w++) {
                int _w = w;
                auto one_head_stride = stride[yolo];

                auto probPtr = tensor_data_prob + h * _W * 1 + w * 1;
                auto boxPtr = tensor_data_box + h * _W * 4 + w * 4;
                auto clsPtr = tensor_data_cls + h * _W * N_CLASS + w * N_CLASS;

                if (!MULTILABEL) {
                    auto max_prob_ptr = std::max_element(clsPtr, clsPtr + N_CLASS);
                    int max_index = std::distance(clsPtr, max_prob_ptr);
                    auto _prob_ = sigmoid(*max_prob_ptr);//最大类别
                    auto _score_ = sigmoid(*probPtr);//框的置信度
                    auto realscore = _prob_ * _score_;

                    if (realscore > conf) {
                        std::vector<float> xywh = {};
                        //获取box
                        for (size_t i = 0; i < 4; i++) {
                            auto box = boxPtr[i];
                            xywh.push_back(box);
                        }
                        xywh[0] = (xywh[0] + w) * one_head_stride;
                        xywh[1] = (xywh[1] + h) * one_head_stride;
                        xywh[2] = expf(xywh[2]) * one_head_stride;
                        xywh[3] = expf(xywh[3]) * one_head_stride;

                        id_list.emplace_back(max_index);
                        socre_list.emplace_back(realscore);
                        box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),
                            (xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
                    }
                }
                else {
                    for (size_t cls_idx = 0; cls_idx < N_CLASS; cls_idx++) {
                        auto _prob_ = sigmoid(*(clsPtr + cls_idx));
                        auto _score_ = sigmoid(*probPtr);
                        auto realscore = _prob_ * _score_;
                        if (realscore > conf) {
                            std::vector<float> xywh = {};
                            for (size_t i = 0; i < 4; i++) {
                                auto box = boxPtr[i];
                                xywh.push_back(box);
                            }
                            xywh[0] = (xywh[0] + w) * one_head_stride;
                            xywh[1] = (xywh[1] + h) * one_head_stride;
                            xywh[2] = expf(xywh[2]) * one_head_stride;
                            xywh[3] = expf(xywh[3]) * one_head_stride;

                            id_list.emplace_back(cls_idx);
                            socre_list.emplace_back(realscore);
                            box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),
                                (xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
                        }
                    }
                }


            }

        }
    }

    std::vector<std::tuple<int, float, cv::Rect2f>> nms_res = nms_soft(id_list, socre_list, box_list, iou_thresh); // NMS

    std::vector<std::vector<float>> output_res = coordTrans(nms_res, img, false); // 测试精度时，check_border必须为false,避免将边界值优化

    //-------------STEP2: TRACK POST PROCESS ------------------//
    cv::Mat res_img = img.ori_img;
    obj_tracking(output_res, img, tracker, res_export, res_img, frame_id);

    //-------------VIS OR SAVE-----------------------//
    if (show) {
        cv::imshow("results", res_img);
        cv::waitKey(1);
    }
    if (save) {
        std::string save_path = resRoot + '/' + name+ ".txt";
        std::fstream outFile(save_path, std::ios::out | std::ios::trunc);
        for (size_t line = 0; line < res_export.size(); ++line) {
            for (size_t i = 0; i < 10; ++i) {
                outFile << res_export[line][i];
                if (i != 9) {
                    outFile << ",";
                }
            }
            outFile << "\n";
        }
        outFile.close();
    }

}
