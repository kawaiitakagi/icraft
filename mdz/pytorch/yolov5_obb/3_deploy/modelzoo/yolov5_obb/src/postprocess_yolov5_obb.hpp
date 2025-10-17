#pragma once
#include <opencv2/opencv.hpp>
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>

#include <modelzoo_utils.hpp>

#include "yolov5_obb_utils.hpp"

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
void post_process(std::vector<int> &angle_list, std::vector<int> &id_list, std::vector<float> &socre_list, std::vector<cv::Rect2f> &box_list, T* tensor_data, int obj_ptr_start,
    Grid grid, std::vector<float> norm, int stride,
    std::vector<float> anchor,int NOC, float CONF,bool MULTILABEL) {
    if (!MULTILABEL) {
        //auto i_s = this->getMaxRealScore(tensor_data, obj_ptr_start, norm);
        auto _score_ = sigmoid(tensor_data[obj_ptr_start + 4] * norm[0]);
        T* class_ptr_start = tensor_data + obj_ptr_start + 5;
        T* max_prob_ptr = std::max_element(class_ptr_start, class_ptr_start + NOC);
        int max_index = std::distance(class_ptr_start, max_prob_ptr);
        auto _prob_ = sigmoid(*max_prob_ptr * norm[0]);
        auto realscore = _score_ * _prob_;

        if (realscore > CONF) {
            //auto bbox = this->getBbox(tensor_data, norm, obj_ptr_start, location_x, location_y, stride, anchor);
            std::vector<float> xywh = sigmoid(tensor_data, norm[0], obj_ptr_start, 4);
            xywh[0] = (2 * xywh[0] + grid.location_x - 0.5) * stride;
            xywh[1] = (2 * xywh[1] + grid.location_y - 0.5) * stride;
            xywh[2] = 4 * powf(xywh[2], 2) * anchor[0];
            xywh[3] = 4 * powf(xywh[3], 2) * anchor[1];

            T* angle_ptr_start = tensor_data + obj_ptr_start + 5 + NOC;
            T* max_prob_ptr = std::max_element(angle_ptr_start, angle_ptr_start + 180);
            int max_index_angle = std::distance(angle_ptr_start, max_prob_ptr);
            angle_list.push_back(max_index_angle);

            id_list.emplace_back(max_index);
            socre_list.emplace_back(realscore);
            box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),
                (xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
        }
    }
    else {
        for (size_t i = 0; i < NOC; i++) {
            //auto realscore = this->getRealScore(tensor_data, obj_ptr_start, norm, i);

            auto _score_ = sigmoid(tensor_data[obj_ptr_start + 4] * norm[0]);
            auto _prob_ = sigmoid(tensor_data[obj_ptr_start + 5 + i] * norm[0]);
            auto realscore = _score_ * _prob_;
            if (realscore > CONF) {
                //auto bbox = this->getBbox(tensor_data, norm, obj_ptr_start, location_x, location_y, stride, anchor);
                std::vector<float> xywh = sigmoid(tensor_data, norm[0], obj_ptr_start, 4);
                xywh[0] = (2 * xywh[0] + grid.location_x - 0.5) * stride;
                xywh[1] = (2 * xywh[1] + grid.location_y - 0.5) * stride;
                xywh[2] = 4 * powf(xywh[2], 2) * anchor[0];
                xywh[3] = 4 * powf(xywh[3], 2) * anchor[1];

                T* angle_ptr_start = tensor_data + obj_ptr_start + 5 + NOC;
                T* max_prob_ptr = std::max_element(angle_ptr_start, angle_ptr_start + 180);
                int max_index_angle = std::distance(angle_ptr_start, max_prob_ptr);
                angle_list.push_back(max_index_angle);

                id_list.emplace_back(i);
                socre_list.emplace_back(realscore);
                box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),
                    (xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
            }
        }
    }

}

// socket_nms是为了匹配socket模式下nms输出输入的参数类型，其输出方便后续的coordTrans（bbox解码）
std::vector<std::tuple<int, float, cv::Rect2f, int>> obb_socket_nms(std::vector<int>& id_list, std::vector<float>& socre_list, std::vector<cv::Rect2f>& box_list, std::vector<int>& angle_list,float IOU, int max_nms = 3000) {
    std::vector<std::tuple<int, float, cv::Rect2f, int>> filter_res;
    std::vector<std::tuple<int, float, cv::Rect2f, int>> nms_res;
    auto bbox_num = id_list.size();
    for (size_t i = 0; i < bbox_num; i++)
    {
        filter_res.push_back({ id_list[i],socre_list[i],box_list[i],angle_list[i] });
    }

    std::stable_sort(filter_res.begin(), filter_res.end(),
        [](const std::tuple<int, float, cv::Rect2f, int>& tuple1, const std::tuple<int, float, cv::Rect2f, int>& tuple2) {
            return std::get<1>(tuple1) > std::get<1>(tuple2);
        }
    );

    int idx = 0;
    for (auto res : filter_res) {
        bool keep = true;
        for (int k = 0; k < nms_res.size() && keep; ++k) {
            if (std::get<0>(res) == std::get<0>(nms_res[k])) {
                if (1.f - jaccardDistance(std::get<2>(res), std::get<2>(nms_res[k])) > IOU) {
                    keep = false;
                }
            }

        }
        if (keep == true)
            nms_res.emplace_back(res);
        if (idx > max_nms) {
            break;
        }
        idx++;
    }
    return nms_res;
};

std::vector<std::vector<float>> coordTrans_obb(std::vector<std::tuple<int, float, cv::Rect2f, int>> &obb_nms_res,PicPre& img, bool check_border = true) {
    std::vector<std::vector<float>> output_data;
    int left_pad = img.getPad().first;
    int top_pad = img.getPad().second;
    float ratio = img.getRatio().first;
    
    for (auto&& res : obb_nms_res) {
        float class_id = std::get<0>(res);
        float score = std::get<1>(res);
        auto box = std::get<2>(res);
        float x1 = (box.tl().x - left_pad) / ratio;
        float y1 = (box.tl().y - top_pad) / ratio;
        float x2 = (box.br().x - left_pad) / ratio;
        float y2 = (box.br().y - top_pad) / ratio;
        x1 = checkBorder(x1, 0.f, (float)img.src_img.cols);
        y1 = checkBorder(y1, 0.f, (float)img.src_img.rows);
        x2 = checkBorder(x2, 0.f, (float)img.src_img.cols);
        y2 = checkBorder(y2, 0.f, (float)img.src_img.rows);
        float w = x2 - x1;
        float h = y2 - y1;

        float theta_pred_index = std::get<3>(res) - 90;
        float theta_pred = theta_pred_index / 180 * 3.141592;

        float x0 = x1 + w / 2;
        float y0 = y1 + h / 2;

        float Cos = cos(theta_pred);
        float Sin = sin(theta_pred);

        float w_cos = w / 2 * Cos;
        float w_sin = -w / 2 * Sin;
        float h_sin = -h / 2 * Sin;
        float h_cos = -h / 2 * Cos;

        float point_x1 = x0 + w_cos + h_sin;
        float point_y1 = y0 + w_sin + h_cos;
        float point_x2 = x0 + w_cos - h_sin;
        float point_y2 = y0 + w_sin - h_cos;
        float point_x3 = x0 - w_cos - h_sin;
        float point_y3 = y0 - w_sin - h_cos;
        float point_x4 = x0 - w_cos + h_sin;
        float point_y4 = y0 - w_sin + h_cos;
        //bbox：左上角点和wh
        output_data.emplace_back(std::vector<float>({ class_id, point_x1, point_y1, point_x2, point_y2,
                point_x3,point_y3,point_x4,point_y4,score }));
    }
    return output_data;
}
//--------------for pl
std::vector<std::vector<float>> coordTrans_plin(std::vector<std::tuple<int, float, cv::Rect2f, int>> &obb_nms_res,const cv::Mat& img,std::tuple<int, int, int, int >& ratio_bias,bool check_border = true) {
    std::vector<std::vector<float>> output_data;
    int left_pad = 0; //BIAS_W
    int top_pad = 0; //BIAS_H
    // float ratio = 1;//RATIO = 1
    float RATIO_W = std::get<0>(ratio_bias);
    float RATIO_H = std::get<1>(ratio_bias);
    float ratio = (std::min)(RATIO_W, RATIO_H);// resize mode = LONG_SIDE
    for (auto&& res : obb_nms_res) {
        float class_id = std::get<0>(res);
        float score = std::get<1>(res);
        auto box = std::get<2>(res);
        float x1 = (box.tl().x - left_pad) / ratio;
        float y1 = (box.tl().y - top_pad) / ratio;
        float x2 = (box.br().x - left_pad) / ratio;
        float y2 = (box.br().y - top_pad) / ratio;
        
        x1 = checkBorder(x1, 0.f, (float)2425);
        y1 = checkBorder(y1, 0.f, (float)1689);
        x2 = checkBorder(x2, 0.f, (float)2425);
        y2 = checkBorder(y2, 0.f, (float)1689);
        float w = x2 - x1;
        float h = y2 - y1;

        float theta_pred_index = std::get<3>(res) - 90;
        float theta_pred = theta_pred_index / 180 * 3.141592;

        float x0 = x1 + w / 2;
        float y0 = y1 + h / 2;

        float Cos = cos(theta_pred);
        float Sin = sin(theta_pred);

        float w_cos = w / 2 * Cos;
        float w_sin = -w / 2 * Sin;
        float h_sin = -h / 2 * Sin;
        float h_cos = -h / 2 * Cos;

        float point_x1 = x0 + w_cos + h_sin;
        float point_y1 = y0 + w_sin + h_cos;
        float point_x2 = x0 + w_cos - h_sin;
        float point_y2 = y0 + w_sin - h_cos;
        float point_x3 = x0 - w_cos - h_sin;
        float point_y3 = y0 - w_sin - h_cos;
        float point_x4 = x0 - w_cos + h_sin;
        float point_y4 = y0 - w_sin + h_cos;
        //bbox：左上角点和wh
        output_data.emplace_back(std::vector<float>({ class_id, point_x1, point_y1, point_x2, point_y2,
                point_x3,point_y3,point_x4,point_y4,score }));
    }
    return output_data;
}
void visualize_obb(std::vector<std::vector<float>> &output_data,const cv::Mat& img, const std::string resRoot, const std::string name,const std::vector<std::string>& names) {
    std::default_random_engine e;
    std::uniform_int_distribution<unsigned> u(10, 200);
    for (auto res : output_data) {
        int class_id = (int)res[0];
        float score = res[9];
        float point_x1 = res[1];
        float point_y1 = res[2];
        float point_x2 = res[3];
        float point_y2 = res[4];
        float point_x3 = res[5];
        float point_y3 = res[6];
        float point_x4 = res[7];
        float point_y4 = res[8];

        std::vector<std::vector<cv::Point>> point_list = { { cv::Point(point_x1,point_y1),cv::Point(point_x2,point_y2),
            cv::Point(point_x3,point_y3),cv::Point(point_x4,point_y4) } };
        cv::Scalar color_ = cv::Scalar(u(e), u(e), u(e));
        cv::drawContours(img, point_list, -1, color_, 2);
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << score;
        std::string s = std::to_string(class_id) + "_" + names[class_id] + " " + ss.str();
        auto s_size = cv::getTextSize(s, cv::FONT_HERSHEY_DUPLEX, 0.5, 1, 0);

        cv::rectangle(img, cv::Point2f(point_x1 - 1, point_y1 - s_size.height - 7), cv::Point2f(point_x1 + s_size.width, point_y1 - 2), color_, -1);
        cv::putText(img, s, cv::Point2f(point_x1, point_y1 - 2), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 255, 255), 0.2);
    }
    #ifdef _WIN32
        cv::imshow("results", img);
        cv::waitKey(0);
    #elif __linux__
        std::string save_path = resRoot + '/' + name;
        std::regex rgx("\\.(?!.*\\.)"); // 匹配最后一个点号（.）之前的位置，且该点号后面没有其他点号
        std::string RES_PATH = std::regex_replace(save_path, rgx, "_result.");
        cv::imwrite(RES_PATH, img);
    #endif

}

void post_detpost_hard(const std::vector<Tensor>& output_tensors, PicPre& img, NetInfo& netinfo,
    float conf, float iou_thresh, bool MULTILABEL, bool fpga_nms, int N_CLASS, 
    std::vector<std::vector<std::vector<float>>> &ANCHORS,std::vector<std::string>& LABELS, 
    bool & show, bool & save , std::string &resRoot, std::string & name,icraft::xrt::Device device,bool & run_sim,std::vector<std::vector<float>> & _norm,std::vector<float> & _stride) {
        std::vector<int> id_list;
        std::vector<float> socre_list;
        std::vector<cv::Rect2f> box_list;
        std::vector<int> angle_list = {};
		for (int head_index = 0; head_index < output_tensors.size(); head_index++) {
			auto host_tensor = output_tensors[head_index].to(HostDevice::MemRegion());
			int output_tensors_bits = output_tensors[head_index].dtype()->element_dtype.getStorageType().bits();
			int obj_num = output_tensors[head_index].dtype()->shape[2];
			int anchor_length = output_tensors[head_index].dtype()->shape[3];

			auto norm = _norm[head_index];
			auto stride = _stride[head_index];
			std::vector<float> _anchor_ = {};
			switch(output_tensors_bits){
				case 8:{
					auto tensor_data = (int8_t*)host_tensor.data().cptr();
					for(size_t obj = 0; obj < obj_num; obj++){
                        int base_addr = obj * anchor_length;
                        Grid grid = get_grid(output_tensors_bits, tensor_data, base_addr, anchor_length);
                        if (ANCHORS.size() != 0) {
                            _anchor_ = ANCHORS[head_index][grid.anchor_index];
                        }
						post_process(angle_list,id_list, socre_list, box_list, (int8_t*)tensor_data, base_addr, grid, norm, stride, _anchor_, N_CLASS, conf, MULTILABEL);
					}
					break;
				}
				//case 16
                case 16:{
                    auto tensor_data = (int16_t*)host_tensor.data().cptr();
                    for(size_t obj = 0; obj < obj_num; obj++){
                        int base_addr = obj * anchor_length;
                        Grid grid = get_grid(output_tensors_bits, tensor_data, base_addr, anchor_length);
                        if (ANCHORS.size() != 0) {
                            _anchor_ = ANCHORS[head_index][grid.anchor_index];
                        }
						post_process(angle_list,id_list, socre_list, box_list, (int16_t*)tensor_data, base_addr, grid, norm, stride, _anchor_, N_CLASS, conf, MULTILABEL);
					}
					break;
                }
				default: {
                	throw "wrong bits num!";
                	exit(EXIT_FAILURE);
            	}
			}
		}
		//后处理之NMS
        // std::cout << "number of results before nms = " << id_list.size() << '\n';
        std::vector<std::tuple<int, float, cv::Rect2f, int>> obb_nms_res ;
        // if (fpga_nms&& !run_sim) {
        //     nms_res = nms_hard(box_list, socre_list, id_list, iou_thresh, device);
        // }
        // else {
        //     nms_res = obb_socket_nms(id_list, socre_list, box_list, angle_list,iou_thresh);   

        // }
        
        obb_nms_res  = obb_socket_nms(id_list, socre_list, box_list, angle_list,iou_thresh);  
		std::vector<std::vector<float>> output_res = coordTrans_obb(obb_nms_res , img);
        
        #ifdef _WIN32
        if (show) {
            visualize_obb(output_res, img.ori_img, resRoot, name, LABELS);
        }
        #endif
        if (save) {
            #ifdef _WIN32
                saveRes(output_res, resRoot, name);
            #elif __linux__
                visualize_obb(output_res, img.ori_img, resRoot, name, LABELS);
            #endif
        }
}
void post_detpost_soft(const std::vector<Tensor>& output_tensors, PicPre& img, NetInfo& netinfo,
    float conf, float iou_thresh, bool MULTILABEL, bool fpga_nms, int N_CLASS, 
    std::vector<std::vector<std::vector<float>>> &ANCHORS,std::vector<std::string>& LABELS, 
    bool & show, bool & save , std::string &resRoot, std::string & name,icraft::xrt::Device device,bool & run_sim) {
    
    std::vector<int> id_list;
    std::vector<float> socre_list;
    std::vector<cv::Rect2f> box_list;
    std::vector<int> angle_list = {};
    std::vector<float> stride = get_stride(netinfo);
    
    //from (n,h,w,603)-> (n,h,w,201),(n,h,w,201),(n,h,w,201) 3 anchors
    for (int yolo = 0; yolo < output_tensors.size(); yolo++) {
        int _H = output_tensors[yolo].dtype()->shape[1];
        int _W = output_tensors[yolo].dtype()->shape[2];

        auto host_tensor = output_tensors[yolo].to(HostDevice::MemRegion());
        auto tensor_data = (float*)host_tensor.data().cptr();
        
        for (size_t h = 0; h < _H; h++) {
            int _h = h;
            for (size_t w = 0; w < _W; w++) {
                int _w = w;
                for (size_t anchor_index = 0; anchor_index < ANCHORS.size(); anchor_index++) {
                    int _anchor_id = anchor_index;
                    auto one_head_stride = stride[yolo];
                    std::vector<float> one_head_anchor = {};
                    if (ANCHORS.size() != 0) {
                        one_head_anchor = ANCHORS[yolo][anchor_index];
                    }
                    int anchor_length = 201;
                    auto boxPtr = tensor_data + h * _W * (anchor_length) * ANCHORS.size()+ w * (anchor_length) * ANCHORS.size() + anchor_index * (anchor_length);
                    auto scorePtr = boxPtr + 4;
                    auto _score_ = sigmoid(*scorePtr);
                    auto classPtr = boxPtr + 5;
                    float* max_prob_ptr = std::max_element(classPtr, classPtr + N_CLASS);
                    int max_index = std::distance(classPtr, max_prob_ptr);
                    auto _prob_ = sigmoid(*max_prob_ptr);
                    auto realscore = _score_ * _prob_;
                    
                    if(realscore > conf){
                        //cal angle
                        auto angle_ptr_start = boxPtr + 21;
                        float* angle_max_prob_ptr = std::max_element(angle_ptr_start, angle_ptr_start + 180);
                        int max_index_angle = std::distance(angle_ptr_start, angle_max_prob_ptr);
                        angle_list.push_back(max_index_angle);

                        auto xywh = sigmoid(boxPtr, 4);
                        xywh[0] = (2 * xywh[0] + w - 0.5) * one_head_stride;
                        xywh[1] = (2 * xywh[1] + h - 0.5) * one_head_stride;
                        xywh[2] = 4 * powf(xywh[2], 2) * one_head_anchor[0];
                        xywh[3] = 4 * powf(xywh[3], 2) * one_head_anchor[1];

                        id_list.emplace_back(max_index);
                        socre_list.emplace_back(realscore);
                        box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),
                            (xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
                    }
                }
            }
        }
    }
    // std::cout << "number of results before nms = " << id_list.size() << '\n';
    //后处理之NMS
    std::vector<std::tuple<int, float, cv::Rect2f, int>> obb_nms_res ;
    obb_nms_res  = obb_socket_nms(id_list, socre_list, box_list, angle_list,iou_thresh); 
    std::vector<std::vector<float>> output_res = coordTrans_obb(obb_nms_res , img);
    #ifdef _WIN32
    if (show) {
        visualize_obb(output_res, img.ori_img, resRoot, name, LABELS);
    }
    #endif
    if (save) {
        #ifdef _WIN32
            saveRes(output_res, resRoot, name);
        #elif __linux__
            visualize_obb(output_res, img.ori_img, resRoot, name, LABELS);
        #endif
    }
}

// smooth
const float ALPHA = 0.5f;
const float SMOOTH_IOU = 0.80f;

using YoloPostResult = std::tuple<std::vector<int>, std::vector<float>, std::vector<cv::Rect2f>,std::vector<std::vector<float>>  >; // box_list, id_list, score_list,point_list

YoloPostResult post_detpost_plin(const std::vector<Tensor>& output_tensors, YoloPostResult& last_frame_result,NetInfo& netinfo,
    float conf, float iou_thresh, bool MULTILABEL, bool fpga_nms, int N_CLASS,
    std::vector<std::vector<std::vector<float>>>& ANCHORS, icraft::xrt::Device device,
    bool & run_sim,std::vector<std::vector<float>> & _norm,std::vector<int> & real_out_channles,std::vector<float> & _stride,int bbox_info_channel,const cv::Mat& img,std::vector<std::string>& LABELS,std::tuple<int, int, int, int >& ratio_bias) {

    std::vector<int> id_list;
    std::vector<float> socre_list;
    std::vector<cv::Rect2f> box_list;
    std::vector<float> stride = get_stride(netinfo);
    std::vector<int> angle_list = {};
    for (size_t i = 0; i < output_tensors.size(); i++) {

        auto host_tensor = output_tensors[i].to(HostDevice::MemRegion());
        int output_tensors_bits = output_tensors[i].dtype()->element_dtype.getStorageType().bits();
        int obj_num = output_tensors[i].dtype()->shape[2];
        int anchor_length = output_tensors[i].dtype()->shape[3];

        auto norm = _norm[i];
		auto stride = _stride[i];
		std::vector<float> _anchor_ = {};
        switch(output_tensors_bits){
            case 8:{
                auto tensor_data = (int8_t*)host_tensor.data().cptr();
                for (size_t obj = 0; obj < obj_num; obj++) {
                    int base_addr = obj * anchor_length;
                    Grid grid = get_grid(output_tensors_bits, tensor_data, base_addr, anchor_length);
                    if (ANCHORS.size() != 0) {
                                _anchor_ = ANCHORS[i][grid.anchor_index];
                            }
                    post_process(angle_list,id_list, socre_list, box_list, (int8_t*)tensor_data, base_addr, grid, norm, stride, _anchor_, N_CLASS, conf, MULTILABEL);
                }
                break;
            }
            case 16:{
                auto tensor_data = (int16_t*)host_tensor.data().cptr();
                for (size_t obj = 0; obj < obj_num; obj++) {
                    int base_addr = obj * anchor_length;
                    Grid grid = get_grid(output_tensors_bits, tensor_data, base_addr, anchor_length);
                    if (ANCHORS.size() != 0) {
                            _anchor_ = ANCHORS[i][grid.anchor_index];
                    }
                    post_process(angle_list,id_list, socre_list, box_list, (int16_t*)tensor_data, base_addr, grid, norm, stride, _anchor_, N_CLASS, conf, MULTILABEL);
                }
                break;
            }
            default:{
                throw "wrong bits num!";
                exit(EXIT_FAILURE);
            }
        }
    }
    //后处理之NMS
    // std::cout << "number of results before nms = " << id_list.size() << '\n';
    std::vector<std::tuple<int, float, cv::Rect2f, int>> obb_nms_res ;
    obb_nms_res  = obb_socket_nms(id_list, socre_list, box_list, angle_list,iou_thresh);  
	std::vector<std::vector<float>> output_res = coordTrans_plin(obb_nms_res,img,ratio_bias,true);
    return YoloPostResult { id_list, socre_list, box_list,output_res};

}