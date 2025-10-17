#pragma once
#include <opencv2/opencv.hpp>
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>
#include <modelzoo_utils.hpp>

#include "yolov6_utils.hpp"

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
void post_process(std::vector<int> &id_list, std::vector<float> &socre_list, std::vector<cv::Rect2f> &box_list, T* tensor_data, int obj_ptr_start,
    Grid& grid, std::vector<int>& real_out_channles,int bbox_info_channel, std::vector<float> norm, int stride,
    std::vector<float> anchor,int NOC, float CONF,bool MULTILABEL) {
    if (!MULTILABEL) {
        //getMaxRealScore;
        T* class_ptr_start = tensor_data + obj_ptr_start;
        T* max_prob_ptr = std::max_element(class_ptr_start, class_ptr_start + NOC);
        int max_index = std::distance(class_ptr_start, max_prob_ptr);
        auto _prob_ = sigmoid(*max_prob_ptr * norm[0]);
        auto realscore =  _prob_;
        if (realscore > CONF) {
            float _x = tensor_data[obj_ptr_start+real_out_channles[0]] * norm[1];
            float _y = tensor_data[obj_ptr_start+real_out_channles[0] + 1] * norm[1];
            float _w = tensor_data[obj_ptr_start+real_out_channles[0] + 2] * norm[1];
            float _h = tensor_data[obj_ptr_start+real_out_channles[0] + 3] * norm[1];
            float x = ((_w - _x) * 0.5 + grid.location_x + 0.5) * stride ;
            float y = ((_h - _y) * 0.5 + grid.location_y + 0.5) * stride ;
            float w = (_x + _w) * stride ;
            float h = (_y +_h) * stride ;

            id_list.emplace_back(max_index);
            socre_list.emplace_back(realscore);
            box_list.emplace_back(cv::Rect2f((x - w / 2), (y - h / 2), w, h));
        }
    }
    else 
    {
    for (size_t i = 0; i < NOC; i++) {
        //getRealScore
        auto _prob_ = sigmoid(tensor_data[obj_ptr_start + i] * norm[0]);
        auto realscore =_prob_;
        if (realscore > CONF) {
            //getBbox
            float _x = tensor_data[obj_ptr_start+ real_out_channles[0]] * norm[1];
            float _y = tensor_data[obj_ptr_start+ real_out_channles[0] + 1] * norm[1];
            float _w = tensor_data[obj_ptr_start+ real_out_channles[0] + 2] * norm[1];
            float _h = tensor_data[obj_ptr_start+ real_out_channles[0] + 3] * norm[1];
            float x = ((_w - _x) * 0.5 + grid.location_x + 0.5) * stride ;
            float y = ((_h - _y) * 0.5 + grid.location_y + 0.5) * stride ;
            float w = (_x + _w) * stride ;
            float h = (_y +_h) * stride ;
            id_list.emplace_back(i);
            socre_list.emplace_back(realscore);
            box_list.emplace_back(cv::Rect2f((x - w / 2),
                (y - h / 2), w, h));
            }
        }
    }
}

void post_detpost_hard(const std::vector<Tensor>& output_tensors, PicPre& img, NetInfo& netinfo,
    float conf, float iou_thresh, bool MULTILABEL, bool fpga_nms, int N_CLASS, 
    std::vector<std::vector<std::vector<float>>> &ANCHORS,std::vector<std::string>& LABELS, 
    bool & show, bool & save , std::string &resRoot, std::string & name,icraft::xrt::Device device,bool & run_sim,std::vector<std::vector<float>> & _norm,std::vector<int> & real_out_channles,std::vector<float> & _stride,int bbox_info_channel) {
        std::vector<int> id_list;
        std::vector<float> socre_list;
        std::vector<cv::Rect2f> box_list;
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
						post_process(id_list, socre_list, box_list, (int8_t*)tensor_data, base_addr, grid, real_out_channles, bbox_info_channel, norm, stride, _anchor_, N_CLASS, conf, MULTILABEL);
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
						post_process(id_list, socre_list, box_list, (int16_t*)tensor_data, base_addr, grid, real_out_channles, bbox_info_channel, norm, stride, _anchor_, N_CLASS, conf, MULTILABEL);
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
        std::vector<std::tuple<int, float, cv::Rect2f>> nms_res;
        if (fpga_nms&& !run_sim) {
            nms_res = nms_hard(box_list, socre_list, id_list, iou_thresh, device);
        }
        else {
            nms_res = nms_soft(id_list, socre_list, box_list, iou_thresh);   

        }
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
    std::vector<std::vector<std::vector<float>>>& ANCHORS, NetInfo& netinfo, int N_CLASS, float conf, float iou_thresh,bool& fpga_nms, icraft::xrt::Device device, bool& run_sim,
    bool& MULTILABEL,bool& show, bool& save, std::string& resRoot, std::string& name) {

    std::vector<int> id_list;
    std::vector<float> socre_list;
    std::vector<cv::Rect2f> box_list;
    std::vector<float> stride = get_stride(netinfo);
    for (int yolo = 0; yolo < output_tensors.size()-1; yolo = yolo + 2) {
        int _H = output_tensors[yolo].dtype()->shape[1];
        int _W = output_tensors[yolo].dtype()->shape[2];
        int _C = output_tensors[yolo].dtype()->shape[3];
        auto host_tensor_class = output_tensors[yolo+0].to(HostDevice::MemRegion());//c = 80
        auto host_tensor_box = output_tensors[yolo+1].to(HostDevice::MemRegion());//c = 4
        auto tensor_data_class = (float*)host_tensor_class.data().cptr();
        auto tensor_data_box = (float*)host_tensor_box.data().cptr();
        for (size_t h = 0; h < _H; h++) {
            for (size_t w = 0; w < _W; w++) {
                auto one_head_stride = stride[yolo];
                auto classPtr = tensor_data_class + h * _W * N_CLASS + w * N_CLASS;
                auto boxPtr = tensor_data_box + h * _W * 4 + w * 4;
                if (!MULTILABEL) {
                    float* max_prob_ptr = std::max_element(classPtr, classPtr + N_CLASS);
                    int max_index = std::distance(classPtr, max_prob_ptr);
                    auto _prob_ = sigmoid(*max_prob_ptr);
                    auto realscore = _prob_;
                    
                    if (realscore > conf) {
                        //getBbox
                        float x1 = boxPtr[0];
                        float y1 = boxPtr[1];
                        float x2 = boxPtr[2];
                        float y2 = boxPtr[3];

                        float x_ = (w + 0.5 + ((x2 - x1) / 2.f)) * one_head_stride;
                        float y_ = (h + 0.5 + ((y2 - y1) / 2.f)) * one_head_stride;
                        float w_ = (x1 + x2) * one_head_stride;
                        float h_ = (y1 + y2) * one_head_stride;
                        // 调试用
                        // std::cout<<" x1 ="<<x1<<" y1 ="<<y1<<" x2 ="<<x2<<" y2 ="<<y2<<std::endl;
                        // std::cout<<" x ="<<x_<<" y ="<<y_<<" w ="<<w_<<" h ="<<h_<<std::endl;
                        std::vector<float> xywh = { x_,y_,w_,h_ };
                        id_list.emplace_back(max_index);
                        socre_list.emplace_back(realscore);
                        box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),(xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
                        
                    }
                }
                else{
                    for (size_t cls_idx = 0; cls_idx < N_CLASS; cls_idx++) {
                        auto _prob_ = sigmoid(*(classPtr + cls_idx));
                        auto realscore =  _prob_;
                        if (realscore > conf) {
                            
                            //getBbox
                            float x1 = boxPtr[0];
                            float y1 = boxPtr[1];
                            float x2 = boxPtr[2];
                            float y2 = boxPtr[3];

                            float x_ = (w + 0.5 + ((x2 - x1) / 2.f)) * one_head_stride;
                            float y_ = (h + 0.5 + ((y2 - y1) / 2.f)) * one_head_stride;
                            float w_ = (x1 + x2) * one_head_stride;
                            float h_ = (y1 + y2) * one_head_stride;
                            std::vector<float> xywh = { x_,y_,w_,h_ };

                            id_list.emplace_back(cls_idx);
                            socre_list.emplace_back(realscore);
                            box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),(xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
                            
                        }
                    }
                }
            }
        }
    }
    // std::cout << "number of results before nms = " << id_list.size() << '\n';
    //后处理之NMS
    std::vector<std::tuple<int, float, cv::Rect2f>> nms_res;
    //auto post_start = std::chrono::system_clock::now();
    
    if (fpga_nms&& !run_sim) {
        nms_res = nms_hard(box_list, socre_list, id_list, iou_thresh, device);
    }
    else {
        nms_res = nms_soft(id_list, socre_list, box_list, iou_thresh);   
    }
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