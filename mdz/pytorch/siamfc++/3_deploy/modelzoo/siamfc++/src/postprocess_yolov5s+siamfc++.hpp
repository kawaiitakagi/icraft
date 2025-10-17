#pragma once
#include <opencv2/opencv.hpp>
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>
#include <modelzoo_utils.hpp>
#define PI 3.1415926

using namespace icraft::xrt;

#define BIT 8
#if BIT == 8
#define TENSOR_DATA_TYPE int8_t
#define MAXC 64
#define MINC 8
#define ANCHOR_LENGTH_DIVISOR 1
#elif BIT == 16
#define TENSOR_DATA_TYPE int16_t
#define MAXC 32
#define MINC 4
#define ANCHOR_LENGTH_DIVISOR 2
#endif

//-------------------------------------//
//       加载yolov5_hand网络参数
//-------------------------------------//
std::vector<float> STRIDE = { 8, 16, 32 };
std::vector<std::vector<std::vector<float>>> ANCHORS = { { { 10, 13}, { 16,  30}, { 33,  23} } ,
                                                         { { 30, 61}, { 62,  45}, { 59, 119} } ,
                                                         { {116, 90}, {156, 198}, {373, 326} } };


cv::Mat CreatHannWindow(int width, int height)
{
    cv::Mat vertical(height, 1, CV_32FC1);
    cv::Mat horizontal(1, width, CV_32FC1);
    for (int r = 0; r < height; r++)
    {
        vertical.at<float>(r, 0) = 0.5 - 0.5 * cos(2 * PI * r / (height - 1));
    }
    for (int c = 0; c < width; c++)
    {
        horizontal.at<float>(0, c) = 0.5 - 0.5 * cos(2 * PI * c / (width - 1));
    }
    cv::Mat out = vertical * horizontal;
    return out;
}


struct Grid {
	uint16_t location_x = 0;
	uint16_t location_y = 0;
	uint16_t anchor_index = 0;
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

std::vector<float> get_stride(NetInfo& netinfo) {
    std::vector<float> stride;
    for (auto i : netinfo.o_cubic) {
		stride.emplace_back(netinfo.head_hardop_i_shape_cubic[0].h / i.h);
		std::cout << "stride:" << std::endl;
		std::cout << netinfo.head_hardop_i_shape_cubic[0].h / i.h << std::endl;
    }
    return stride;
};

template <typename T>
void get_cls_bbox(std::vector<int>& id_list, std::vector<float>& socre_list, std::vector<cv::Rect2f>& box_list, T* tensor_data, int base_addr,
	Grid& grid, float& SCALE, int stride,
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
		for (size_t cls_idx = N_CLASS-1; cls_idx < N_CLASS; cls_idx++) {
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


std::tuple<bool, cv::Rect> yolov5_post_detpost_plin(const std::vector<Tensor>& output_tensors, NetInfo& netinfo,
	float conf, float iou_thresh, bool MULTILABEL, int N_CLASS, icraft::xrt::Device device) {

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
	std::vector<std::tuple<int, float, cv::Rect2f>> nms_res = nms_soft(id_list, socre_list, box_list, iou_thresh);   // 后处理 之 NMS
	std::vector<int> id_list_ret;   
    std::vector<float> score_list_ret; 
    std::vector<cv::Rect2f> box_list_ret; 
	for (auto idx_score_bbox : nms_res) {
        // store 
        id_list_ret.emplace_back(std::get<0>(idx_score_bbox));
        score_list_ret.emplace_back(std::get<1>(idx_score_bbox));
        box_list_ret.emplace_back(std::get<2>(idx_score_bbox));
    }
	//std::cout<<id_list_ret.size()<<std::endl;

	//选择score最大的目标作为追踪目标
	int max_score = 0;
	int biggest_index = -1;
	for (int index : id_list_ret){
		float score = socre_list[index];
		if (score > max_score) {
			max_score = score;
			biggest_index = index;
		}
		
	}
	//选择区域最大的目标作为追踪目标
	//float max_area = FLT_MIN;
	// for (int index : id_list_ret){
	// 	float w = box_list[index].width;
	// 	float h = box_list[index].height;
	// 	if ((w * h) > max_area) {
	// 		max_area = w * h;
	// 		biggest_index = index;
	// 	}
	// }

	return biggest_index == -1 ? std::tuple<bool, cv::Rect>{false, cv::Rect(0, 0, 0, 0)} : std::tuple<bool, cv::Rect>{ true, box_list[biggest_index] };

}

// 对DetPost得到的检测结果进行nms后处理
std::vector<int> nms(std::vector<cv::Rect>& box_list, std::vector<float>& socre_list, const float& conf, const float& iou) {

	std::vector<int> nms_indices;
	std::vector<std::pair<float, int> > score_index_vec;

	for (size_t i = 0; i < socre_list.size(); ++i) {
		if (socre_list[i] > conf) {
			score_index_vec.emplace_back(std::make_pair(socre_list[i], i));
		}
	}

	std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
		[](const std::pair<float, int>& pair1, const std::pair<float, int>& pair2) {return pair1.first > pair2.first; });

	for (size_t i = 0; i < score_index_vec.size(); ++i) {
		const int idx = score_index_vec[i].second;
		bool keep = true;
		for (int k = 0; k < nms_indices.size() && keep; ++k) {
			if (1.f - jaccardDistance(box_list[idx], box_list[nms_indices[k]]) > iou) {
				keep = false;
			}
		}
		if (keep == true)
			nms_indices.emplace_back(idx);
	}

	return nms_indices;
}

// siamfc++模型前处理
void siamfc_preprocess(std::vector<float> target_pos, std::vector<float> target_sz, std::vector<std::vector<float>>& M_inversed, float& scale, float context_amount, int z_size, int t_size) {
	float wc = target_sz[0] + context_amount * (target_sz[0] + target_sz[1]);
	float hc = target_sz[1] + context_amount * (target_sz[0] + target_sz[1]);
	float s_crop = sqrt(wc * hc);
	scale = z_size / s_crop;
	s_crop = t_size / scale;

	std::vector<float> crop_cxywh = { target_pos[0],target_pos[1], round(s_crop), round(s_crop) };

	float crop_xyxy_0 = crop_cxywh[0] - (crop_cxywh[2] - 1) / 2;
	float crop_xyxy_1 = crop_cxywh[1] - (crop_cxywh[3] - 1) / 2;
	float crop_xyxy_2 = crop_cxywh[0] + (crop_cxywh[2] - 1) / 2;
	float crop_xyxy_3 = crop_cxywh[1] + (crop_cxywh[3] - 1) / 2;

	float M_11 = (crop_xyxy_2 - crop_xyxy_0) / (t_size - 1);
	float M_22 = (crop_xyxy_3 - crop_xyxy_1) / (t_size - 1);
	//update
	M_inversed[0][0] = M_11;
	M_inversed[0][2] = crop_xyxy_0;
	M_inversed[1][1] = M_22;
	M_inversed[1][2] = crop_xyxy_1;
}

// siamfc++_net2模型后处理，使用cast算子
void net2_postprocess_withcast(const std::vector<Tensor>& output_tensors, std::vector<float>& target_pos, std::vector<float>& target_sz, cv::Mat xy_ctr,
	 cv::Mat window, float window_influence, float x_size, float scale, int im_w, int im_h) {

	auto net2_data_ptr_0 = (float*)output_tensors[0].data().cptr();
	auto net2_data_ptr_1 = (float*)output_tensors[1].data().cptr();
	auto net2_data_ptr_2 = (float*)output_tensors[2].data().cptr();

	cv::Mat net_result_mat = cv::Mat(289, 4, CV_32F, net2_data_ptr_0);
	net_result_mat = (net_result_mat * 0.78319091 + 1.60675728);
	cv::exp(net_result_mat, net_result_mat);
	net_result_mat = net_result_mat * 8;

	cv::Mat mat_1 = net_result_mat(cv::Rect(0, 0, 2, net_result_mat.rows));
	cv::Mat mat_2 = net_result_mat(cv::Rect(2, 0, 2, net_result_mat.rows));
	cv::Mat mat_3 = xy_ctr - mat_1;
	cv::Mat mat_4 = xy_ctr + mat_2;

	cv::Mat box;
	cv::hconcat(mat_3, mat_4, box);

	cv::Mat cls_score = cv::Mat(289, 1, CV_32F, net2_data_ptr_1);
	cv::Mat ctr_score = cv::Mat(289, 1, CV_32F, net2_data_ptr_2);

	for (int i = 0; i < cls_score.rows; i++) {
		for (int j = 0; j < cls_score.cols; j++) {
			float value = cls_score.at<float>(i, j);
			value = 1.0 / (1.0 + exp(-value));
			cls_score.at<float>(i, j) = value;
		}
	}

	for (int i = 0; i < ctr_score.rows; i++) {
		for (int j = 0; j < ctr_score.cols; j++) {
			float value = ctr_score.at<float>(i, j);
			value = 1.0 / (1.0 + exp(-value));
			ctr_score.at<float>(i, j) = value;
		}
	}
	// score = cls*ctr
	cv::Mat score;
	cv::multiply(cls_score, ctr_score, score);

	// box：xyxy2cxywh
	for (int i = 0; i < box.rows; i++) {
		float value_0 = box.at<float>(i, 0);
		float value_1 = box.at<float>(i, 1);
		float value_2 = box.at<float>(i, 2);
		float value_3 = box.at<float>(i, 3);
		box.at<float>(i, 0) = (value_0 + value_2) / 2;
		box.at<float>(i, 1) = (value_1 + value_3) / 2;
		box.at<float>(i, 2) = value_2 - value_0 + 1;
		box.at<float>(i, 3) = value_3 - value_1 + 1;
	}

	// score post-processing
	float penalty_k = 0.08;
	std::vector<float> target_sz_in_crop = { target_sz[0] * scale, target_sz[1] * scale };

	// box_wh[:, 2] box_wh[:, 3]
	cv::Mat box_wh_2 = box(cv::Rect(2, 0, 1, box.rows));
	cv::Mat box_wh_3 = box(cv::Rect(3, 0, 1, box.rows));

	// sz 
	cv::Mat pad_1 = (box_wh_2 + box_wh_3) * 0.5;
	cv::Mat sz2_1;
	cv::multiply((box_wh_2 + pad_1), (box_wh_3 + pad_1), sz2_1);
	cv::sqrt(sz2_1, sz2_1);

	// sz_wh 
	float pad_2 = (target_sz_in_crop[0] + target_sz_in_crop[1]) * 0.5;
	float sz2_2 = (target_sz_in_crop[0] + pad_2) * ((target_sz_in_crop[1] + pad_2));
	sz2_2 = sqrt(sz2_2);
	sz2_1 = sz2_1 / sz2_2;

	// change 
	for (int i = 0; i < sz2_1.rows; i++) {
		for (int j = 0; j < sz2_1.cols; j++) {
			float value = sz2_1.at<float>(i, j);
			sz2_1.at<float>(i, j) = value > (1 / value) ? value : (1 / value);
		}
	}

	float data_1 = target_sz_in_crop[0] / target_sz_in_crop[1];
	cv::Mat r_c = data_1 / (box_wh_2 / box_wh_3);

	// change
	for (int i = 0; i < r_c.rows; i++) {
		for (int j = 0; j < r_c.cols; j++) {
			float value = r_c.at<float>(i, j);
			r_c.at<float>(i, j) = value > (1 / value) ? value : (1 / value);
		}
	}

	cv::Mat penalty;
	cv::multiply(r_c, sz2_1, penalty);
	penalty = (penalty - 1) * (-penalty_k);
	cv::exp(penalty, penalty);

	// pscore = penalty * score
	cv::Mat pscore;
	cv::multiply(penalty, score, pscore);

	pscore = pscore * (1 - window_influence) + window * window_influence;
	cv::Point maxLoc;
	cv::minMaxLoc(pscore, NULL, NULL, NULL, &maxLoc);

	int best_pscore_id = maxLoc.y;

	// box post-processing
	cv::Mat pred_in_crop = box(cv::Rect(0, best_pscore_id, 4, 1)) / scale;
	float test_lr = 0.58;
	float lr = penalty.at<float>(best_pscore_id, 0) * score.at<float>(best_pscore_id, 0) * test_lr;
	float res_x = pred_in_crop.at<float>(0, 0) + target_pos[0] - (int(x_size) / 2) / scale;
	float res_y = pred_in_crop.at<float>(0, 1) + target_pos[1] - (int(x_size) / 2) / scale;
	float res_w = target_sz[0] * (1 - lr) + pred_in_crop.at<float>(0, 2) * lr;
	float res_h = target_sz[1] * (1 - lr) + pred_in_crop.at<float>(0, 3) * lr;

	// restrict new_target_pos & new_target_sz
	//int x = i > j ? i : j; // max(i,j)
	//int y = i < j ? i : j; // min(i,j)

	float min_1 = im_w < res_x ? im_w : res_x;
	target_pos[0] = 0 > min_1 ? 0 : min_1;
	float min_2 = im_h < res_y ? im_h : res_y;
	target_pos[1] = 0 > min_2 ? 0 : min_2;
	float min_3 = im_w < res_w ? im_w : res_w;
	target_sz[0] = 10 > min_3 ? 10 : min_3;
	float min_4 = im_h < res_h ? im_h : res_h;
	target_sz[1] = 10 > min_4 ? 10 : min_4;

}

// siamfc++_net2模型后处理，去除cast算子
void net2_postprocess(const std::vector<Tensor>& output_tensors, std::vector<float> net2_output_normratio, std::vector<float>& target_pos, std::vector<float>& target_sz, cv::Mat xy_ctr,
	cv::Mat window, float window_influence, float x_size, float scale,int im_w,int im_h) {

	// 去除cast算子，手动将输出的tensor 从 pl_ddr 搬移到 ps_ddr
	auto host_tensor_0 = output_tensors[0].to(HostDevice::MemRegion());
	auto net2_data_ptr_0 = (int8_t*)host_tensor_0.data().cptr();

	auto host_tensor_1 = output_tensors[1].to(HostDevice::MemRegion());
	auto net2_data_ptr_1 = (int8_t*)host_tensor_1.data().cptr();

	auto host_tensor_2 = output_tensors[2].to(HostDevice::MemRegion());
	auto net2_data_ptr_2 = (int8_t*)host_tensor_2.data().cptr();


	cv::Mat net_result_mat = cv::Mat(289, 4, CV_8S, net2_data_ptr_2);
	net_result_mat.convertTo(net_result_mat, CV_32F);
	net_result_mat = net_result_mat * net2_output_normratio[2];//反量化

	net_result_mat = (net_result_mat * 0.78319091 + 1.60675728);
	cv::exp(net_result_mat, net_result_mat);
	net_result_mat = net_result_mat * 8;

	cv::Mat mat_1 = net_result_mat(cv::Rect(0, 0, 2, net_result_mat.rows));
	cv::Mat mat_2 = net_result_mat(cv::Rect(2, 0, 2, net_result_mat.rows));
	cv::Mat mat_3 = xy_ctr - mat_1;
	cv::Mat mat_4 = xy_ctr + mat_2;

	cv::Mat box;
	cv::hconcat(mat_3, mat_4, box);

	cv::Mat cls_score = cv::Mat(289, 1, CV_8S, net2_data_ptr_0);
	cls_score.convertTo(cls_score, CV_32F);
	cls_score = cls_score * net2_output_normratio[0];//反量化

	cv::Mat ctr_score = cv::Mat(289, 1, CV_8S, net2_data_ptr_1);
	ctr_score.convertTo(ctr_score, CV_32F);
	ctr_score = ctr_score * net2_output_normratio[1];//反量化

	for (int i = 0; i < cls_score.rows; i++) {
		for (int j = 0; j < cls_score.cols; j++) {
			float value = cls_score.at<float>(i, j);
			value = 1.0 / (1.0 + exp(-value));
			cls_score.at<float>(i, j) = value;
		}
	}

	for (int i = 0; i < ctr_score.rows; i++) {
		for (int j = 0; j < ctr_score.cols; j++) {
			float value = ctr_score.at<float>(i, j);
			value = 1.0 / (1.0 + exp(-value));
			ctr_score.at<float>(i, j) = value;
		}
	}
	// score = cls*ctr
	cv::Mat score;
	cv::multiply(cls_score, ctr_score, score);

	// box：xyxy2cxywh
	for (int i = 0; i < box.rows; i++) {
		float value_0 = box.at<float>(i, 0);
		float value_1 = box.at<float>(i, 1);
		float value_2 = box.at<float>(i, 2);
		float value_3 = box.at<float>(i, 3);
		box.at<float>(i, 0) = (value_0 + value_2) / 2;
		box.at<float>(i, 1) = (value_1 + value_3) / 2;
		box.at<float>(i, 2) = value_2 - value_0 + 1;
		box.at<float>(i, 3) = value_3 - value_1 + 1;
	}

	// score post-processing
	float penalty_k = 0.08;
	std::vector<float> target_sz_in_crop = { target_sz[0] * scale, target_sz[1] * scale };

	// box_wh[:, 2] box_wh[:, 3]
	cv::Mat box_wh_2 = box(cv::Rect(2, 0, 1, box.rows));
	cv::Mat box_wh_3 = box(cv::Rect(3, 0, 1, box.rows));

	// sz 
	cv::Mat pad_1 = (box_wh_2 + box_wh_3) * 0.5;
	cv::Mat sz2_1;
	cv::multiply((box_wh_2 + pad_1), (box_wh_3 + pad_1), sz2_1);
	cv::sqrt(sz2_1, sz2_1);

	// sz_wh 
	float pad_2 = (target_sz_in_crop[0] + target_sz_in_crop[1]) * 0.5;
	float sz2_2 = (target_sz_in_crop[0] + pad_2) * ((target_sz_in_crop[1] + pad_2));
	sz2_2 = sqrt(sz2_2);
	sz2_1 = sz2_1 / sz2_2;

	// change 
	for (int i = 0; i < sz2_1.rows; i++) {
		for (int j = 0; j < sz2_1.cols; j++) {
			float value = sz2_1.at<float>(i, j);
			sz2_1.at<float>(i, j) = value > (1 / value) ? value : (1 / value);
		}
	}

	float data_1 = target_sz_in_crop[0] / target_sz_in_crop[1];
	cv::Mat r_c = data_1 / (box_wh_2 / box_wh_3);

	// change
	for (int i = 0; i < r_c.rows; i++) {
		for (int j = 0; j < r_c.cols; j++) {
			float value = r_c.at<float>(i, j);
			r_c.at<float>(i, j) = value > (1 / value) ? value : (1 / value);
		}
	}

	cv::Mat penalty;
	cv::multiply(r_c, sz2_1, penalty);
	penalty = (penalty - 1) * (-penalty_k);
	cv::exp(penalty, penalty);

	// pscore = penalty * score
	cv::Mat pscore;
	cv::multiply(penalty, score, pscore);

	pscore = pscore * (1 - window_influence) + window * window_influence;
	cv::Point maxLoc;
	cv::minMaxLoc(pscore, NULL, NULL, NULL, &maxLoc);

	int best_pscore_id = maxLoc.y;

	// box post-processing
	cv::Mat pred_in_crop = box(cv::Rect(0, best_pscore_id, 4, 1)) / scale;
	float test_lr = 0.58;
	float lr = penalty.at<float>(best_pscore_id, 0) * score.at<float>(best_pscore_id, 0) * test_lr;
	// float test = pred_in_crop.at<float>(0, 0)- (int(x_size) / 2) / scale;
	// std::cout<<"test"<<test<<std::endl;
	float res_x = pred_in_crop.at<float>(0, 0) + target_pos[0] - (int(x_size) / 2) / scale;
	float res_y = pred_in_crop.at<float>(0, 1) + target_pos[1] - (int(x_size) / 2) / scale;
	float res_w = target_sz[0] * (1 - lr) + pred_in_crop.at<float>(0, 2) * lr;
	float res_h = target_sz[1] * (1 - lr) + pred_in_crop.at<float>(0, 3) * lr;

	// restrict new_target_pos & new_target_sz
	//int x = i > j ? i : j; // max(i,j)
	//int y = i < j ? i : j; // min(i,j)

	float min_1 = im_w < res_x ? im_w : res_x;
	target_pos[0] = 0 > min_1 ? 0 : min_1;
	float min_2 = im_h < res_y ? im_h : res_y;
	target_pos[1] = 0 > min_2 ? 0 : min_2;
	float min_3 = im_w < res_w ? im_w : res_w;
	target_sz[0] = 10 > min_3 ? 10 : min_3;
	float min_4 = im_h < res_h ? im_h : res_h;
	target_sz[1] = 10 > min_4 ? 10 : min_4;
}

// siamfc++_net2模型后处理,使用removeoutputcast接口去除cast&pruneaxis,需手动搬数&反量化
void net2_postprocess_removeoutputcast(const std::vector<Tensor>& output_tensors, std::vector<float> net2_output_normratio, std::vector<float>& target_pos, std::vector<float>& target_sz, cv::Mat xy_ctr,
	cv::Mat window, float window_influence, float x_size, float scale, int im_w, int im_h) {

	// 去除cast算子，手动将输出的tensor 从 pl_ddr 搬移到 ps_ddr
	auto host_tensor_0 = output_tensors[0].to(HostDevice::MemRegion());
	auto net2_data_ptr_0 = (int8_t*)host_tensor_0.data().cptr();

	auto host_tensor_1 = output_tensors[1].to(HostDevice::MemRegion());
	auto net2_data_ptr_1 = (int8_t*)host_tensor_1.data().cptr();

	auto host_tensor_2 = output_tensors[2].to(HostDevice::MemRegion());
	auto net2_data_ptr_2 = (int8_t*)host_tensor_2.data().cptr();


	cv::Mat net_result_mat = cv::Mat(289, 4, CV_8S, net2_data_ptr_0);
	net_result_mat.convertTo(net_result_mat, CV_32F);
	net_result_mat = net_result_mat * net2_output_normratio[0];//反量化

	net_result_mat = (net_result_mat * 0.78319091 + 1.60675728);
	cv::exp(net_result_mat, net_result_mat);
	net_result_mat = net_result_mat * 8;

	cv::Mat mat_1 = net_result_mat(cv::Rect(0, 0, 2, net_result_mat.rows));
	cv::Mat mat_2 = net_result_mat(cv::Rect(2, 0, 2, net_result_mat.rows));
	cv::Mat mat_3 = xy_ctr - mat_1;
	cv::Mat mat_4 = xy_ctr + mat_2;

	cv::Mat box;
	cv::hconcat(mat_3, mat_4, box);

	cv::Mat cls_score = cv::Mat(289, 1, CV_8S, net2_data_ptr_1);
	cls_score.convertTo(cls_score, CV_32F);
	cls_score = cls_score * net2_output_normratio[1];//反量化

	cv::Mat ctr_score = cv::Mat(289, 1, CV_8S, net2_data_ptr_2);
	ctr_score.convertTo(ctr_score, CV_32F);
	ctr_score = ctr_score * net2_output_normratio[2];//反量化

	for (int i = 0; i < cls_score.rows; i++) {
		for (int j = 0; j < cls_score.cols; j++) {
			float value = cls_score.at<float>(i, j);
			value = 1.0 / (1.0 + exp(-value));
			cls_score.at<float>(i, j) = value;
		}
	}

	for (int i = 0; i < ctr_score.rows; i++) {
		for (int j = 0; j < ctr_score.cols; j++) {
			float value = ctr_score.at<float>(i, j);
			value = 1.0 / (1.0 + exp(-value));
			ctr_score.at<float>(i, j) = value;
		}
	}
	// score = cls*ctr
	cv::Mat score;
	cv::multiply(cls_score, ctr_score, score);

	// box：xyxy2cxywh
	for (int i = 0; i < box.rows; i++) {
		float value_0 = box.at<float>(i, 0);
		float value_1 = box.at<float>(i, 1);
		float value_2 = box.at<float>(i, 2);
		float value_3 = box.at<float>(i, 3);
		box.at<float>(i, 0) = (value_0 + value_2) / 2;
		box.at<float>(i, 1) = (value_1 + value_3) / 2;
		box.at<float>(i, 2) = value_2 - value_0 + 1;
		box.at<float>(i, 3) = value_3 - value_1 + 1;
	}

	// score post-processing
	float penalty_k = 0.08;
	std::vector<float> target_sz_in_crop = { target_sz[0] * scale, target_sz[1] * scale };

	// box_wh[:, 2] box_wh[:, 3]
	cv::Mat box_wh_2 = box(cv::Rect(2, 0, 1, box.rows));
	cv::Mat box_wh_3 = box(cv::Rect(3, 0, 1, box.rows));

	// sz 
	cv::Mat pad_1 = (box_wh_2 + box_wh_3) * 0.5;
	cv::Mat sz2_1;
	cv::multiply((box_wh_2 + pad_1), (box_wh_3 + pad_1), sz2_1);
	cv::sqrt(sz2_1, sz2_1);

	// sz_wh 
	float pad_2 = (target_sz_in_crop[0] + target_sz_in_crop[1]) * 0.5;
	float sz2_2 = (target_sz_in_crop[0] + pad_2) * ((target_sz_in_crop[1] + pad_2));
	sz2_2 = sqrt(sz2_2);
	sz2_1 = sz2_1 / sz2_2;

	// change 
	for (int i = 0; i < sz2_1.rows; i++) {
		for (int j = 0; j < sz2_1.cols; j++) {
			float value = sz2_1.at<float>(i, j);
			sz2_1.at<float>(i, j) = value > (1 / value) ? value : (1 / value);
		}
	}

	float data_1 = target_sz_in_crop[0] / target_sz_in_crop[1];
	cv::Mat r_c = data_1 / (box_wh_2 / box_wh_3);

	// change
	for (int i = 0; i < r_c.rows; i++) {
		for (int j = 0; j < r_c.cols; j++) {
			float value = r_c.at<float>(i, j);
			r_c.at<float>(i, j) = value > (1 / value) ? value : (1 / value);
		}
	}

	cv::Mat penalty;
	cv::multiply(r_c, sz2_1, penalty);
	penalty = (penalty - 1) * (-penalty_k);
	cv::exp(penalty, penalty);

	// pscore = penalty * score
	cv::Mat pscore;
	cv::multiply(penalty, score, pscore);

	pscore = pscore * (1 - window_influence) + window * window_influence;
	cv::Point maxLoc;
	cv::minMaxLoc(pscore, NULL, NULL, NULL, &maxLoc);

	int best_pscore_id = maxLoc.y;

	// box post-processing
	cv::Mat pred_in_crop = box(cv::Rect(0, best_pscore_id, 4, 1)) / scale;
	float test_lr = 0.58;
	float lr = penalty.at<float>(best_pscore_id, 0) * score.at<float>(best_pscore_id, 0) * test_lr;
	// float test = pred_in_crop.at<float>(0, 0)- (int(x_size) / 2) / scale;
	// std::cout<<"test"<<test<<std::endl;
	float res_x = pred_in_crop.at<float>(0, 0) + target_pos[0] - (int(x_size) / 2) / scale;
	float res_y = pred_in_crop.at<float>(0, 1) + target_pos[1] - (int(x_size) / 2) / scale;
	float res_w = target_sz[0] * (1 - lr) + pred_in_crop.at<float>(0, 2) * lr;
	float res_h = target_sz[1] * (1 - lr) + pred_in_crop.at<float>(0, 3) * lr;

	// restrict new_target_pos & new_target_sz
	//int x = i > j ? i : j; // max(i,j)
	//int y = i < j ? i : j; // min(i,j)

	float min_1 = im_w < res_x ? im_w : res_x;
	target_pos[0] = 0 > min_1 ? 0 : min_1;
	float min_2 = im_h < res_y ? im_h : res_y;
	target_pos[1] = 0 > min_2 ? 0 : min_2;
	float min_3 = im_w < res_w ? im_w : res_w;
	target_sz[0] = 10 > min_3 ? 10 : min_3;
	float min_4 = im_h < res_h ? im_h : res_h;
	target_sz[1] = 10 > min_4 ? 10 : min_4;
}

std::string intToString(int v)
{
	char buf[32] = {0};
	snprintf(buf, sizeof(buf), "%u", v);
 
	std::string str = buf;
	return str;
}

std::vector<float> getOutputsNormratio(icraft::xir::NetworkView network) {

	auto iore_post_results = network.outputs();
	std::vector<float> ret;
	ret.reserve(iore_post_results.size());
	for (auto&& value : iore_post_results) {
		auto b = value->dtype.getNormratio().value();
		ret.emplace_back(b[0]);
	}
	return ret;
}



