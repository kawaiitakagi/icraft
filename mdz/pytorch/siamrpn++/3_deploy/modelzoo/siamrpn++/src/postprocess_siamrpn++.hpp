#pragma once
#include <opencv2/opencv.hpp>
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>
#include <modelzoo_utils.hpp>
#define PI 3.1415926

using namespace icraft::xrt;

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


// siamrpn++模型前处理
void siamrpn_preprocess(std::vector<float> target_pos, std::vector<float> target_sz,  std::vector<float> im_sz, cv::Mat& im_patch_template, cv::Mat& frame, float context_amount,float& scale, int z_size, int t_size) {
	float wc = target_sz[0] + context_amount * (target_sz[0] + target_sz[1]);
	float hc = target_sz[1] + context_amount * (target_sz[0] + target_sz[1]);
	float s_crop = sqrt(wc * hc);
	scale = z_size / s_crop;
	float s_x = s_crop * (t_size / z_size);

	float sz = round(s_x);
	float c = (sz + 1) / 2;
	float context_xmin = floor(target_pos[0] - c + 0.5);
	float context_xmax = context_xmin + sz - 1;
	float context_ymin = floor(target_pos[1] - c + 0.5);
	float context_ymax = context_ymin + sz - 1;

	//int x = i > j ? i : j; // max(i,j)
	//int y = i < j ? i : j; // min(i,j)
	int left_pad = 0. > -context_xmin ? 0. : -context_xmin;
	int top_pad = 0. > -context_ymin ? 0. : -context_ymin;
	int right_pad = 0. > (context_xmax - im_sz[1] + 1) ? 0. : (context_xmax - im_sz[1] + 1);
	int bottom_pad = 0. > (context_ymax - im_sz[0] + 1) ? 0. : (context_ymax - im_sz[0] + 1);

	context_xmin = context_xmin + left_pad;
	context_xmax = context_xmax + left_pad;
	context_ymin = context_ymin + top_pad;
	context_ymax = context_ymax + top_pad;

	float r = im_sz[0];
	float c_ = im_sz[1];
	float k = im_sz[2];

	// 仿射变换
	if (top_pad || bottom_pad || left_pad || right_pad) {
		std::vector<float> size = { r + top_pad + bottom_pad, c_ + left_pad + right_pad, k };
		cv::Mat te_im = cv::Mat::zeros(cv::Size(size[1], size[0]), frame.type());
		cv::Mat te_im_crop = te_im(cv::Rect(left_pad, top_pad, c_, r));
		frame.copyTo(te_im_crop);
		im_patch_template = te_im(cv::Range(int(context_ymin), int(context_ymax + 1)), cv::Range(int(context_xmin), int(context_xmax + 1)));

	}
	else {
		im_patch_template = frame(cv::Range(int(context_ymin), int(context_ymax + 1)), cv::Range(int(context_xmin), int(context_xmax + 1)));
	}

	if (sz != t_size) {
		cv::resize(im_patch_template, im_patch_template, cv::Size(t_size, t_size));
	}
}

// siamfc++_net2模型后处理，使用cast算子
void net2_postprocess_withcast(const std::vector<Tensor>& output_tensors, std::vector<float>& target_pos, std::vector<float>& target_sz, cv::Mat anchor, cv::Mat window, float window_influence, float scale, std::vector<float> im_sz) {
	auto net2out_ptr_0 = (float*)output_tensors[0].data().cptr();//cls
	auto net2out_ptr_1 = (float*)output_tensors[1].data().cptr();//loc

	//对cls进行后处理
	cv::Mat cls_mat = cv::Mat(25 * 25, 10, CV_32F, net2out_ptr_0);//[25*25,10,1]
	cv::Mat cls_mat_t = cls_mat.t();
	cv::Mat cls_mat_shape = cls_mat_t.reshape(0, 2).t();
	// 用opencv按行求 Softmax
	cv::Mat cls_softmaxMat;
	cv::exp(cls_mat_shape, cls_softmaxMat);
	cv::Mat rowSums;
	cv::reduce(cls_softmaxMat, rowSums, 1, cv::REDUCE_SUM);
	cv::Mat rowSums_repeated;
	cv::repeat(rowSums, 1, 2, rowSums_repeated);
	cv::divide(cls_softmaxMat, rowSums_repeated, cls_softmaxMat);

	cv::Mat cls_score = cls_softmaxMat(cv::Rect(1, 0, 1, cls_softmaxMat.rows));


	// 对loc进行后处理
	cv::Mat loc_mat = cv::Mat(25 * 25, 20, CV_32F, net2out_ptr_1);//[25*25,20,1]
	cv::Mat loc_mat_t = loc_mat.t();
	cv::Mat loc_mat_shape = loc_mat_t.reshape(0, 4);
	cv::multiply(loc_mat_shape.row(0), anchor.row(2), loc_mat_shape.row(0));
	loc_mat_shape.row(0) = loc_mat_shape.row(0) + anchor.row(0);
	cv::multiply(loc_mat_shape.row(1), anchor.row(3), loc_mat_shape.row(1));
	loc_mat_shape.row(1) = loc_mat_shape.row(1) + anchor.row(1);
	cv::exp(loc_mat_shape.row(2), loc_mat_shape.row(2));
	cv::multiply(loc_mat_shape.row(2), anchor.row(2), loc_mat_shape.row(2));
	cv::exp(loc_mat_shape.row(3), loc_mat_shape.row(3));
	cv::multiply(loc_mat_shape.row(3), anchor.row(3), loc_mat_shape.row(3));

	float pad = (target_sz[0] * scale + target_sz[1] * scale) * 0.5;
	float sz_1 = sqrt((target_sz[0] * scale + pad) * (target_sz[1] * scale + pad));

	float rc_1 = target_sz[0] / target_sz[1];

	// scale penalty
	cv::Mat loc_pad = (loc_mat_shape.row(2) + loc_mat_shape.row(3)) * 0.5;
	cv::Mat loc_pad_;
	cv::multiply((loc_mat_shape.row(2) + loc_pad), (loc_mat_shape.row(3) + loc_pad), loc_pad_);
	cv::sqrt(loc_pad_, loc_pad_);
	loc_pad_ = loc_pad_ / sz_1;
	for (int col = 0; col < loc_pad_.cols; col++) {
		float data = loc_pad_.at<float>(0, col);
		loc_pad_.at<float>(0, col) = data > (1 / data) ? data : (1 / data);
	}
	cv::Mat rc_mat;
	cv::divide(loc_mat_shape.row(2), loc_mat_shape.row(3), rc_mat);
	cv::Mat rc_mat_;
	cv::divide(rc_1, rc_mat, rc_mat_);
	for (int col = 0; col < rc_mat_.cols; col++) {
		float data = rc_mat_.at<float>(0, col);
		rc_mat_.at<float>(0, col) = data > (1 / data) ? data : (1 / data);
	}
	cv::Mat penalty_mat;
	cv::multiply(rc_mat_, loc_pad_, penalty_mat);
	cv::Mat penalty_mat_;
	cv::subtract(penalty_mat, 1.0, penalty_mat_);
	penalty_mat_ = penalty_mat_ * (-0.04);
	cv::exp(penalty_mat_, penalty_mat_);
	cv::Mat pscore_mat;
	cv::Mat penalty_mat_t = penalty_mat_.t();
	cv::multiply(penalty_mat_t, cls_score, pscore_mat);

	//window penalty
	cv::Mat pscore_mat_;
	pscore_mat_ = pscore_mat * window_influence + window * (1-window_influence);
	cv::Point maxLoc;
	cv::minMaxLoc(pscore_mat_, NULL, NULL, NULL, &maxLoc);
	int best_idx = 0;
	best_idx = maxLoc.y;
	std::vector<float> bbox;
	bbox.push_back(loc_mat_shape.at<float>(0, best_idx) / scale);
	bbox.push_back(loc_mat_shape.at<float>(1, best_idx) / scale);
	bbox.push_back(loc_mat_shape.at<float>(2, best_idx) / scale);
	bbox.push_back(loc_mat_shape.at<float>(3, best_idx) / scale);

	float lr = penalty_mat_.at<float>(0, best_idx) * cls_score.at<float>(best_idx, 0) * 0.5;
	float cx = bbox[0] + target_pos[0];
	float cy = bbox[1] + target_pos[1];
	//std::cout << "lr: " << lr << std::endl;
	// smooth bbox
	float width = target_sz[0] * (1 - lr) + bbox[2] * lr;
	float height = target_sz[1] * (1 - lr) + bbox[3] * lr;

	//clip boundary
	float cx_min = cx < im_sz[1] ? cx : im_sz[1];
	float cy_min = cy < im_sz[0] ? cy : im_sz[0];
	float width_min = width < im_sz[1] ? width : im_sz[1];
	float height_min = height < im_sz[0] ? height : im_sz[0];

	cx = 0 > cx_min ? 0 : cx_min;
	cy = 0 > cy_min ? 0 : cy_min;
	width = 10 > width_min ? 10 : width_min;
	height = 10 > height_min ? 10 : height_min;

	// 更新参数
	target_pos[0] = cx;
	target_pos[1] = cy;
	target_sz[0] = width;
	target_sz[1] = height;
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



