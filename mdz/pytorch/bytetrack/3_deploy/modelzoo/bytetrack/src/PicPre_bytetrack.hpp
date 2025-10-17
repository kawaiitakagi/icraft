#pragma once
#include <vector>
#include <string>
#include "icraft_utils.hpp"


class PicPre_bytetrack : public  PicPre {
public:
	PicPre_bytetrack(const std::string& filename, int flags = IMREAD_COLOR) {
		this->src_img = cv::imread(filename, flags);
		this->ori_img = cv::imread(filename, flags);
		this->src_dims = { src_img.channels(), src_img.rows, src_img.cols };
		if (this->src_img.channels() == 3) {
			cv::cvtColor(this->src_img, this->src_img, cv::COLOR_BGR2RGB);
		}

	}
	PicPre_bytetrack(const cv::Mat& img) {
		this->src_img = img.clone();
		this->ori_img = img.clone();
		this->src_dims = { src_img.channels(), src_img.rows, src_img.cols };
		if (this->src_img.channels() == 3) {
			cv::cvtColor(this->src_img, this->src_img, cv::COLOR_BGR2RGB);
		}

	}
	
	virtual PicPre_bytetrack& Resize(std::pair<int, int> dst_shape_hw, int mode = LONG_SIDE, int interpolation = INTER_LINEAR) {
		std::cout << "test" << std::endl;
		int ori_img_h = std::get<1>(this->src_dims);
		int ori_img_w = std::get<2>(this->src_dims);
		int resized_h = dst_shape_hw.first;
		int resized_w = dst_shape_hw.second;

		float ratio_h = (float)resized_h / (float)ori_img_h;
		float ratio_w = (float)resized_w / (float)ori_img_w;

		this->_dst_shape = { resized_h, resized_w };
		switch (mode) {
		case 0: {
			cv::resize(this->src_img, this->dst_img, cv::Size(resized_w, resized_h), 0, 0, interpolation);
			this->_real_resized_ratio = { ratio_h,ratio_w };
			this->_real_resized_hw = { resized_h,resized_w };
			break;
		}

		case 1: {
			float ratio = (std::min)(ratio_w, ratio_h);
			int real_resized_h = int(ori_img_h * ratio);
			int real_resized_w = int(ori_img_w * ratio);
			cv::resize(this->src_img, this->dst_img, cv::Size(real_resized_w, real_resized_h), 0, 0, interpolation);
			this->_real_resized_ratio = { ratio,ratio };
			this->_real_resized_hw = { real_resized_h,real_resized_w };
			break;
		}

		case 2: {
			float ratio = (std::max)(ratio_w, ratio_h);
			int real_resized_h = int(ori_img_h * ratio);
			int real_resized_w = int(ori_img_w * ratio);
			cv::resize(this->src_img, this->dst_img, cv::Size(real_resized_w, real_resized_h), 0, 0, interpolation);
			this->_real_resized_ratio = { ratio,ratio };
			this->_real_resized_hw = { real_resized_h,real_resized_w };
			break;
		}

		default: {
			throw "wrong resize mode!";
			exit(EXIT_FAILURE);
		}

		}

		return *this;

	}
};