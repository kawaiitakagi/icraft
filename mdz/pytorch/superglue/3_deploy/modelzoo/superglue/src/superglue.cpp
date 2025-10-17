#include <iostream>
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/buyi_device.h>
#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/cuda/device.h>
#include <icraft-backends/hostbackend/backend.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include "icraft_utils.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <random>
#include "yaml-cpp/yaml.h"
#include <opencv2/core.hpp>

using namespace icraft::xrt;
using namespace icraft::xir;
namespace fs = std::filesystem;

cv::Mat kpts0_draw_points;
cv::Mat kpts1_draw_points;




int8_t* rearrangeData(int8_t* originalData, int dim1, int dim2, int dim3, int dim4) {
    // 计算总数据量
    int totalElements = dim1 * dim2 * dim3 * dim4;

    // 分配新内存
    int8_t* newData = new int8_t[totalElements];

    // 重新排列数据
    for (int i = 0; i < dim1; ++i) { // 遍历第一个维度
        for (int j = 0; j < dim2; ++j) { // 遍历原来的第二个维度（新的第三个维度）
            for (int k = 0; k < dim3; ++k) { // 遍历原来的第三个维度（新的第二个维度）
                for (int l = 0; l < dim4; ++l) { // 遍历第四个维度
                    // 计算原始数据的索引
                    int originalIndex = (i * dim2 * dim3 * dim4) + (j * dim3 * dim4) + (k * dim4) + l;
                    // 计算新数据的索引
                    int newIndex = (i * dim3 * dim2 * dim4) + (k * dim2 * dim4) + (j * dim4) + l;
                    // 将数据从原始位置复制到新位置
                    newData[newIndex] = originalData[originalIndex];
                }
            }
        }
    }

    // 返回新指针
    return newData;
}



void saveMatToTxt(const cv::Mat& mat, const std::string& filename) {
	// 打开输出文件流
	std::ofstream outputFile(filename);

	// 检查文件是否成功打开
	if (!outputFile.is_open()) {
		std::cerr << "can not open file！" << std::endl;
		return;
	}

	// 写入关键点描述符矩阵展开后的数据到文件的同一行
	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {
			if (mat.type() == CV_32S) {
				outputFile << mat.at<int>(i, j) << " ";
			}
			else if (mat.type() == CV_32F) {
				outputFile << mat.at<float>(i, j) << " ";
			}
			else if (mat.type() == CV_8S) {
				outputFile << static_cast<int>(mat.at<char>(i, j)) << " ";
			}
			else {
				std::cerr << "Unsupported matrix type!" << std::endl;
				outputFile.close();
				return;
			}
		}
	}

	outputFile << std::endl; // 每行数据结束添加换行符

	// 关闭文件流
	outputFile.close();

	std::cout << "Data mat in " << filename << " OK!" << std::endl;
}


cv::Mat logsumexp(const cv::Mat& input, int dim) {
	cv::Mat max_val;
	cv::reduce(input, max_val, dim, cv::REDUCE_MAX);
	//saveMatToTxt(max_val, "max_val.txt");
	cv::Mat result;
	cv::exp(input - cv::repeat(max_val, input.rows / max_val.rows, input.cols / max_val.cols), result);
	//saveMatToTxt(result, "result.txt");
	cv::reduce(result, result, dim, cv::REDUCE_SUM);
	//saveMatToTxt(result, "result.txt");
	cv::log(result, result);
	//saveMatToTxt(result, "result.txt");
	result += max_val;
	return result;
}


// Log-space Sinkhorn iterations
cv::Mat log_sinkhorn_iterations(const cv::Mat& Z, const cv::Mat& log_mu, const cv::Mat& log_nu, int iters) {
	cv::Mat u = cv::Mat::zeros(log_mu.size(), log_mu.type());
	cv::Mat v = cv::Mat::zeros(log_nu.size(), log_nu.type());

	for (int i = 0; i < iters; ++i) {
		cv::Mat temp;
		temp = Z + cv::repeat(v.t(), Z.rows, 1);
		u = log_mu - logsumexp(temp, 1);
		//saveMatToTxt(u, "u.txt");

		cv::Mat temp1;
		temp1 = Z + cv::repeat(u, 1, Z.cols);
		//saveMatToTxt(temp1, "temp1.txt");
		v = log_nu - logsumexp(temp1, 0).t();
		//std::cout << v.at<float>(0, 0) << std::endl;
		//saveMatToTxt(v, "v.txt");
	}

	return Z + cv::repeat(u, 1, Z.cols) + cv::repeat(v.t(), Z.rows, 1);
}

cv::Mat log_optimal_transport(const cv::Mat& scores, float alpha, int iters) {
	int m = scores.rows;
	int n = scores.cols;

	float ms = m * 1.0;
	float ns = n * 1.0;

	// 创建couplings矩阵
	cv::Mat couplings = cv::Mat::zeros(m + 1, n + 1, CV_32F);

	// 填充couplings矩阵
	scores.copyTo(couplings(cv::Rect(0, 0, n, m)));
	couplings(cv::Rect(n, 0, 1, m)) = alpha;
	couplings(cv::Rect(0, m, n, 1)) = alpha;
	couplings.at<float>(m, n) = alpha;
	//saveMatToTxt(couplings, "couplings.txt");
	float norm = -std::log(ms + ns);
	cv::Mat log_mu = cv::Mat::ones(m + 1, 1, CV_32F) * norm;
	log_mu.at<float>(m, 0) = std::log(ns) + norm;

	cv::Mat log_nu = cv::Mat::ones(n + 1, 1, CV_32F) * norm;
	log_nu.at<float>(n, 0) = std::log(ms) + norm;
	//saveMatToTxt(log_mu, "log_mu.txt");
	//saveMatToTxt(log_nu, "log_nu.txt");
	cv::Mat Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters);
	//saveMatToTxt(Z, "Z.txt");
	Z -= norm;

	return Z;
}



std::vector<std::vector<std::string>> readDataFromFile(const std::string& filename) {
	std::ifstream file(filename);
	std::vector<std::vector<std::string>> data;

	if (file.is_open()) {
		std::string line;
		while (std::getline(file, line)) {
			std::istringstream iss(line);
			std::vector<std::string> row;
			std::string word;
			while (iss >> word) {
				row.push_back(word);
			}
			data.push_back(row);
		}

		file.close();
	}
	else {
		std::cerr << "Unable to open file." << std::endl;
	}

	return data;
}


void saveDataToTxt(const std::string& img_file1, const std::string& img_file2, const cv::Mat& kpts0, const cv::Mat& kpts1, const cv::Mat& mscores0, const cv::Mat& indices0, const std::string& filename) {
	// 打开输出文件流，以追加模式打开
	std::ofstream outputFile(filename);

	// 检查文件是否成功打开
	if (!outputFile.is_open()) {
		std::cerr << "can not open file！" << std::endl;
		return;
	}

	// 写入图像文件名和额外字符串到文件
	outputFile << img_file1 << " " << img_file2 << " ";

	// 写入关键点描述符矩阵展开后的数据到文件的同一行
	for (int i = 0; i < kpts0.rows; i++) {
		for (int j = 0; j < kpts0.cols; j++) {
			outputFile << kpts0.at<float>(i, j) << " ";
		}
	}

	for (int i = 0; i < kpts1.rows; i++) {
		for (int j = 0; j < kpts1.cols; j++) {
			outputFile << kpts1.at<float>(i, j) << " ";
		}
	}

	for (int i = 0; i < mscores0.rows; i++) {
		for (int j = 0; j < mscores0.cols; j++) {
			outputFile << mscores0.at<float>(i, j) << " ";
		}
	}

	for (int i = 0; i < indices0.rows; i++) {
		for (int j = 0; j < indices0.cols; j++) {
			outputFile << indices0.at<float>(i, j) << " ";
			if (j < indices0.cols - 1) {
				outputFile << " "; // 在每个矩阵数据之间添加空格分隔符
			}
		}
	}

	outputFile << std::endl; // 每行数据结束添加换行符

	// 关闭文件流
	outputFile.close();

	std::cout << "Data save in " << filename << " OK!" << std::endl;
}


icraft::xrt::Tensor Image2Tensor(const std::string& img_path, uint64_t height, uint64_t width, const NetworkView& network)
{
	if (std::filesystem::exists(img_path) == false) {
		//LOG_EXCEPT.append("[Error in HostBackend::UserUtils::img2Xtensor] Image file not exist.");
	}

	cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

	cv::Mat resized;
	if (height != -1 && width != -1) {
		cv::Mat float_image;
		img.convertTo(float_image, CV_32F);
		//saveMatToTxt(float_image, "float_image.txt");
		cv::resize(float_image, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
	}
	else {
		resized = img;
	}

	// 根据网络输入数据类型构造输入数据
	auto input_value = network.inputs()[0];
	auto out_dtype = input_value.tensorType().clone();
	auto out_stor_type = out_dtype->element_dtype.getStorageType();
	cv::Mat converted;
	if (out_stor_type.is<xir::FloatType>()) {
		auto float_stor_type = out_stor_type.cast<xir::FloatType>();
		if (float_stor_type.isFP32()) {
			resized.convertTo(converted, CV_32F);
		}
		else if (float_stor_type.isFP16()) {
			resized.convertTo(converted, CV_16F);
		}
		else {
			ICRAFT_LOG(EXCEPT).append("[Error in HostBackend Image2Tensor] DataType {} is not supported.", float_stor_type->typeKey());
		}
	}
	else if (out_stor_type.is<xir::IntegerType>()) {
		auto int_stor_type = out_stor_type.cast<xir::IntegerType>();
		if (int_stor_type.isSInt8()) {
			resized.convertTo(converted, CV_8S);
		}
		else if (int_stor_type.isUInt8()) {
			resized.convertTo(converted, CV_8U);
		}
		else if (int_stor_type.isSInt16()) {
			resized.convertTo(converted, CV_16S);
		}
		else if (int_stor_type.isUInt16()) {
			resized.convertTo(converted, CV_16U);
		}
		else if (int_stor_type.isSInt32()) {
			resized.convertTo(converted, CV_32S);
		}
		else {
			ICRAFT_LOG(EXCEPT).append("[Error in HostBackend Image2Tensor] DataType {} is not supported.", int_stor_type->typeKey());
		}
	}
	else {
		ICRAFT_LOG(EXCEPT).append("[Error in HostBackend Image2Tensor] DataType {} is not supported.", out_stor_type->typeKey());
	}
	int H = converted.rows;
	int W = converted.cols;
	int C = converted.channels();
	//saveMatToTxt(resized, "resized.txt");
	std::vector<int64_t> output_shape = { 1, H, W, C };
	auto tensor_layout = xir::Layout("NHWC");
	out_dtype.setShape(output_shape);
	auto img_tensor = xrt::Tensor(out_dtype).mallocOn(xrt::HostDevice::MemRegion());
	memcpy(img_tensor.data().cptr(), converted.data, H * W * C * out_dtype->element_dtype.bits() / 8);
	return img_tensor;
}



void forimkdata(icraft::xrt::Tensor& input_tensor, Device& device) {

	auto dims = input_tensor.dtype()->shape;
	auto dst_c = dims[-1];
	auto dst_w = dims[-2];
	auto dst_h = dims[-3];

	//auto device = devices[0];
	uint64_t demo_reg_base = 0x1000C0000;  //new BYolo_demo_reg_base
	uint64_t mapped_base;
	auto uregion_ = device.getMemRegion("udma");
	auto utensor = input_tensor.to(uregion_);
	std::ofstream utensor_ofm("utensor.ftmp", std::ios::binary);
	utensor.dump(utensor_ofm);
	utensor_ofm.close();
	auto ImageMakeRddrBase = utensor.data().addr();

	auto ImageMakeChannel = dst_c;
	auto ImageMakeWidth = dst_w;
	auto ImageMakeHeight = dst_h;

	auto ImageMakeRlen = ((ImageMakeWidth * ImageMakeHeight - 1) / (24 / ImageMakeChannel) + 1) * 3;   //this
	auto ImageMakeLastSft = ImageMakeWidth * ImageMakeHeight - (ImageMakeRlen - 3) / 3 * (24 / ImageMakeChannel);
	device.defaultRegRegion().write(demo_reg_base + 0x4, ImageMakeRddrBase, true);
	device.defaultRegRegion().write(demo_reg_base + 0x8, ImageMakeRlen, true);
	device.defaultRegRegion().write(demo_reg_base + 0xC, ImageMakeLastSft, true);
	device.defaultRegRegion().write(demo_reg_base + 0x10, ImageMakeChannel, true);
	device.defaultRegRegion().write(demo_reg_base + 0x1C, 1, true); // 1 -> data from hp
	device.defaultRegRegion().write(demo_reg_base + 0x20, 0, true);

}


// 对矩阵按行求最大值和index
std::pair<cv::Mat, cv::Mat> rowMax(const cv::Mat& mat) {
	cv::Mat maxValues(mat.rows, 1, CV_32F);
	cv::Mat maxIndices(mat.rows, 1, CV_32S);

	for (int i = 0; i < mat.rows; i++) {
		cv::Mat row = mat.row(i);
		double maxValue;
		cv::Point maxLoc;
		cv::minMaxLoc(row, nullptr, &maxValue, nullptr, &maxLoc);
		maxValues.at<float>(i) = static_cast<float>(maxValue);
		maxIndices.at<int>(i) = maxLoc.x;
	}

	return std::make_pair(maxValues, maxIndices);
}


// 对矩阵按列求最大值和index
std::pair<cv::Mat, cv::Mat> columnMax(const cv::Mat& mat) {
	cv::Mat maxValues(1, mat.cols, CV_32F);
	cv::Mat maxIndices(1, mat.cols, CV_32S);

	for (int i = 0; i < mat.cols; i++) {
		cv::Mat col = mat.col(i);
		double maxValue;
		cv::Point maxLoc;
		cv::minMaxLoc(col, nullptr, &maxValue, nullptr, &maxLoc);
		maxValues.at<float>(i) = static_cast<float>(maxValue);
		maxIndices.at<int>(i) = maxLoc.y;
	}

	return std::make_pair(maxValues, maxIndices);
}





std::pair<cv::Mat, std::vector<icraft::xrt::Tensor> > superpoint_post(int border, const bool& run_sim, std::vector<icraft::xrt::Tensor>& output_tensors, const TensorType& dtype1, const TensorType& dtype2, const TensorType& dtype3, int net_w, int net_h, std::vector<float> norm, float keypoint_threshold, int n_kpt, cv::Mat img) {

	auto get_keypoints_start = std::chrono::system_clock::now();

	// 将输出的 tensor 从 pl_ddr 搬移到 ps_ddr
	auto host_tensor_0 = output_tensors[0].to(HostDevice::MemRegion());
	auto host_tensor_1 = output_tensors[1].to(HostDevice::MemRegion());

	auto tensor_data_0 = (int8_t*)host_tensor_0.data().cptr();  // 像素点得分矩阵
	auto tensor_data_1_base = (int8_t*)host_tensor_1.data().cptr();  // 特征描述符
	// 排布转换：rearrangeData等效cast对维度的转换 
	auto out_shape = output_tensors[1].dtype()->shape;
	int8_t* tensor_data_1 = rearrangeData(tensor_data_1_base, int(out_shape[1]), int(out_shape[2]), int(out_shape[3]), int(out_shape[4]));

	int output_type = host_tensor_0.dtype().getStorageType().bits();
	// 判别输出类型
	if (output_type == 32) {
		norm = { 1,1 };
	}

	cv::Mat out_scores;
	if (output_type == 8) {
		out_scores = cv::Mat(net_h, net_w, CV_8S, tensor_data_0);
		out_scores.convertTo(out_scores, CV_32F);
	}
	else if (output_type == 16) {
		out_scores = cv::Mat(net_h, net_w, CV_16S, tensor_data_0);
		out_scores.convertTo(out_scores, CV_32F);
	}
	else {
		out_scores = cv::Mat(net_h, net_w, CV_32F, tensor_data_0);
	}
	// 反量化
	out_scores = out_scores * norm[0];

	// 筛选出 score > keypoint_threshold 的关键点
	cv::Mat scores_more;  // 大于阈值的像素点的矩阵
	cv::compare(out_scores, keypoint_threshold, scores_more, cv::CMP_GT);

	std::vector<cv::Point> indices;  // 大于阈值的像素坐标
	cv::findNonZero(scores_more, indices);

	cv::Mat keypoints = cv::Mat(indices.size(), 2, CV_32F);  // 筛选后的坐标列表
	cv::Mat scores_tmp = cv::Mat(1, indices.size(), CV_32F);  // 筛选后的像素值
	int height = net_h;
	int width = net_w;
	int k = 0; // 记录 keypoints 数量
	// remove border直接就可以省去 flip 操作 通过 index.x 和 index.y 位置调整
	for (const cv::Point& index : indices) {
		if (index.y >= border && index.y < (height - border) && index.x >= border && index.x < (width - border)) {
			float value = out_scores.at<float>(index);
			scores_tmp.at<float>(0, k) = value;
			keypoints.at<float>(k, 0) = float(index.x);
			keypoints.at<float>(k, 1) = float(index.y);
			k++;
		}

	}
	std::cout << "keypoints_num: " << k << std::endl;


	cv::Mat sort_scores_id;
	cv::sortIdx(scores_tmp, sort_scores_id, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);

	auto nms_syj_start = std::chrono::system_clock::now();
	std::vector<float> scores_list; // nms之后的得分list
	cv::Mat keypoints_nms = cv::Mat(sort_scores_id.cols, 2, CV_32F); // nms之后的坐标
	std::vector<int> nms_indices;  // nms之后的排序下标
	int kidx = 0;
	for (int i = 0; i < sort_scores_id.cols; i++) {
		int idx = sort_scores_id.at<int>(0, i);
		bool keep = true;
		for (int nms_idx = 0; nms_idx < nms_indices.size(); nms_idx++) {
			if ((powf(keypoints.at<float>(idx, 0) - keypoints.at<float>(nms_indices[nms_idx], 0), 2) + powf(keypoints.at<float>(idx, 1) - keypoints.at<float>(nms_indices[nms_idx], 1), 2)) < 16) {
				keep = false;
				break;
			}
		}
		if (keep) {
			nms_indices.emplace_back(idx);
			scores_list.push_back(scores_tmp.at<float>(0, idx));
			keypoints_nms.at<float>(kidx, 0) = keypoints.at<float>(idx, 0);
			keypoints_nms.at<float>(kidx, 1) = keypoints.at<float>(idx, 1);
			kidx++;
		}
	}

	auto nms_syj_end = std::chrono::system_clock::now();
	auto nms_syj_time = std::chrono::duration_cast<std::chrono::microseconds>(nms_syj_end - nms_syj_start);
	auto time_nms_syj = double(nms_syj_time.count()) * std::chrono::microseconds::period::num / std::chrono::milliseconds::period::den;
	std::cout << "nms_syj_time: " << time_nms_syj << std::endl;

	std::cout << "nms_obj_num : " << nms_indices.size() << std::endl;

	keypoints = keypoints_nms(cv::Rect2f(0, 0, 2, nms_indices.size()));
	cv::Mat scores_top_nkpt = cv::Mat(1, scores_list.size(), CV_32F, scores_list.data());
	k = nms_indices.size();

	if (nms_indices.size() > n_kpt) {
		keypoints = keypoints(cv::Rect2f(0, 0, 2, n_kpt));
		scores_top_nkpt = scores_top_nkpt(cv::Rect2f(0, 0, n_kpt, 1));
		k = n_kpt;
	}

	auto get_keypoints_end = std::chrono::system_clock::now();
	auto get_keypoints_time = std::chrono::duration_cast<std::chrono::microseconds>(get_keypoints_end - get_keypoints_start);
	auto time_get_keypoints = double(get_keypoints_time.count()) * std::chrono::microseconds::period::num / std::chrono::milliseconds::period::den;
	std::cout << "get_keypoints_time: " << time_get_keypoints << std::endl;


	// 对superpoint画图

	// for (int i = 0; i < nms_indices.size(); i++) {

	// 	cv::Scalar color_ = cv::Scalar(u(e), u(e), u(e));

	// 	int x0 = keypoints.at<float>(i, 0);
	// 	int y0 = keypoints.at<float>(i, 1);

	// 	cv::circle(img, cv::Point(x0, y0), 2, (255, 255, 255), -1, cv::LINE_AA);

	// 	std::string str_match_number = "match_number    " + std::to_string(int(nms_indices.size()));

	// 	cv::putText(img, str_match_number, cv::Point(30, 40), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255), 1);

	// }

	// cv::imwrite("../io/output/superpoint_res.png", img);
	// cv::imshow("superpoint result", img);
	// cv::waitKey(0);


	auto result_1_h = net_h / 8;
	auto result_1_w = net_w / 8;
	auto result_1_c = 256;

	auto sample_descriptors_start = std::chrono::system_clock::now();
	// 对描述子进行后处理
	cv::Mat normalized_descriptors;
	if (output_type == 8) {
		normalized_descriptors = cv::Mat((net_w / 8) * (net_h / 8), 256, CV_8S, tensor_data_1);
		normalized_descriptors.convertTo(normalized_descriptors, CV_32F);
	}
	else if(output_type == 16) {
		normalized_descriptors = cv::Mat((net_w / 8) * (net_h / 8), 256, CV_16S, tensor_data_1);
		normalized_descriptors.convertTo(normalized_descriptors, CV_32F);
	}
	else {
		normalized_descriptors = cv::Mat((net_w / 8) * (net_h / 8), 256, CV_32F, tensor_data_1);
	}
	// 反量化
	normalized_descriptors = normalized_descriptors * norm[1];
	// saveMatToTxt(normalized_descriptors, "normalized_descriptors1.txt");
	// 所有元素减去3.5
	cv::Mat keypoints_des;
	cv::subtract(keypoints, 3.5, keypoints_des);

	cv::Mat net_scale = (cv::Mat_<float>(1, 2) << (net_w - 4.5), (net_h - 4.5));
	cv::Mat scale_k_repeated;

	cv::repeat(net_scale, keypoints_des.rows, 1, scale_k_repeated);
	cv::divide(keypoints_des, scale_k_repeated, keypoints_des);
	cv::subtract(keypoints_des * 2, 1, keypoints_des);

	// 对 grid_sample 进行优化 用opencv实现减少时间
	auto grid_sample_start = std::chrono::system_clock::now();

	cv::Mat grid_sample_out = cv::Mat(k, result_1_c, CV_32F, cv::Scalar(0));
	int flag_grid_id = 0;

	for (int row = 0; row < keypoints_des.rows; row++) {

		float grid_x = (keypoints_des.at<float>(row, 0) + 1) * 0.5 * (result_1_w - 1);
		float grid_y = (keypoints_des.at<float>(row, 1) + 1) * 0.5 * (result_1_h - 1);

		int x_w = floor(grid_x);
		int x_e = x_w + 1;

		int y_n = floor(grid_y);
		int y_s = y_n + 1;


		float d_w = grid_x - x_w;
		float d_e = x_e - grid_x;
		float d_n = grid_y - y_n;
		float d_s = y_s - grid_y;

		int flag = 0;

		if (y_n >= 0 && y_n < result_1_h && x_w >= 0 && x_w < result_1_w) {
			cv::Mat wn = normalized_descriptors.row(y_n * result_1_w + x_w);
			grid_sample_out.row(flag_grid_id) = grid_sample_out.row(flag_grid_id) + wn * d_e * d_s;
			flag++;
		}
		if (y_n >= 0 && y_n < result_1_h && x_e >= 0 && x_e < result_1_w) {
			cv::Mat en = normalized_descriptors.row(y_n * result_1_w + x_e);
			grid_sample_out.row(flag_grid_id) = grid_sample_out.row(flag_grid_id) + en * d_w * d_s;
			flag++;
		}

		if (y_s >= 0 && y_s < result_1_h && x_w >= 0 && x_w < result_1_w) {
			cv::Mat ws = normalized_descriptors.row(y_s * result_1_w + x_w);
			grid_sample_out.row(flag_grid_id) = grid_sample_out.row(flag_grid_id) + ws * d_e * d_n;
			flag++;
		}

		if (y_s >= 0 && y_s < result_1_h && x_e >= 0 && x_e < result_1_w) {
			cv::Mat es = normalized_descriptors.row(y_s * result_1_w + x_e);
			grid_sample_out.row(flag_grid_id) = grid_sample_out.row(flag_grid_id) + es * d_w * d_n;
			flag++;
		}

		if (flag > 0) {
			flag_grid_id++;
		}

	}

	cv::Mat des_out = grid_sample_out;

	auto sample_descriptors_end = std::chrono::system_clock::now();
	auto sample_descriptors_time = std::chrono::duration_cast<std::chrono::microseconds>(sample_descriptors_end - sample_descriptors_start);
	auto time_sample_descriptors = double(sample_descriptors_time.count()) * std::chrono::microseconds::period::num / std::chrono::milliseconds::period::den;
	std::cout << "sample_descriptors_time: " << time_sample_descriptors << std::endl;


	auto grid_sample_time = std::chrono::duration_cast<std::chrono::microseconds>(sample_descriptors_end - grid_sample_start);
	auto time_grid_sample = double(grid_sample_time.count()) * std::chrono::microseconds::period::num / std::chrono::milliseconds::period::den;
	std::cout << "grid_sample_time: " << time_grid_sample << std::endl;


	// 少补 补充到 top_k 
	auto mat2icrafttensor_start = std::chrono::system_clock::now();
	cv::Mat keypoints_norm;

	if (nms_indices.size() < n_kpt) {
		int flag = n_kpt - k;
		cv::Mat zeros_keypoints = cv::Mat::zeros(flag, keypoints.cols, keypoints.type());
		keypoints.push_back(zeros_keypoints);
		cv::Mat zeros_scores = cv::Mat::zeros(scores_top_nkpt.rows, flag, scores_top_nkpt.type());
		cv::hconcat(scores_top_nkpt, zeros_scores, scores_top_nkpt); // 增加 列
		cv::Mat zeros_des = cv::Mat::zeros(flag, des_out.cols, des_out.type());
		des_out.push_back(zeros_des);
	}


	// 将输出的 des, keypoints, scores 转换为 icraft::tensor
	auto& tensor_shape1 = dtype1->shape;
	auto& tensor_layout1 = dtype1->layout;

	std::vector<int64_t> output_shape1 = { 1, n_kpt, 1, 256 };
	auto output_type1 = TensorType(FloatType::FP32(), output_shape1, tensor_layout1);
	auto output_tensor1 = Tensor(output_type1).mallocOn(HostDevice::MemRegion());
	auto output_tptr1 = (float*)output_tensor1.data().cptr();
	std::copy_n((float*)des_out.data, n_kpt * 1 * 256, output_tptr1);

	auto& tensor_shape2 = dtype2->shape;
	auto& tensor_layout2 = dtype2->layout;

	cv::Mat center = (cv::Mat_<float>(1, 2) << (net_w / 2), (net_h / 2));
	cv::Mat center_repeated;
	cv::repeat(center, keypoints.rows, 1, center_repeated);
	cv::subtract(keypoints, center_repeated, keypoints_norm);
	cv::divide(keypoints_norm, (net_w * 0.7), keypoints_norm);

	std::vector<int64_t> output_shape2 = { 1, n_kpt, 1, 2 };
	auto output_type2 = TensorType(FloatType::FP32(), output_shape2, tensor_layout2);
	auto output_tensor2 = Tensor(output_type2).mallocOn(HostDevice::MemRegion());
	auto output_tptr2 = (float*)output_tensor2.data().cptr();
	std::copy_n((float*)keypoints_norm.data, n_kpt * 1 * 2, output_tptr2);

	auto& tensor_shape3 = dtype3->shape;
	auto& tensor_layout3 = dtype3->layout;


	std::vector<int64_t> output_shape3 = { 1, n_kpt, 1, 1 };
	auto output_type3 = TensorType(FloatType::FP32(), output_shape3, tensor_layout3);
	auto output_tensor3 = Tensor(output_type3).mallocOn(HostDevice::MemRegion());
	auto output_tptr3 = (float*)output_tensor3.data().cptr();
	std::copy_n((float*)scores_top_nkpt.data, n_kpt * 1 * 1, output_tptr3);

	std::vector<icraft::xrt::Tensor> out;
	out.push_back(output_tensor1);
	out.push_back(output_tensor2);
	out.push_back(output_tensor3);


	auto mat2icrafttensor_end = std::chrono::system_clock::now();
	auto mat2icrafttensor_time = std::chrono::duration_cast<std::chrono::microseconds>(mat2icrafttensor_end - mat2icrafttensor_start);
	auto time_mat2icrafttensor = double(mat2icrafttensor_time.count()) * std::chrono::microseconds::period::num / std::chrono::milliseconds::period::den;
	std::cout << "mat2icrafttensor_time: " << time_mat2icrafttensor << std::endl;


	// saveMatToTxt(des_out, "des_out.txt");
	// saveMatToTxt(keypoints_norm, "keypoints_norm.txt");
	// saveMatToTxt(scores_top_nkpt, "scores_top_nkpt.txt");

	return std::make_pair(keypoints, out);

}




int main(int argc, char* argv[]) {

	try {

		YAML::Node config = YAML::LoadFile(argv[1]);

		// ----------------------------icraft模型部署相关参数配置--------------------------
		auto imodel = config["imodel"];
		std::string folderPath1 = imodel["dir1"].as<std::string>();
		std::string folderPath2 = imodel["dir1"].as<std::string>();
		std::string folderPath3 = imodel["dir2"].as<std::string>();
		bool run_sim = imodel["sim"].as<bool>();
    	bool cudamode = imodel["cudamode"].as<bool>();
		std::string ip = imodel["ip"].as<std::string>();
		bool show = imodel["show"].as<bool>();
		bool save = imodel["save"].as<bool>();

		std::string JSON_PATH1 = getJrPath(run_sim, folderPath1, imodel["stage"].as<std::string>());
		std::string JSON_PATH2 = getJrPath(run_sim, folderPath2, imodel["stage"].as<std::string>());
		std::string JSON_PATH3 = getJrPath(run_sim, folderPath3, imodel["stage"].as<std::string>());
		std::regex rgx3(".json");
		std::string RAW_PATH1 = std::regex_replace(JSON_PATH1, rgx3, ".raw");
		std::string RAW_PATH2 = std::regex_replace(JSON_PATH2, rgx3, ".raw");
		std::string RAW_PATH3 = std::regex_replace(JSON_PATH3, rgx3, ".raw");

		// 网络参数
		// 模型自身相关参数配置
		auto param = config["param"];
		int net_1_w = param["net_1_w"].as<int>();
		int net_1_h = param["net_1_h"].as<int>();
		int net_2_w = param["net_2_w"].as<int>();
		int net_2_h = param["net_2_h"].as<int>();
		int n_kpt = param["n_kpt"].as<int>(); // 输入superglue的关键点个数 目前是默认两张图提取相同个数的关键点  多删 少补零
		std::vector<float> net_1_norm = param["net_1_norm"].as<std::vector<float>>();
		std::vector<float> net_2_norm = param["net_2_norm"].as<std::vector<float>>();
		float net_1_keypoint_threshold = param["net_1_keypoint_threshold"].as<float>();
		float net_2_keypoint_threshold = param["net_2_keypoint_threshold"].as<float>();

		double match_threshold = param["match_threshold"].as<double>();
		float bin_score = param["bin_score"].as<float>();
		int border = param["border"].as<int>();

		// 数据集相关参数配置
		auto dataset = config["dataset"];
		std::string imgRoot = dataset["dir"].as<std::string>();
		std::string imgList = dataset["list"].as<std::string>();
		std::string resRoot = dataset["res"].as<std::string>();
		checkDir(resRoot);

		// ---------------------------device and network setting-------------------------------------
		// create network
		auto create_net_start = std::chrono::system_clock::now();
		std::cout << "creatFromJsonFile start..." << std::endl;
		auto network1w = Network::CreateFromJsonFile(JSON_PATH1);
		auto network2w = Network::CreateFromJsonFile(JSON_PATH2);
		auto network3 = Network::CreateFromJsonFile(JSON_PATH3);
		std::cout << "creatFromJsonFile finish..." << std::endl;
		auto create_net_end = std::chrono::system_clock::now();
		auto create_net_time = std::chrono::duration_cast<std::chrono::microseconds>(create_net_end - create_net_start);
		auto time_create_net = double(create_net_time.count()) * std::chrono::microseconds::period::num / std::chrono::milliseconds::period::den;
		std::cout << "create_net_time: " << time_create_net << std::endl;
		// open device
		NetInfo netinfo = NetInfo(network3);
		Device device = openDevice(run_sim, ip, netinfo.mmu || imodel["mmu"].as<bool>(), cudamode);
		// load raw
		auto load_raw_start = std::chrono::system_clock::now();
		network1w.lazyLoadParamsFromFile(RAW_PATH1);
		network2w.lazyLoadParamsFromFile(RAW_PATH2);
		network3.lazyLoadParamsFromFile(RAW_PATH3);
		// delete SuperPoint cast
		// parse、optimize、quantize without 134、135、143、144, only delete 0 and 36
		auto network1 = network1w.viewExcept({ 0, 134, 135, 142, 143, 36 });
		auto network2 = network2w.viewExcept({ 0, 134, 135, 142, 143, 36 });
		auto load_raw_end = std::chrono::system_clock::now();
		auto load_raw_time = std::chrono::duration_cast<std::chrono::microseconds>(load_raw_end - load_raw_start);
		auto time_load_raw = double(load_raw_time.count()) * std::chrono::microseconds::period::num / std::chrono::milliseconds::period::den;
		std::cout << "load_raw_time: " << time_load_raw << std::endl;

		// ---------------------------session set and apply-------------------------------------
		auto init_net_start = std::chrono::system_clock::now();
		std::cout << "init start..." << std::endl;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		auto sess1 = initSession(run_sim, network1, device, netinfo.mmu || imodel["mmu"].as<bool>(), imodel["speedmode"].as<bool>(), imodel["compressFtmp"].as<bool>());
		auto sess2 = initSession(run_sim, network2, device, netinfo.mmu || imodel["mmu"].as<bool>(), imodel["speedmode"].as<bool>(), imodel["compressFtmp"].as<bool>());
		auto sess3 = initSession(run_sim, network3, device, netinfo.mmu || imodel["mmu"].as<bool>(), imodel["speedmode"].as<bool>(), imodel["compressFtmp"].as<bool>());

		std::cout << "init finish..." << std::endl;

		if (!run_sim) {
			auto buyi_backend1 = sess1->backends[0].cast<BuyiBackend>();
			auto buyi_backend2 = sess2->backends[0].cast<BuyiBackend>();
			auto buyi_backend3 = sess3->backends[0].cast<BuyiBackend>();

			auto weight_bytesize1 = buyi_backend1->phy_segment_map.at(Segment::WEIGHT)->byte_size; // wsc: WEIGHT to OUTPUT
			auto weight_bytesize2 = buyi_backend2->phy_segment_map.at(Segment::WEIGHT)->byte_size; // wsc: WEIGHT to OUTPUT
			auto weight_bytesize3 = buyi_backend3->phy_segment_map.at(Segment::WEIGHT)->byte_size;

			auto ftmp_bytesize1 = buyi_backend1->phy_segment_map.at(Segment::FTMP)->byte_size;
			auto ftmp_bytesize2 = buyi_backend2->phy_segment_map.at(Segment::FTMP)->byte_size;
			auto ftmp_bytesize3 = buyi_backend3->phy_segment_map.at(Segment::FTMP)->byte_size;

			auto ftmp_bytesize = ftmp_bytesize1 > ftmp_bytesize3 ? ftmp_bytesize1 : ftmp_bytesize3;
			auto ftmp_memchunk = device.defaultMemRegion().malloc(ftmp_bytesize);
			//auto ftmp_memchunk = device.getMemRegion("plddr").malloc(ftmp_bytesize, true); // wsc
			std::cout << "ftmp_bytesize" << ftmp_bytesize << std::endl;
			std::cout << "ftmp_memchunk" << ftmp_memchunk << std::endl;


			buyi_backend1.userSetSegment(ftmp_memchunk, Segment::FTMP);
			buyi_backend2.userSetSegment(ftmp_memchunk, Segment::FTMP);
			buyi_backend3.userSetSegment(ftmp_memchunk, Segment::FTMP);
		}

		sess1.enableTimeProfile(true);
		sess2.enableTimeProfile(true);
		sess3.enableTimeProfile(true);
		sess1.apply();
		sess2.apply();
		sess3.apply();


		auto init_net_end = std::chrono::system_clock::now();
		auto init_net_time = std::chrono::duration_cast<std::chrono::microseconds>(init_net_end - init_net_start);
		auto time_init_net = double(init_net_time.count()) * std::chrono::microseconds::period::num / std::chrono::milliseconds::period::den;
		std::cout << "init_net_time: " << time_init_net << std::endl;

		auto net_3_input_dtype1 = network3.inputs()[0].tensorType();
		auto net_3_input_dtype2 = network3.inputs()[1].tensorType();
		auto net_3_input_dtype3 = network3.inputs()[2].tensorType();
		auto net_3_input_dtype4 = network3.inputs()[3].tensorType();
		auto net_3_input_dtype5 = network3.inputs()[4].tensorType();
		auto net_3_input_dtype6 = network3.inputs()[5].tensorType();

		int index = 0;
		std::vector<std::vector<std::string>> namevector = readDataFromFile(imgList);
		int totalnum = namevector.size();
		for (auto name : namevector)
		{
			progress(index, totalnum);
			index++;
			std::string img_file1 = imgRoot + '/' + name[0];
			std::string img_file2 = imgRoot + '/' + name[1];

			std::cout << "Net1//////////////////////// " << std::endl;
			auto net_1_read_img_start = std::chrono::system_clock::now();
			//auto net_1_input_tensor = Image2Tensor(img_file1, net_1_h, net_1_w, network1);
			// 前处理
			PicPre img1(img_file1, cv::IMREAD_COLOR);

			img1.Resize({ net_1_h, net_1_w }, PicPre::LONG_SIDE).rPad();
			Tensor net_1_input_tensor = CvMat2Tensor(img1.dst_img, network1w);

			auto net_1_read_img_end = std::chrono::system_clock::now();
			auto net_1_read_img_time = std::chrono::duration_cast<std::chrono::microseconds>(net_1_read_img_end - net_1_read_img_start);
			auto time_net_1_read_img = double(net_1_read_img_time.count()) * std::chrono::microseconds::period::num / std::chrono::milliseconds::period::den;
			std::cout << "net_1_read_img_time: " << time_net_1_read_img << std::endl;

			///////////////////////////////////////////////////////////
			uint64_t demo_reg_base = 0x1000C0000;

			if (!run_sim) {
				forimkdata(net_1_input_tensor, device);

				//imkmk initial ：两个网络共用一个imk,每次使用需要重新初始化imk
				auto ops_net1 = network1->ops;
				Operation imagemake_op1;
				for (auto&& op : ops_net1) {
					if (op->typeKey() == "customop::ImageMakeNode") {
						imagemake_op1 = op;
						break;
					}
				}
				auto buyi_backend1 = sess1->backends[0].cast<BuyiBackend>();
				buyi_backend1.initOp(imagemake_op1);

				device.defaultRegRegion().write(demo_reg_base, 1, true);  // 启动imk

			}

			///////////////////////////////////////////////////////////
			auto superpoint_forward_start = std::chrono::system_clock::now();

			auto output_tensors1 = sess1.forward({ net_1_input_tensor });
			device.reset(1);

			auto superpoint_forward_end = std::chrono::system_clock::now();
			auto superpoint_forward_time = std::chrono::duration_cast<std::chrono::microseconds>(superpoint_forward_end - superpoint_forward_start);
			auto time_superpoint_forward = double(superpoint_forward_time.count()) * std::chrono::microseconds::period::num / std::chrono::milliseconds::period::den;
			std::cout << "superpoint_forward_time: " << time_superpoint_forward << std::endl;


			auto net1_forimk_time = std::chrono::duration_cast<std::chrono::microseconds>(superpoint_forward_start - net_1_read_img_end);
			auto time_net1_forimk = double(net1_forimk_time.count()) * std::chrono::microseconds::period::num / std::chrono::milliseconds::period::den;
			std::cout << "net1_forimk_time: " << time_net1_forimk << std::endl;

			// for(int i = 0; i < 100; i++){
			// 	std::cout << "No: " << i << std::endl;
			// 	auto superpoint_post_start = std::chrono::system_clock::now();
			// 	std::pair<cv::Mat, std::vector<icraft::xrt::Tensor> > out1 = superpoint_post(border, run_sim, output_tensors1, net_3_input_dtype1, net_3_input_dtype2, net_3_input_dtype3, net_1_w, net_1_h, net_1_norm, net_1_keypoint_threshold, n_kpt);

			// 	auto superpoint_post_end = std::chrono::system_clock::now();
			// 	auto superpoint_post_time = std::chrono::duration_cast<std::chrono::microseconds>(superpoint_post_end - superpoint_post_start);
			// 	auto time_superpoint_post = double(superpoint_post_time.count()) * std::chrono::microseconds::period::num / std::chrono::milliseconds::period::den;
			// 	std::cout << "supepoint_post_time: " << time_superpoint_post << std::endl;
			// }

			auto superpoint_post_start = std::chrono::system_clock::now();
			
			std::pair<cv::Mat, std::vector<icraft::xrt::Tensor> > out1 = superpoint_post(border, run_sim, output_tensors1, net_3_input_dtype1, net_3_input_dtype2, net_3_input_dtype3, net_1_w, net_1_h, net_1_norm, net_1_keypoint_threshold, n_kpt, img1.dst_img);

			auto superpoint_post_end = std::chrono::system_clock::now();
			auto superpoint_post_time = std::chrono::duration_cast<std::chrono::microseconds>(superpoint_post_end - superpoint_post_start);
			auto time_superpoint_post = double(superpoint_post_time.count()) * std::chrono::microseconds::period::num / std::chrono::milliseconds::period::den;
			std::cout << "supepoint_post_time: " << time_superpoint_post << std::endl;

			std::cout << "Net2//////////////////////// " << std::endl;
			//auto net_2_input_tensor = Image2Tensor(img_file2, net_2_h, net_2_w, network2);
			PicPre img2(img_file2, cv::IMREAD_COLOR);

			img2.Resize({ net_2_h, net_2_w }, PicPre::LONG_SIDE).rPad();
			Tensor net_2_input_tensor = CvMat2Tensor(img2.dst_img, network2w);

			//////////////////////////////////////////////////////////////////////////
			if (!run_sim) {
				forimkdata(net_2_input_tensor, device);

				//imkmk initial ：两个网络共用一个imk,每次使用需要重新初始化imk
				auto ops_net2 = network2->ops;
				Operation imagemake_op2;
				for (auto&& op : ops_net2) {
					if (op->typeKey() == "customop::ImageMakeNode") {
						imagemake_op2 = op;
						break;
					}
				}
				auto buyi_backend2 = sess2->backends[0].cast<BuyiBackend>();
				buyi_backend2.initOp(imagemake_op2);

				device.defaultRegRegion().write(demo_reg_base, 1, true);

			}
			////////////////////////////////////////////////////////////////////////

			auto output_tensors2 = sess2.forward({ net_2_input_tensor });

			for (uint64_t i = 0; i < output_tensors2.size(); i++)
			{
				output_tensors2[i].waitForReady(1000ms);
			}

			std::pair<cv::Mat, std::vector<icraft::xrt::Tensor> > out2 = superpoint_post(border, run_sim, output_tensors2, net_3_input_dtype4, net_3_input_dtype5, net_3_input_dtype6, net_2_w, net_2_h, net_2_norm, net_2_keypoint_threshold, n_kpt, img2.dst_img);
			device.reset(1);

			std::cout << "Net3//////////////////////// " << std::endl;
			auto superglue_forward_start = std::chrono::system_clock::now();

			auto output_tensors3 = sess3.forward({ out1.second[0], out1.second[1], out1.second[2], out2.second[0], out2.second[1], out2.second[2] });

			for (uint64_t i = 0; i < output_tensors3.size(); i++)
			{
				output_tensors3[i].waitForReady(1000ms);
			}

			device.reset(1);
			auto superglue_forward_end = std::chrono::system_clock::now();
			auto superglue_forward_time = std::chrono::duration_cast<std::chrono::microseconds>(superglue_forward_end - superglue_forward_start);
			auto time_superglue_forward = double(superglue_forward_time.count()) * std::chrono::microseconds::period::num / std::chrono::milliseconds::period::den;


			std::cout << "superglue_forward_time: " << time_superglue_forward << std::endl;

			// 绘制匹配结果

			auto log_optimal_transport_start = std::chrono::system_clock::now();

			cv::Mat scores_match = cv::Mat(n_kpt+1, n_kpt+1, CV_32F, (float*)output_tensors3[0].data().cptr());
			cv::Mat Z_mat;
			//saveMatToTxt(scores_match, "scores_match.txt");
			auto superglue_post_start = std::chrono::system_clock::now();

			// Z_mat = log_optimal_transport(scores_match, bin_score, 20);  // 对应superglue-indoor-256-norm-sinkhorn.pt
			cv::log(scores_match, Z_mat);  // 取对数 syj sinkhorn等效实现，对应superglue-indoor-256-norm.pt

			auto log_optimal_transport_end = std::chrono::system_clock::now();

			auto log_optimal_transport_time = std::chrono::duration_cast<std::chrono::microseconds>(log_optimal_transport_end - log_optimal_transport_start);
			auto log_optimal_transport = double(log_optimal_transport_time.count()) * std::chrono::microseconds::period::num / std::chrono::milliseconds::period::den;
			std::cout << "log_optimal_transport_time: " << log_optimal_transport << std::endl;


			Z_mat = Z_mat(cv::Rect2f(0, 0, n_kpt, n_kpt));  // 裁剪

			std::pair<cv::Mat, cv::Mat> max0 = rowMax(Z_mat);

			std::pair<cv::Mat, cv::Mat> max1 = columnMax(Z_mat);

			cv::Mat indices0 = max0.second;
			cv::Mat indices1 = max1.second;

			// indices1.gather(1, indices0)

			cv::Mat result(indices0.rows, indices0.cols, indices1.type());

			for (int row = 0; row < indices0.rows; ++row) {
				for (int col = 0; col < indices0.cols; ++col) {
					auto index = indices0.at<int>(row, col);
					result.at<int>(row, col) = indices1.at<int>(0, index);
				}
			}


			// (arange_like(indices0, 1)[None])

			cv::Mat result_index(indices0.rows, indices0.cols, indices1.type());

			for (int row = 0; row < result_index.rows; ++row) {
				result_index.at<int>(row, 0) = row;
			}

			// == 


			cv::Mat mutual0;

			cv::compare(result_index, result, mutual0, cv::CMP_EQ);

			mutual0 = mutual0 / 255.0;


			// indices0.gather(1, indices1)

			cv::Mat result_1(indices1.rows, indices1.cols, indices0.type());

			for (int row = 0; row < indices1.rows; ++row) {
				for (int col = 0; col < indices1.cols; ++col) {
					auto index = indices1.at<int>(row, col);
					result_1.at<int>(row, col) = indices0.at<int>(index, 0);
				}
			}

			cv::Mat result_1_index(indices1.rows, indices1.cols, indices0.type());

			for (int col = 0; col < result_1_index.cols; ++col) {
				result_1_index.at<int>(0, col) = col;
			}

			cv::Mat mutual1;

			cv::compare(result_1_index, result_1, mutual1, cv::CMP_EQ);

			mutual1 = mutual1 / 255.0;

			cv::Mat exp_max0_value;
			cv::exp(max0.first, exp_max0_value);  // 计算矩阵的指数


			mutual0.convertTo(mutual0, CV_32F);

			cv::Mat mscores0;

			cv::multiply(mutual0, exp_max0_value, mscores0);
			//std::cout << mscores0 << std::endl;

			cv::Mat mscores0_gather(indices1.rows, indices1.cols, mscores0.type());

			for (int row = 0; row < indices1.rows; ++row) {
				for (int col = 0; col < indices1.cols; ++col) {
					auto index = indices1.at<int>(row, col);
					mscores0_gather.at<int>(row, col) = mscores0.at<int>(index, 0);
				}
			}

			mutual1.convertTo(mutual1, CV_32F);

			cv::Mat mscores1;

			cv::multiply(mutual1, mscores0_gather, mscores1);

			cv::Mat mscores0_mask;  // mscores0 > self.config['match_threshold']

			cv::Mat conf_mat(mscores0.rows, mscores0.cols, mscores0.type(), cv::Scalar(match_threshold));

			//std::cout << mscores0 << std::endl;

			cv::compare(mscores0, conf_mat, mscores0_mask, cv::CMP_GT);

			mscores0_mask = mscores0_mask / 255.0;

			mscores0_mask.convertTo(mscores0_mask, CV_32F);

			cv::Mat valid0 = mutual0 & mscores0_mask;


			cv::Mat valid0_gather(indices1.rows, indices1.cols, valid0.type());

			for (int row = 0; row < indices1.rows; ++row) {
				for (int col = 0; col < indices1.cols; ++col) {
					auto index = indices1.at<int>(row, col);
					valid0_gather.at<int>(row, col) = valid0.at<int>(index, 0);
				}
			}

			cv::Mat valid1 = mutual1 & valid0_gather;

			cv::Mat valid0_ones(indices0.rows, indices0.cols, CV_32F, cv::Scalar(1));

			cv::Mat indices0_new(indices0.rows, indices0.cols, CV_32F, cv::Scalar(-1));

			indices0.convertTo(indices0, CV_32F);

			cv::Mat mul_1;
			cv::multiply(valid0, indices0, mul_1);

			cv::Mat mul_2;
			cv::multiply((valid0_ones - valid0), indices0_new, mul_2);

			indices0 = mul_1 + mul_2;

			cv::Mat valid1_ones(indices1.rows, indices1.cols, CV_32F, cv::Scalar(1));

			cv::Mat indices1_new(indices1.rows, indices1.cols, CV_32F, cv::Scalar(-1));

			indices1.convertTo(indices1, CV_32F);

			cv::Mat mul_3;
			cv::multiply(valid1, indices1, mul_3);

			cv::Mat mul_4;
			cv::multiply((valid1_ones - valid1), indices1_new, mul_4);

			indices1 = mul_3 + mul_4;

			cv::Mat kpts0 = out1.first;
			cv::Mat kpts1 = out2.first;


			auto superglue_post_end = std::chrono::system_clock::now();

			auto superglue_post_time = std::chrono::duration_cast<std::chrono::microseconds>(superglue_post_end - superglue_post_start);
			auto time_superglue_post = double(superglue_post_time.count()) * std::chrono::microseconds::period::num / std::chrono::milliseconds::period::den;
			std::cout << "superglue_post_time: " << time_superglue_post << std::endl;


			auto project_time = std::chrono::duration_cast<std::chrono::microseconds>(superglue_post_end - net_1_read_img_start);
			auto time_project = double(project_time.count()) * std::chrono::microseconds::period::num / std::chrono::milliseconds::period::den;
			std::cout << "---------------project_time ---> : " << time_project << std::endl;

			if (save) {
				std::string res_file = resRoot + "/" + name[0] + "_" + name[1] + "_result.txt";
				saveDataToTxt(img_file1, img_file2, kpts0, kpts1, mscores0, indices0, res_file);
			}

			if (show) {
				// 开始画图 

				cv::Mat mkpts0 = cv::Mat(n_kpt, 2, CV_32F);
				cv::Mat mkpts1 = cv::Mat(n_kpt, 2, CV_32F);
				cv::Mat match_scores = cv::Mat(n_kpt, 1, CV_32F);


				int match_number = 0;

				for (int row = 0; row < indices0.rows; row++) {
					auto index = indices0.at<float>(row, 0);
					if (index > -1) {
						mkpts0.at<float>(match_number, 0) = kpts0.at<float>(row, 0);
						mkpts0.at<float>(match_number, 1) = kpts0.at<float>(row, 1);

						mkpts1.at<float>(match_number, 0) = kpts1.at<float>(index, 0);
						mkpts1.at<float>(match_number, 1) = kpts1.at<float>(index, 1);

						match_scores.at<float>(match_number, 0) = mscores0.at<float>(row, 0);

						match_number++;

					}
				}

				mkpts0 = mkpts0(cv::Rect2f(0, 0, 2, match_number));
				mkpts1 = mkpts1(cv::Rect2f(0, 0, 2, match_number));
				match_scores = match_scores(cv::Rect2f(0, 0, 1, match_number));

				cv::Mat img_a = cv::imread(img_file1, 1);
				cv::Mat img_b = cv::imread(img_file2, 1);

				cv::resize(img_a, img_a, cv::Size(net_1_w, net_1_h), 0, 0, cv::INTER_LINEAR);
				cv::resize(img_b, img_b, cv::Size(net_2_w, net_2_h), 0, 0, cv::INTER_LINEAR);

				cv::Mat out = cv::Mat(net_1_h, (net_1_w + 10 + net_2_w), CV_8UC3);

				cv::Mat a_crop_out = out(cv::Rect2f(0, 0, net_1_w, net_1_h));
				img_a.copyTo(a_crop_out);

				cv::Mat b_crop_out = out(cv::Rect2f(net_1_w + 10, 0, net_2_w, net_2_h));
				img_b.copyTo(b_crop_out);

				// 绘制 关键点 和 连线

				std::cout << "match_num: " << match_number << std::endl;


				std::default_random_engine e;
				std::uniform_int_distribution<unsigned> u(10, 200);

				for (int i = 0; i < match_number; i++) {



					cv::Scalar color_ = cv::Scalar(u(e), u(e), u(e));

					int x0 = mkpts0.at<float>(i, 0);
					int y0 = mkpts0.at<float>(i, 1);

					int x1 = mkpts1.at<float>(i, 0);
					int y1 = mkpts1.at<float>(i, 1);

				cv:line(out, cv::Point(x0, y0), cv::Point(x1 + 10 + net_1_w, y1), color_, 1, cv::LINE_AA);

					cv::circle(out, cv::Point(x0, y0), 2, (255, 255, 255), -1, cv::LINE_AA);

					cv::circle(out, cv::Point(x1 + 10 + net_1_w, y1), 2, (255, 255, 255), -1, cv::LINE_AA);

				}


				std::string str_match_number = "match_number    " + std::to_string(int(match_number));

				cv::putText(out, str_match_number, cv::Point(30, 40), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255), 1);

			#ifdef _WIN32
				if (show) {
					cv::imshow("results", out);
					cv::waitKey(0);
				}
			#endif
				if (save) {
					cv::imwrite("../io/output/superglue_res.png", out);
				}
			}

			std::cout << "*********************************************************" << std::endl;

		}
		calctime_detail(sess1);
		calctime_detail(sess2);
		calctime_detail(sess3);
	}
	catch (const icraft::InternalError& err) {
		ICRAFT_LOG(FATAL) << err.what();
		ICRAFT_LOG(FATAL) << err.errorCode();
		std::exit(err.errorCode());
	}
	catch (const std::exception& err) {
		ICRAFT_LOG(FATAL) << err.what();
		std::exit(-1);
	}



	return 0;
}