#pragma once
#include <opencv2/opencv.hpp>
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>
#include <modelzoo_utils.hpp>
#include <opencv2/opencv.hpp>
//#define DEBUG_PRINT

void calctime_detail_ylm(icraft::xrt::Session& session, std::string& network_name) {
	checkDir("./logs/");
	std::string filePath = "./logs/" + network_name + "_time" + ".txt";
	std::ofstream ofs(filePath.c_str(), std::ios::out);

	float total_hard_time = 0;
	float total_time = 0;
	float total_memcpy_time = 0;
	float total_other_time = 0;
	float hardop_total_time = 0;
	float hardop_hard_time = 0;
	float hardop_memcpy_time = 0;

	bool imk_on = false;
	bool post_on = false;

	float out_cast_time = 0;
	float icore_in_time = 0;
	float icore_out_time = 0;
	float icore_time = 0;
	float cpu_time = 0;
	float customop_total_time = 0;
	float customop_hard_time = 0;
	std::string in_fpgaop = "cdma";
	std::string out_fpgaop = "cdma";
	std::string icore_fpgaop = "Null";
	std::string cpu_op = "Null";
	std::vector<std::tuple<std::string, float, float>> customops;
	std::map<std::string, float> customop_total_times;
	std::map<std::string, float> customop_hard_times;
	auto result = session.timeProfileResults();
	for (auto k_v : result) {
		//std::cout << k_v.first << std::endl;
		//std::cout << session->network_view.network().getOpById(k_v.first)->typeKey() << std::endl;
		//std::cout << session->network_view.network().getOpById(k_v.first)->name << std::endl;
		auto& [time1, time2, time3, time4] = k_v.second;

		for (auto& [op, _, _1, _2, _3] : session.getForwards()) {
			if (op->op_id == k_v.first) {
				auto op_typekey = op->typeKey();
				auto op_name = op->name;
				//std::cout << op->typeKey() << std::endl;
				//std::cout << op->name << std::endl;
				ofs << fmt::format("op_id: {}, op_type: {}, op_name: {}, total_time: {}, memcpy_time: {}, hard_time: {}, other_time: {}\n",
					k_v.first, op_typekey, op->name
					, time1, time2, time3, time4);
				total_time += time1;
				total_memcpy_time += time2;
				total_hard_time += time3;
				total_other_time += time4;
				if (op_typekey == "icraft::xir::HardOpNode") {
					hardop_total_time += time1;
					//hardop_total_time -= time2;
					hardop_memcpy_time += time2;
					hardop_hard_time += time3;
				}
				if (op_typekey == "icraft::xir::CastNode") {
					if (time2 > 0.001) {
						out_cast_time += time2;
					}
				}
				if (op_typekey.find("customop") != std::string::npos) {
					if (op_typekey.find("ImageMake") != std::string::npos) {
						imk_on = true;
						icore_in_time += time3;
						in_fpgaop = "ImageMake";
					}
					else if (op_typekey.find("Post") != std::string::npos) {
						post_on = true;
						icore_out_time += time1;
						if (out_fpgaop == "cdma") {
							out_fpgaop = op_typekey.substr(0, op_typekey.size() - 4).substr(10);
						}
						else if (out_fpgaop.find(std::string(op_typekey.substr(0, op_typekey.size() - 4).substr(10))) == std::string::npos) {
							out_fpgaop = out_fpgaop + ";" + std::string(op_typekey.substr(0, op_typekey.size() - 4).substr(10));
						}
					}
					else {
						icore_time += time1;
						if (icore_fpgaop == "Null") {
							icore_fpgaop = std::string(op_typekey.substr(0, op_typekey.size() - 4).substr(10));

						}
						else if (icore_fpgaop.find(std::string(op_typekey.substr(0, op_typekey.size() - 4).substr(10))) == std::string::npos) {
							icore_fpgaop = icore_fpgaop + ";" + std::string(op_typekey.substr(0, op_typekey.size() - 4).substr(10));

						}
					}
					customop_total_time += time1;
					customop_hard_time += time3;
					if (customop_total_times.find(std::string(op_typekey)) != customop_total_times.end()) {

						customop_total_times[std::string(op_typekey)] += time1;
						customop_hard_times[std::string(op_typekey)] += time3;

					}
					else {
						customop_total_times[std::string(op_typekey)] = time1;
						customop_hard_times[std::string(op_typekey)] = time3;
					}
				}

			}
		}
		// ofs << fmt::format("op_id: {}, op_type: {}, op_name: {}, total_time: {}, memcpy_time: {}, hard_time: {}, other_time: {}\n",
		//    k_v.first, session->network_view.network().getOpById(k_v.first)->typeKey(), session->network_view.network().getOpById(k_v.first)->name
		//    , time1, time2, time3, time4);

	}
	if (!post_on) {
		icore_out_time = out_cast_time;
		cpu_time = total_time - hardop_total_time - customop_total_time - icore_out_time;
	}
	else {
		cpu_time = total_time - hardop_total_time - customop_total_time;
	}
	if (!imk_on) {
		hardop_total_time -= hardop_memcpy_time;
		icore_in_time = hardop_memcpy_time;
	}

	if (cpu_time < 0) cpu_time = 0;
	ofs << "************************************" << std::endl;
	ofs << fmt::format("Total_TotalTime: {}, Total_MemcpyTime: {}, Total_HardTime: {}, Total_OtherTime: {}\n",
		total_time, total_memcpy_time, total_hard_time, total_other_time);
	ofs << fmt::format("Hardop_Total_Time: {} ms, Hardop_Hard_Time : {} ms.\n",
		hardop_total_time, hardop_hard_time);
	// ofs << fmt::format("Customop_Total_Time: {} ms, Customop_Hard_Time : {} ms.",
	// 	customop_total_time, customop_hard_time);    

	std::cout << "\n" << fmt::format("Total_TotalTime: {} ms, Total_MemcpyTime : {} ms, Total_HardTime : {} ms, Total_OtherTime : {} ms .",
		total_time, total_memcpy_time, total_hard_time, total_other_time) << std::endl;
	std::cout << fmt::format("Hardop_TotalTime: {} ms, Hardop_HardTime : {} ms.",
		hardop_total_time, hardop_hard_time) << std::endl;
	// std::cout << fmt::format("Customop_Total_Time: {} ms, Customop_Hard_Time : {} ms.",
	// 	customop_total_time, customop_hard_time) << std::endl;
	icore_time += hardop_total_time;
	for (const auto& pair : customop_total_times) {

		ofs << fmt::format("Customop: {},TotalTime: {} ms, HardTime : {} ms.\n",
			pair.first.substr(0, pair.first.size() - 4).substr(10), pair.second, customop_hard_times[pair.first]);
		std::cout << fmt::format("Customop: {},TotalTime: {} ms, HardTime : {} ms.",
			pair.first.substr(0, pair.first.size() - 4).substr(10), pair.second, customop_hard_times[pair.first]) << std::endl;
	}
	ofs << "******************************************************\n";
	std::cout << "******************************************************\n";
	ofs << "统计分析结果如下(The analysis results are as follows):\n";
	std::cout << "统计分析结果如下(The analysis results are as follows):\n";
	ofs << "数据传入耗时(Data input time consumption):\n";
	std::cout << "数据传入耗时(Data input time consumption):\n";
	ofs << "Time(ms):" << icore_in_time << "     Device:" << in_fpgaop << std::endl;
	std::cout << "Time(ms):" << icore_in_time << "     Device:" << in_fpgaop << std::endl;
	ofs << "icore[npu]耗时(Icore [npu] time-consuming):\n";
	std::cout << "icore[npu]耗时(Icore [npu] time-consuming):\n";
	ofs << "Time(ms):" << icore_time << "     Device:" << icore_fpgaop << std::endl;
	std::cout << "Time(ms):" << icore_time << "     Device:" << icore_fpgaop << std::endl;
	ofs << "数据传出耗时(Data output time consumption):\n";
	std::cout << "数据传出耗时(Data output time consumption):\n";
	ofs << "Time(ms):" << icore_out_time << "     Device:" << out_fpgaop << std::endl;
	std::cout << "Time(ms):" << icore_out_time << "     Device:" << out_fpgaop << std::endl;
	ofs << "cpu算子耗时(CPU operator time consumption):\n";
	std::cout << "cpu算子耗时(CPU operator time consumption):\n";
	ofs << "Time(ms):" << cpu_time << "     Device:" << cpu_op << std::endl;
	std::cout << "Time(ms):" << cpu_time << "     Device:" << cpu_op << std::endl;
	std::cout << "******************************************************\n";
	ofs.close();

	std::cout << "For details about running time meassage of the network, check the " + network_name + "_time" + ".txt" + " in path: " + "./logs/" << std::endl;
};

void postprocess_tsn(const std::vector<Tensor>& result_tensor, bool& show, bool& save, std::string& labelRoot,std::string& resRoot, std::string& img_path, std::string& name
) {
    //std::cout << result_tensor[0].dtype()->shape << std::endl;//[1,256,256,1]
	auto net2_post_start = std::chrono::high_resolution_clock::now();
    auto host_tensor = result_tensor[0].to(HostDevice::MemRegion());
    auto tensor_data = (float*)host_tensor.data().cptr();
	cv::Mat cls_scores = cv::Mat(1, 400, CV_32F, tensor_data);
	std::cout << "cls_scores shape: " << cls_scores.size << std::endl;
	auto net2_post1 = std::chrono::high_resolution_clock::now();
	auto net2_post1_dura = std::chrono::duration_cast<std::chrono::microseconds>
		(net2_post1 - net2_post_start);
	// softmax
	cv::Mat exp_scores;
	cv::exp(cls_scores, exp_scores);
	cv::Mat sum_exp_scores;
	cv::reduce(exp_scores, sum_exp_scores, 1, cv::REDUCE_SUM);
	cv::Mat softmax_scores = exp_scores / (sum_exp_scores.at<float>(0, 0) + 1e-6);
	//std::cout << "softmax_scores shape: " << softmax_scores.size << std::endl;
	auto net2_softmax = std::chrono::high_resolution_clock::now();
	auto net2_softmax_dura = std::chrono::duration_cast<std::chrono::microseconds>
		(net2_softmax - net2_post1);
	// 计算在第1维的均值
	cv::Mat mean_scores;
	cv::reduce(softmax_scores, mean_scores, 0, cv::REDUCE_AVG);
	auto net2_mean = std::chrono::high_resolution_clock::now();
	auto net2_mean_dura = std::chrono::duration_cast<std::chrono::microseconds>
		(net2_mean - net2_softmax);
	// 获取最大索引
	cv::Point maxIdx;
	double maxVal;
	cv::minMaxLoc(mean_scores, nullptr, &maxVal, nullptr, &maxIdx);
	int max_index = maxIdx.x;

	// 读取标签文件，获取分类类别名称
	auto labels = toVector(labelRoot);
	std::string top1_label = labels[max_index];
	double top1_score = mean_scores.at<float>(0, max_index);
	std::cout << "[Top1] label: " << top1_label << ", score: " << top1_score << std::endl;
	auto net2_maxlabel = std::chrono::high_resolution_clock::now();
	auto net2_maxlabel_dura = std::chrono::duration_cast<std::chrono::microseconds>
		(net2_maxlabel - net2_mean);
	auto net2_post_total = std::chrono::duration_cast<std::chrono::microseconds>
		(net2_maxlabel - net2_post_start);
	#ifdef DEBUG_PRINT
		spdlog::info("[Net2 Postprocess] build_host_tensor={:.2f}ms, softmax_tensor={:.2f}ms, mean_tensor={:.2f}ms, max&label_tensor={:.2f}ms, postprocess total={:.2f}ms",
			float(net2_post1_dura.count()) / 1000,
			float(net2_softmax_dura.count()) / 1000,
			float(net2_mean_dura.count()) / 1000,
			float(net2_maxlabel_dura.count()) / 1000,
			float(net2_post_total.count()) / 1000
			);
	#endif
	// 排序找到 top5 类别
	//std::vector<std::pair<int, float>> score_tuples;
	//for (int i = 0; i < mean_scores.cols; ++i) {
	//	score_tuples.push_back({ i, mean_scores.at<float>(0, i) });
	//}

	//std::sort(score_tuples.begin(), score_tuples.end(),
	//	[](const std::pair<int, float>& a, const std::pair<int, float>& b) {
	//		return a.second > b.second;
	//	});

	//std::cout << "The top-5 labels with corresponding scores are:" << std::endl;
	//for (int i = 0; i < 5 && i < score_tuples.size(); ++i) {
	//	std::cout << labels[score_tuples[i].first] << ": " << score_tuples[i].second << std::endl;
	//}

	cv::Mat cur_img = cv::imread(img_path);
	double font_scale = 1;
	int thickness = 1;
	cv::putText(cur_img, top1_label, cv::Point(5, 25), cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
    #ifdef _WIN32
        if (show) {
            cv::imshow("results", cur_img);
            cv::waitKey(0);
        }
        if (save) {
            std::string save_path = resRoot + '/' + name;
            std::regex rgx("\\.(?!.*\\.)"); // 匹配最后一个点号（.）之前的位置，且该点号后面没有其他点号
            std::string RES_PATH = std::regex_replace(save_path, rgx, "_result.");
            cv::imwrite(RES_PATH, cur_img);
}
    #elif __linux__
        if (save) {
            std::string save_path = resRoot + '/' + name;
            std::regex rgx("\\.(?!.*\\.)"); // 匹配最后一个点号（.）之前的位置，且该点号后面没有其他点号
            std::string RES_PATH = std::regex_replace(save_path, rgx, "_result.");
            cv::imwrite(RES_PATH, cur_img);
        }
    #endif
}

std::string plin_tsn_postprocess(const std::vector<Tensor>& result_tensor, std::string labelRoot) {
	auto host_tensor = result_tensor[0].to(HostDevice::MemRegion());
	auto tensor_data = (float*)host_tensor.data().cptr();
	cv::Mat cls_scores = cv::Mat(1, 400, CV_32F, tensor_data);
	//std::cout << "cls_scores shape: " << cls_scores.size << std::endl;
	// softmax
	cv::Mat exp_scores;
	cv::exp(cls_scores, exp_scores);
	cv::Mat sum_exp_scores;
	cv::reduce(exp_scores, sum_exp_scores, 1, cv::REDUCE_SUM);
	cv::Mat softmax_scores = exp_scores / (sum_exp_scores.at<float>(0, 0) + 1e-6);
	//std::cout << "softmax_scores shape: " << softmax_scores.size << std::endl;

	// 计算在第1维的均值
	cv::Mat mean_scores;
	cv::reduce(softmax_scores, mean_scores, 0, cv::REDUCE_AVG);

	// 获取最大索引
	cv::Point maxIdx;
	double maxVal;
	cv::minMaxLoc(mean_scores, nullptr, &maxVal, nullptr, &maxIdx);
	int max_index = maxIdx.x;

	// 读取标签文件，获取分类类别名称
	auto labels = toVector(labelRoot);
	std::string top1_label = labels[max_index];
	double top1_score = mean_scores.at<float>(0, max_index);
	std::cout << "[Top1] label: " << top1_label << ", score: " << top1_score << std::endl;
	return top1_label;
}
