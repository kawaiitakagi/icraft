#include <algorithm>
#include <memory>
#include <string>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/buyi_device.h>
#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/cuda/device.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "icraft_utils.hpp"
#include "yaml-cpp/yaml.h"
#include "post_process_yolov5s_seg.hpp"
#include <task_queue.hpp>
#include <et_device.hpp>
using namespace icraft::xrt;
using namespace icraft::xir;


int main(int argc, char* argv[])
{
	YAML::Node config = YAML::LoadFile(argv[1]);
	// icraft模型部署相关参数配置
	auto imodel = config["imodel"];
	// 仿真上板的jrpath配置
	std::string folderPath = imodel["dir"].as<std::string>();
	bool run_sim = imodel["sim"].as<bool>();
    bool cudamode = imodel["cudamode"].as<bool>();
	std::string JSON_PATH = getJrPath(run_sim, folderPath, imodel["stage"].as<std::string>());
	std::regex rgx3(".json");
	std::string RAW_PATH = std::regex_replace(JSON_PATH, rgx3, ".raw");


	// 数据集相关参数配置
	auto dataset = config["dataset"];
	std::string imgRoot = dataset["dir"].as<std::string>();
	std::string imgList = dataset["list"].as<std::string>();
	std::string names_path = dataset["names"].as<std::string>();
	auto LABELS = toVector(names_path);
	// 模型自身相关参数配置
	auto param = config["param"];
	float conf = param["conf"].as<float>();
	float iou_thresh = param["iou"].as<float>();
	bool MULTILABEL = param["multilabel"].as<bool>();
	bool fpga_nms = param["fpga_nms"].as<bool>();
	int N_CLASS = param["number_of_class"].as<int>();
	int NOH = param["number_of_head"].as<int>();
	std::vector<std::vector<std::vector<float>>> ANCHORS =
		param["anchors"].as<std::vector<std::vector<std::vector<float>>>>();
	int mask_channel = param["mask_channel"].as<int>();
	int bbox_info_channel = param["bbox_info_channel"].as<int>();
	int protoh = param["protoh"].as<int>();
	int protow = param["protow"].as<int>();

	int NOA = 1;
	if (ANCHORS.size() != 0) {
		NOA = ANCHORS[0].size();
	}
	std::vector<int> ori_out_channles = { N_CLASS, bbox_info_channel, mask_channel };
	int parts = ori_out_channles.size();




	// 打开device
	Device device = openDevice(false, "", false);
	auto buyi_device = device.cast<BuyiDevice>();

	//-------------------------------------//
	//       配置摄像头
	//-------------------------------------//
	auto camera_config = config["camera"];
	// 摄像头输入尺寸
	int CAMERA_W = camera_config["cameraw"].as<int>();
	int CAMERA_H = camera_config["camerah"].as<int>();
	// ps端图像尺寸
	int FRAME_W = CAMERA_W;
	int FRAME_H = CAMERA_H;

	uint64_t BUFFER_SIZE = FRAME_H * FRAME_W * 2;
	Camera camera(buyi_device, BUFFER_SIZE);

	// 在udmabuf上申请摄像头缓存区 
	// 不同于rt2.2 在初始化camera时候在就在内部申请了缓存区，
	// 而是在外部申请了缓存区，进而 take get 都需要指定缓存区。
	auto camera_buf = buyi_device.getMemRegion("udma").malloc(BUFFER_SIZE, false);
	std::cout << "Cam buffer udma addr=" << camera_buf->begin.addr() << '\n';

	// 同样在 udmabuf上申请display缓存区
	const uint64_t DISPLAY_BUFFER_SIZE = FRAME_H * FRAME_W * 2;    // 摄像头输入为RGB565
	auto display_chunck = buyi_device.getMemRegion("udma").malloc(DISPLAY_BUFFER_SIZE, false);
	auto display = Display_pHDMI_RGB565(buyi_device, DISPLAY_BUFFER_SIZE, display_chunck);
	std::cout << "Display buffer udma addr=" << display_chunck->begin.addr() << '\n';




	// 加载network
	Network network = loadNetwork(JSON_PATH, RAW_PATH);
	//初始化netinfo
	NetInfo netinfo = NetInfo(network);
	float mask_normratio = network->ops[-3]->inputs[0]->dtype.getNormratio().value()[0];
	// PL端图像尺寸，即神经网络网络输入图片尺寸
	int NET_W = netinfo.i_cubic[0].w;
	int NET_H = netinfo.i_cubic[0].h;

	// 将网络拆分为image make和icore
	auto image_make = network.view(netinfo.inp_shape_opid + 1, netinfo.inp_shape_opid + 2);
	auto icore = network.view(netinfo.inp_shape_opid + 2);

	auto imk_session = Session::Create<BuyiBackend, HostBackend>(image_make, { buyi_device, HostDevice::Default() });
	auto icore_session = Session::Create<BuyiBackend, HostBackend>(icore, { buyi_device, HostDevice::Default() });

	const uint64_t IMK_OUTPUT_FTMP_SIZE = NET_H * NET_W * 4;
	auto chunck = buyi_device.getMemRegion("plddr").malloc(IMK_OUTPUT_FTMP_SIZE, false);

	// 将imagemake和icore的输入输出连接起来
	auto& imk_backend = imk_session->backends;
	auto imk_buyi_backend = imk_backend[0].cast<BuyiBackend>();
	imk_buyi_backend.userSetSegment(chunck, Segment::OUTPUT);

	auto& icore_backend = icore_session->backends;
	auto icore_buyi_backend = icore_backend[0].cast<BuyiBackend>();
	icore_buyi_backend.userSetSegment(chunck, Segment::INPUT);

	icore_buyi_backend.speedMode();

	// 初始化session
	const std::string MODEL_NAME = icore_session->network_view.network()->name;

	// session执行前必须进行apply部署操作
	imk_session.apply();
	icore_session.apply();

	std::cout << "Presentation forward operator ...." << std::endl;
	auto ops = imk_session.getForwards();
	for (auto&& op : ops) {
		std::cout << "op name:" << std::get<0>(op)->typeKey() << '\n';
	}
	ops = icore_session.getForwards();

	for (auto&& op : ops) {
		std::cout << "op name:" << std::get<0>(op)->typeKey() << '\n';
	}

	std::vector<int> real_out_channles =
		_getReal_out_channles(ori_out_channles, netinfo.detpost_bit, NOA);
	std::vector<std::vector<float>> _norm =
		set_norm_by_head(NOH, parts, netinfo.o_scale);



	// fake input
	std::vector<int64_t> output_shape = { 1, NET_W, NET_H, 3 };
	auto tensor_layout = icraft::xir::Layout("NHWC");
	auto output_type = icraft::xrt::TensorType(icraft::xir::IntegerType::UInt8(), output_shape, tensor_layout);
	auto output_tensor = icraft::xrt::Tensor(output_type).mallocOn(icraft::xrt::HostDevice::MemRegion());
	auto img_tensor_list = std::vector<Tensor>{ output_tensor };

	auto progress_printer = std::make_shared<ProgressPrinter>(1);
	auto FPS_COUNT_NUM = 30;
	auto color = cv::Scalar(255, 0, 128);
	double font_scale = 1;
	int thickness = 1;
	std::atomic<uint64_t> frame_num = 0;
	std::atomic<float> fps = 0.f;
	auto startfps = std::chrono::steady_clock::now();
	YoloPostResult post_results;



	// PL端的resize，需要resize到AI神经网络的尺寸
	auto ratio_bias = preprocess_plin(buyi_device, CAMERA_W, CAMERA_H, NET_W, NET_H, crop_position::top_left);


	// 用于神经网络结果的坐标转换
	float RATIO_W = std::get<0>(ratio_bias);
	float RATIO_H = std::get<1>(ratio_bias);
	int BIAS_W = std::get<2>(ratio_bias);
	int BIAS_H = std::get<3>(ratio_bias);
	std::vector<float> stride = get_stride(netinfo);

	int8_t* display_data = new int8_t[FRAME_W * FRAME_H * 2];

	while (true) {

		camera.take(camera_buf);
		auto image_tensor = imk_session.forward(img_tensor_list);
		camera.wait();


		auto icore_tensor = icore_session.forward(image_tensor);
		device.reset(1);


		camera.get(display_data, camera_buf);
		cv::Mat mat = cv::Mat(FRAME_H, FRAME_W, CV_8UC2, display_data);
		cv::Mat mask = cv::Mat::zeros(cv::Size(CAMERA_W, CAMERA_H), CV_8UC1);

		std::vector<int> id_list;
		std::vector<float> socre_list;
		std::vector<cv::Rect2f> box_list;
		std::vector<std::vector<float>> mask_info;
		for (size_t i = 0; i < icore_tensor.size() - 1; i++) {

			int output_tensors_bits = icore_tensor[i].dtype()->element_dtype.getStorageType().bits();

			int obj_num = icore_tensor[i].dtype()->shape[2];
			int anchor_length = icore_tensor[i].dtype()->shape[3];
			if (output_tensors_bits == 16) {
				auto tensor_data = (int16_t*)icore_tensor[i].data().cptr();
				for (size_t obj = 0; obj < obj_num; obj++) {
					int base_addr = obj * anchor_length;
					Grid grid = get_grid(output_tensors_bits, tensor_data, base_addr, anchor_length);
					get_cls_bbox_maskInfo(id_list, socre_list, box_list, mask_info, tensor_data, base_addr, grid, netinfo.o_scale[i], stride[i], ANCHORS[i][grid.anchor_index], N_CLASS, conf, MULTILABEL, mask_channel);
				}

			}
			else {
				auto tensor_data = (int8_t*)icore_tensor[i].data().cptr();
				for (size_t obj = 0; obj < obj_num; obj++) {
					int base_addr = obj * anchor_length;
					Grid grid = get_grid(output_tensors_bits, tensor_data, base_addr, anchor_length);
					get_cls_bbox_maskInfo(id_list, socre_list, box_list, mask_info, tensor_data, base_addr, grid, netinfo.o_scale[i], stride[i], ANCHORS[i][grid.anchor_index], N_CLASS, conf, MULTILABEL, mask_channel);


				}
			}
		}

		std::vector<std::tuple<int, float, cv::Rect2f, std::vector<float>>> nms_res;
		if (fpga_nms) {
			nms_res = nms_hard_mask(box_list, socre_list, id_list, mask_info, conf, iou_thresh, device);
		}
		else {
			nms_res = nms_soft_mask(id_list, socre_list, box_list, mask_info, iou_thresh);   // 后处理 之 NMS

		}

		auto proto = (float*)icore_tensor[3].data().cptr();
		cv::Mat proto_mat_float = cv::Mat(protoh * protow, mask_channel, CV_32F, proto);

		for (int index = 0; index < nms_res.size()&& index<3; ++index) {

			float x1 = std::get<2>(nms_res[index]).tl().x * RATIO_W;
			float y1 = std::get<2>(nms_res[index]).tl().y * RATIO_H;
			float w = std::get<2>(nms_res[index]).width * RATIO_W;
			float h = std::get<2>(nms_res[index]).height * RATIO_H;


			x1 = checkBorder(x1, 0.f, (float)(NET_W * RATIO_W));
			y1 = checkBorder(y1, 0.f, (float)(NET_H * RATIO_H));
			w = checkBorder(w, -x1, (float)((NET_W * RATIO_W) - x1));
			h = checkBorder(h, -y1, (float)((NET_H * RATIO_H) - y1));
			int id = std::get<0>(nms_res[index]);
			cv::Scalar color = classColor(id);

			cv::rectangle(mat, cv::Rect2f(x1, y1, w, h), color, 6, cv::LINE_8, 0);
			std::string s = LABELS[id].substr(0, LABELS[id].size() - 1) + ":" + std::to_string(int(round(std::get<1>(nms_res[index]) * 100))) + "%";
			cv::Size s_size = cv::getTextSize(s, cv::FONT_HERSHEY_COMPLEX, font_scale, thickness, 0);
			cv::rectangle(mat, cv::Point(x1, y1 - s_size.height - 6), cv::Point(x1 + s_size.width, y1), color, -1);
			cv::putText(mat, s, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);


			cv::Mat obj_mask_mat = cv::Mat(32, 1, CV_32F, std::get<3>(nms_res[index]).data());
			cv::Mat out_mask = proto_mat_float * obj_mask_mat ;
			cv::Mat masks_1 = out_mask.reshape(1, protoh);
			cv::Mat masks_3 = masks_1(cv::Range(y1 / 8, (y1 + h) / 8), cv::Range(x1 / 8, (x1 + w) / 8));
			cv::resize(masks_3, masks_3, cv::Size((w), (h)));
			masks_3.copyTo(mask(cv::Range(y1, y1 + masks_3.rows), cv::Range(x1, x1 + masks_3.cols)));

		}

		drawTextTwoConer(mat, fmt::format("FPS: {:.1f}", fps), MODEL_NAME, color);
		add(mat, color, mat, mask);
		display.show(display_data);

		cv::Mat out1;
		cvtColor(mat, out1, cv::COLOR_BGR5652BGR);
		cv::imwrite("./_thread_result.jpg", out1);
		frame_num++;
		if (frame_num == FPS_COUNT_NUM) {
			frame_num = 0;
			auto duration = std::chrono::duration_cast<microseconds>
				(std::chrono::steady_clock::now() - startfps) / FPS_COUNT_NUM;
			fps = 1000 / (float(duration.count()) / 1000);
			startfps = std::chrono::steady_clock::now();
		}
	}

	Device::Close(device);
	return 0;
}
