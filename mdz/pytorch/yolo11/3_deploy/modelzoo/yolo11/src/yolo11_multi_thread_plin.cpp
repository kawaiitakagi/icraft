
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
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "icraft_utils.hpp"
#include "yaml-cpp/yaml.h"
#include "postprocess_yolo11.hpp"
#include <task_queue.hpp>
#include <et_device.hpp>
using namespace icraft::xrt;
using namespace icraft::xir;


int main(int argc, char* argv[])
{
	auto thread_num = 4;
	YAML::Node config = YAML::LoadFile(argv[1]);
	// icraft模型部署相关参数配置
	auto imodel = config["imodel"];
	// 仿真上板的jrpath配置
	std::string folderPath = imodel["dir"].as<std::string>();
	bool run_sim = imodel["sim"].as<bool>();
	std::string JSON_PATH = getJrPath(run_sim, folderPath, imodel["stage"].as<std::string>());
	std::regex rgx3(".json");
	std::string RAW_PATH = std::regex_replace(JSON_PATH, rgx3, ".raw");


	// 数据集相关参数配置
	auto dataset = config["dataset"];
	std::string imgRoot = dataset["dir"].as<std::string>();
	std::string imgList = dataset["list"].as<std::string>();
	std::string names_path = dataset["names"].as<std::string>();
	std::string resRoot = dataset["res"].as<std::string>();
	checkDir(resRoot);
	auto LABELS = toVector(names_path);
	// 模型自身相关参数配置
	auto param = config["param"];
	float conf = param["conf"].as<float>();
	float iou_thresh = param["iou_thresh"].as<float>();
	bool MULTILABEL = param["multilabel"].as<bool>();
	bool fpga_nms = param["fpga_nms"].as<bool>();
	int N_CLASS = param["number_of_class"].as<int>();
	int NOH = param["number_of_head"].as<int>();
	std::vector<std::vector<std::vector<float>>> ANCHORS =
		param["anchors"].as<std::vector<std::vector<std::vector<float>>>>();
	int bbox_info_channel = 64;

	int NOA = 1;
	if (ANCHORS.size() != 0) {
		NOA = ANCHORS[0].size();
	}
	std::vector<int> ori_out_channles = { N_CLASS, bbox_info_channel};
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
	auto camera_buf_group = std::vector<MemChunk>(thread_num);
	for (int i = 0; i < thread_num; i++) {
		auto chunck = buyi_device.getMemRegion("udma").malloc(BUFFER_SIZE, false);
		std::cout << "Cam buffer index:" << i
			<< " ,udma addr=" << chunck->begin.addr() << '\n';
		camera_buf_group[i] = chunck;
	}


	// 同样在 udmabuf上申请display缓存区
	const uint64_t DISPLAY_BUFFER_SIZE = FRAME_H * FRAME_W * 2;    // 摄像头输入为RGB565
	auto display_chunck = buyi_device.getMemRegion("udma").malloc(DISPLAY_BUFFER_SIZE, false);
	auto display = Display_pHDMI_RGB565(buyi_device, DISPLAY_BUFFER_SIZE, display_chunck);
	std::cout << "Display buffer udma addr=" << display_chunck->begin.addr() << '\n';




	// 加载network
	Network network = loadNetwork(JSON_PATH, RAW_PATH);
	//初始化netinfo
	NetInfo netinfo = NetInfo(network);
	// PL端图像尺寸，即神经网络网络输入图片尺寸
	int NET_W = netinfo.i_cubic[0].w;
	int NET_H = netinfo.i_cubic[0].h;

	// 在PLDDR上申请imagemake缓存区，用来缓存给AI计算的图片
	const uint64_t IMK_OUTPUT_FTMP_SIZE = NET_H * NET_W * 4;
	auto imagemake_buf_group = std::vector<MemChunk>(thread_num);
	for (int i = 0; i < thread_num; i++) {
		auto chunck = buyi_device.getMemRegion("plddr").malloc(IMK_OUTPUT_FTMP_SIZE, false);
		std::cout << "image make buffer index:" << i
			<< " ,plddr addr=" << chunck->begin.addr() << '\n';
		imagemake_buf_group[i] = chunck;
	}

	// 将网络拆分为image make和icore
	auto image_make = network.view(netinfo.inp_shape_opid + 1, netinfo.inp_shape_opid + 2);
	auto icore = network.view(netinfo.inp_shape_opid + 2);

	// 计算复网络的ftmp大小，用于复用相同网络的ftmp
	auto icore_dummy_session = Session::Create<BuyiBackend, HostBackend>(icore, { buyi_device, HostDevice::Default() });
	auto& icore_dummy_backends = icore_dummy_session->backends;
	auto icore_buyi_dummy_backends = icore_dummy_backends[0].cast<BuyiBackend>();
	// auto network_ftmp_size = icore_buyi_dummy_backends->phy_segment_map.at(Segment::FTMP)->byte_size;
	// std::cout << "network ftmp size=" << network_ftmp_size;
	std::cout << "begin to compress ftmp" << std::endl;
	icore_buyi_dummy_backends.compressFtmp();
	std::cout << "compress ftmp done" << std::endl;
	auto network_ftmp_size = icore_buyi_dummy_backends->phy_segment_map.at(Segment::FTMP)->byte_size;
	std::cout << "after compress network ftmp size=" << network_ftmp_size;
	auto network_ftmp_chunck = buyi_device.getMemRegion("plddr").malloc(network_ftmp_size, false);

	const std::string MODEL_NAME = icore_dummy_session->network_view.network()->name;

	// 生成多个session
	auto imk_sessions = std::vector<Session>(thread_num);
	auto icore_sessions = std::vector<Session>(thread_num);
	for (int i = 0; i < thread_num; i++) {
		// 创建session
		imk_sessions[i] = Session::Create<BuyiBackend, HostBackend>(image_make, { buyi_device, HostDevice::Default() });
		icore_sessions[i] = Session::Create<BuyiBackend, HostBackend>(icore, { buyi_device, HostDevice::Default() });

		// 将同一组imagemake和icore的输入输出连接起来
		auto& imk_backends = imk_sessions[i]->backends;
		auto imk_buyi_backend = imk_backends[0].cast<BuyiBackend>();
		imk_buyi_backend.userSetSegment(imagemake_buf_group[i], Segment::OUTPUT);

		auto& icore_backends = icore_sessions[i]->backends;
		auto icore_buyi_backend = icore_backends[0].cast<BuyiBackend>();
		icore_buyi_backend.userSetSegment(imagemake_buf_group[i], Segment::INPUT);

		// 压缩并复用多个网络的ftmp
		icore_buyi_backend.compressFtmp();
		icore_buyi_backend.userSetSegment(network_ftmp_chunck, Segment::FTMP);

		// speedmode
		std::cout << "open speed mode" << std::endl;
		icore_buyi_backend.speedMode();


		imk_sessions[i].apply();
		icore_sessions[i].apply();

		// icore_buyi_backend.log();
		// imk_buyi_backend.log();
		std::cout << "Presentation forward operator ...." << std::endl;
		auto ops = icore_sessions[i].getForwards();
		for (auto&& op : ops) {
			std::cout << "op name:" << std::get<0>(op)->typeKey() << '\n';
		}

	}

	std::vector<int> real_out_channles =
		_getReal_out_channles(ori_out_channles, netinfo.detpost_bit, N_CLASS);
	std::vector<std::vector<float>> _norm =
		set_norm_by_head(NOH, parts, netinfo.o_scale);
	std::vector<float> _stride = get_stride(netinfo);



	// fake input
	std::vector<int64_t> output_shape = { 1, NET_W, NET_H, 3 };
	auto tensor_layout = icraft::xir::Layout("NHWC");
	auto output_type = icraft::xrt::TensorType(icraft::xir::IntegerType::UInt8(), output_shape, tensor_layout);
	auto output_tensor = icraft::xrt::Tensor(output_type).mallocOn(icraft::xrt::HostDevice::MemRegion());
	auto img_tensor_list = std::vector<Tensor>{ output_tensor };

	auto progress_printer = std::make_shared<ProgressPrinter>(1);
	auto FPS_COUNT_NUM = 30;
	auto color = cv::Scalar(128, 0, 128);
	std::atomic<uint64_t> frame_num = 0;
	std::atomic<float> fps = 0.f;
	auto startfps = std::chrono::steady_clock::now();
	YoloPostResult post_results;


	auto icore_task_queue = std::make_shared<Queue<InputMessageForIcore>>(thread_num);
	auto post_task_queue = std::make_shared<Queue<IcoreMessageForPost>>(thread_num);

	// PL端的resize，需要resize到AI神经网络的尺寸
	auto ratio_bias = preprocess_plin(buyi_device, CAMERA_W, CAMERA_H, NET_W, NET_H, crop_position::center);


	// 用于神经网络结果的坐标转换
	float RATIO_W = std::get<0>(ratio_bias);
	float RATIO_H = std::get<1>(ratio_bias);
	int BIAS_W = std::get<2>(ratio_bias);
	int BIAS_H = std::get<3>(ratio_bias);

	std::vector<bool> buffer_avaiable_flag(thread_num, true);
	int8_t* display_data = new int8_t[FRAME_W * FRAME_H * 2];

	auto input_thread = std::thread(
		[&]()
		{
			std::stringstream ss;
			ss << std::this_thread::get_id();
			uint64_t id = std::stoull(ss.str());
			spdlog::info("[PLin_Vpu Demo] Input process thread start!, id={}", id);


			int buffer_index = 0;
			while (true) {
				InputMessageForIcore msg;
				msg.buffer_index = buffer_index;
				auto start = std::chrono::high_resolution_clock::now();

				while (!buffer_avaiable_flag[buffer_index]) {
					usleep(0);
				}

				camera.take(camera_buf_group[buffer_index]);

				try {
					msg.image_tensor = imk_sessions[buffer_index].forward(img_tensor_list);
					// device.reset(1);
				}
				catch (const std::exception& e) {
					msg.error_frame = true;
					icore_task_queue->Push(msg);
					continue;
				}


				auto imk_dura = std::chrono::duration_cast<std::chrono::microseconds>
					(std::chrono::high_resolution_clock::now() - start);

				if (!camera.wait()) {
					msg.error_frame = true;
					icore_task_queue->Push(msg);
					continue;
				}


				// 将buffer标记为不可用，等后处理完成后再释放
				buffer_avaiable_flag[buffer_index] = false;

				auto wait_dura = std::chrono::duration_cast<std::chrono::microseconds>
					(std::chrono::high_resolution_clock::now() - start);
				icore_task_queue->Push(msg);

				buffer_index++;
				buffer_index = buffer_index % camera_buf_group.size();
			}
		}
	);

	auto icore_thread = std::thread(
		[&]()
		{
			std::stringstream ss;
			ss << std::this_thread::get_id();
			uint64_t id = std::stoull(ss.str());
			spdlog::info("[PLin_Vpu Demo] Icore thread start!, id={}", id);

			while (true) {
				InputMessageForIcore input_msg;
				icore_task_queue->Pop(input_msg);

				IcoreMessageForPost post_msg;
				post_msg.buffer_index = input_msg.buffer_index;
				post_msg.error_frame = input_msg.error_frame;

				if (input_msg.error_frame) {
					post_task_queue->Push(post_msg);
					continue;
				}

				post_msg.icore_tensor
					= icore_sessions[input_msg.buffer_index].forward(input_msg.image_tensor);
				// for (auto&& o: post_msg.icore_tensor) {
				//     o.waitForReady(1000ms);
				// }
				device.reset(1);

				post_task_queue->Push(post_msg);
			}
		}
	);

	auto post_thread = std::thread(
		[&]()
		{
			std::stringstream ss;
			ss << std::this_thread::get_id();
			uint64_t id = std::stoull(ss.str());
			spdlog::info("[PLin_Vpu Demo] Post thread start!, id={}", id);
			auto color = cv::Scalar(128, 0, 128);
			int8_t* display_data = new int8_t[FRAME_W * FRAME_H * 2];
			while (true) {
				IcoreMessageForPost post_msg;
				post_task_queue->Pop(post_msg);

				if (post_msg.error_frame) {
					cv::Mat display_mat = cv::Mat::zeros(FRAME_W, FRAME_H, CV_8UC2);
					drawTextTopLeft(display_mat, fmt::format("No input , Please check camera."), cv::Scalar(127, 127, 127));
					display.show(reinterpret_cast<int8_t*>(display_mat.data));
					continue;
				}
				// post_results = post_detpost_plin_woc(post_msg.icore_tensor, post_results,
				// 	netinfo, conf, iou_thresh, MULTILABEL, fpga_nms, N_CLASS,
				// 	device, mask_channel, protoh, protow, _norm, real_out_channles, bbox_info_channel, mask_normratio);
				post_results = post_detpost_plin(post_msg.icore_tensor, post_results, netinfo, conf, iou_thresh, MULTILABEL, fpga_nms,
                	N_CLASS, ANCHORS,device,run_sim,_norm,real_out_channles,_stride,bbox_info_channel);
				std::vector<int> id_list = std::get<0>(post_results);
				std::vector<float> socre_list = std::get<1>(post_results);
				std::vector<cv::Rect2f> box_list = std::get<2>(post_results);
				
				
				// camera.get(display_data, camera_buf);
				camera.get(display_data, camera_buf_group[post_msg.buffer_index]);
				buffer_avaiable_flag[post_msg.buffer_index] = true;
				cv::Mat mat = cv::Mat(FRAME_H, FRAME_W, CV_8UC2, display_data);

				// auto dst_size = (CAMERA_W >= CAMERA_H) ? CAMERA_W : CAMERA_H;
				// auto mask_size = (protoh >= protow) ? protoh : protow;
				// auto ratio = (float)dst_size / (float)mask_size;
				// Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mask_single;
				// cv::Mat mask_all = cv::Mat::zeros(cv::Size(NET_W, NET_H), CV_8UC1);

				// for (int index = 0; index < box_list.size(); ++index) {

				// 	float x1 = box_list[index].tl().x;
				// 	float y1 = box_list[index].tl().y;
				// 	float w = box_list[index].width;
				// 	float h = box_list[index].height;

				// 	x1 = checkBorder(x1, 0.f, (float)NET_W);
				// 	y1 = checkBorder(y1, 0.f, (float)NET_H);
				// 	w = checkBorder(w, -x1, (float)(NET_W - x1));
				// 	h = checkBorder(h, -y1, (float)(NET_H - y1));
				// 	//mask
				// 	auto pic_size = protow * protoh;
				// 	// 取出一个 124 280 的 mask
				// 	Eigen::VectorXf v(pic_size);
				// 	v = mask_res.row(index);
				// 	mask_single = Eigen::Map<Eigen::MatrixXf>(v.data(), protow, protoh).transpose();
				// 	cv::Mat masks_1;
				// 	cv::eigen2cv(mask_single, masks_1);
				// 	cv::resize(masks_1, masks_1, cv::Size(protow * ratio / 2, protoh * ratio / 2));
				// 	//std::cout << masks_1.rows << "," << masks_1.cols << std::endl;

				// 	cv::Mat small_masks_3 = masks_1(cv::Range(y1, y1 + h), cv::Range(x1, x1 + w));
				// 	small_masks_3.copyTo(mask_all(cv::Range(y1, y1 + h), cv::Range(x1, x1 + w)));

				// }
				// // 有点粗粒度的resize 直接变为了1080p 实际上要按照不同的plresize方式去恢复到原1080p
				// cv::resize(mask_all, mask_all, cv::Size(1920, 1080));

				for (int index = 0; index < box_list.size(); ++index) {

					float x1 = box_list[index].tl().x * RATIO_W + BIAS_W;
					float y1 = box_list[index].tl().y * RATIO_H + BIAS_H;
					float w = box_list[index].width * RATIO_W;
					float h = box_list[index].height * RATIO_H;

					int id = id_list[index];
					cv::Scalar color = classColor(id);
					// cv::Scalar color_ = cv::Scalar(u(e), u(e), u(e));
					double font_scale = 1;
					int thickness = 1;
					// cv::rectangle(mask_all, cv::Rect(x1, y1, w, h), color, 6, cv::LINE_8, 0);
					cv::rectangle(mat, cv::Rect(x1, y1, w, h), color, 6, cv::LINE_8, 0);
					std::string s = LABELS[id_list[index]].substr(0, LABELS[id_list[index]].size() - 1) + ":" + std::to_string(int(round(socre_list[index] * 100))) + "%";
					cv::Size s_size = cv::getTextSize(s, cv::FONT_HERSHEY_COMPLEX, font_scale, thickness, 0);
					// cv::rectangle(mask_all, cv::Point(x1, y1 - s_size.height - 6), cv::Point(x1 + s_size.width, y1), color, -1);
					cv::rectangle(mat, cv::Point(x1, y1 - s_size.height - 6), cv::Point(x1 + s_size.width, y1), color, -1);
					// cv::putText(mask_all, s, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
					cv::putText(mat, s, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);

				}
				std::cout<<"fps ="<<fps<<std::endl;
				drawTextTwoConer(mat, fmt::format("FPS: {:.1f}", fps), MODEL_NAME, color);
				display.show(display_data);
				// cv::Mat out;
				// cvtColor(mat, out, cv::COLOR_GRAY2BGR565);
				// display.show((int8_t*)(out.data));

				frame_num++;
				if (frame_num == FPS_COUNT_NUM) {
					frame_num = 0;
					auto duration = std::chrono::duration_cast<microseconds>
						(std::chrono::steady_clock::now() - startfps) / FPS_COUNT_NUM;
					fps = 1000 / (float(duration.count()) / 1000);
					startfps = std::chrono::steady_clock::now();
				}
			}
		}
	);

	input_thread.join();
	icore_thread.join();
	post_thread.join();

	icore_task_queue->Stop();
	post_task_queue->Stop();
	Device::Close(device);
	return 0;
}
