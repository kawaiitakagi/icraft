
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/buyi_device.h>
#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "postprocess_tsn.hpp"
#include "icraft_utils.hpp"
#include "yaml-cpp/yaml.h"
#include <task_queue.hpp>
#include <et_device.hpp>
using namespace icraft::xrt;
using namespace icraft::xir;

//打印时间
//#define DEBUG_PRINT

// 摄像头输入尺寸
const int CAMERA_W = 1920;
const int CAMERA_H = 1080;

// ps端图像尺寸
const int FRAME_W = 1920;
const int FRAME_H = 1080;

// 网络的json&raw文件路径
auto net_1_json_file = "../imodel/plin_TSN_25frames/plin_TSN_25frames_BY.json";
auto net_1_raw_file = "../imodel/plin_TSN_25frames/plin_TSN_25frames_BY.raw"; //行为识别

int num_segments = 25;
int CAM_FPS = 60; //摄像头帧率
int FREE_THR = 360; //回收chunk的帧数阈值
const std::string MODEL_NAME = "TSN";
bool speedmode = true;
bool compress_ftmp = true;
bool save = false; //是否保存结果
bool show = true; //是否显示结果
const std::string resRoot = "../io/track/";
const std::string labelRoot = "../io/kinetics_label_map_k400.txt";


int main(int argc, char* argv[])
{
	try
	{
		auto thread_num = 4;

		// 打开device
		auto device = Device::Open("axi://ql100aiu?npu=0x40000000&dma=0x80000000");
		auto buyi_device = device.cast<BuyiDevice>();

		//关闭MMU
		buyi_device.mmuModeSwitch(false);

		// 模型相关参数
		int font_face = cv::FONT_HERSHEY_COMPLEX;
		int thickness = 1;
		double font_scale = 1;

		// 配置摄像头
		uint64_t BUFFER_SIZE = FRAME_H * FRAME_W * 2;
		Camera camera(buyi_device, BUFFER_SIZE);

		// 在psddr-udmabuf上申请摄像头图像缓存区
		auto camera_buf_group = std::vector<MemChunk>(thread_num);
		for (int i = 0; i < thread_num; i++) {
			auto camera_buf = buyi_device.getMemRegion("udma").malloc(BUFFER_SIZE, false);
			std::cout << "Cam buffer udma addr=" << i << " ,udma addr=" << camera_buf->begin.addr() << '\n';
			camera_buf_group[i] = camera_buf;
		}

		// 同样在 udmabuf上申请display缓存区
		const uint64_t DISPLAY_BUFFER_SIZE = FRAME_H * FRAME_W * 2;    // 摄像头输入为RGB565
		auto display_chunck = buyi_device.getMemRegion("udma").malloc(DISPLAY_BUFFER_SIZE, false);
		auto display = Display_pHDMI_RGB565(buyi_device, DISPLAY_BUFFER_SIZE, display_chunck);
		std::cout << "Display buffer udma addr=" << display_chunck->begin.addr() << '\n';


		// 加载网络
		auto network_1 = loadNetwork(net_1_json_file, net_1_raw_file);

		// 初始化netinfo
		NetInfo net1_netinfo = NetInfo(network_1);
		//netinfo.ouput_allinfo();

		// 将网络拆分为imagemake和icore
		auto image_make = network_1.view(1,2); //imk
		auto tsn_net1 = network_1.viewByOpId(413, 618);//去除input&imk,切分出TSN-net1
		auto tsn_net2 = network_1.viewByOpId(622, 132);//切分出TSN-net2，保留最后的cast算子等cpu算子


		// 在PLDDR上申请imagemake缓存区，用来缓存给AI计算的图片
		const uint64_t imk_output_size = FRAME_H * FRAME_W * 4;
		auto imagemake_buf_group = std::vector<MemChunk>(thread_num);
		for (int i = 0; i < thread_num; i++) {
			auto imk_chunck = buyi_device.getMemRegion("plddr").malloc(imk_output_size, false);
			std::cout << "image make buffer index:" << i << " ,plddr addr=" << imk_chunck->begin.addr() << '\n';
			imagemake_buf_group[i] = imk_chunck;
		}

		// 构建多个session
		auto imk_sessions = std::vector<Session>(thread_num);
		auto tsn_net1_sessions = std::vector<Session>(thread_num);
		auto tsn_net2_sessions = std::vector<Session>(thread_num);

		for (int i = 0; i < thread_num; i++) {
			imk_sessions[i] = Session::Create<BuyiBackend, HostBackend>(image_make, { buyi_device, HostDevice::Default() });
			tsn_net1_sessions[i] = Session::Create<BuyiBackend, HostBackend>(tsn_net1, { buyi_device, HostDevice::Default() });
			tsn_net2_sessions[i] = Session::Create<BuyiBackend, HostBackend>(tsn_net2, { buyi_device, HostDevice::Default() });


			auto imk_buyi_backend = imk_sessions[i]->backends[0].cast<BuyiBackend>();
			auto tsn_net1_buyi_backend = tsn_net1_sessions[i]->backends[0].cast<BuyiBackend>();
			auto tsn_net2_buyi_backend = tsn_net2_sessions[i]->backends[0].cast<BuyiBackend>();
			
			// 将同一组imagemake和icore的输入输出连接起来
			imk_buyi_backend.userSetSegment(imagemake_buf_group[i], Segment::OUTPUT);
			tsn_net1_buyi_backend.userSetSegment(imagemake_buf_group[i], Segment::INPUT);

			if (speedmode) {
				tsn_net1_buyi_backend.speedMode();
				tsn_net2_buyi_backend.speedMode();
			}
			tsn_net1_buyi_backend.compressFtmp();
			tsn_net2_buyi_backend.compressFtmp();

			// 打开计时接口
			//tsn_net1_sessions[i].enableTimeProfile(true);
			//tsn_net2_sessions[i].enableTimeProfile(true);

			imk_sessions[i].apply();
			tsn_net1_sessions[i].apply();
			tsn_net2_sessions[i].apply();

			// 打印log
			//tsn_net1_buyi_backend.log();
			//tsn_net2_buyi_backend.log();
		}

		// 计算TSN-net1输出在PLDDR上的物理地址，用于搬数
		auto tsn_net1_src_base_addrs = std::vector<uint64_t>(thread_num);
		auto tsn_net1_src_end_addrs = std::vector<uint64_t>(thread_num);
		uint64_t tsn_offset;
		for (int i = 0; i < thread_num; i++) {
			auto buyi_backend_tsn_net1 = tsn_net1_sessions[i]->backends[0].cast<BuyiBackend>();
			auto tsn_net1_output_segment = buyi_backend_tsn_net1->phy_segment_map.at(Segment::OUTPUT);
			tsn_offset = tsn_net1_output_segment->byte_size;
			std::cout << "tsn_offset:" << tsn_offset << std::endl;
			auto tsn_net1_src_base_addr = tsn_net1_output_segment->memchunk->begin.addr();
			auto tsn_net1_src_end_addr = tsn_net1_output_segment->memchunk->begin.addr() + tsn_offset;
			tsn_net1_src_base_addrs[i] = tsn_net1_src_base_addr;
			tsn_net1_src_end_addrs[i] = tsn_net1_src_end_addr;
		}
		auto tsn_net2_input_size = tsn_net2_sessions[0]->backends[0].cast<BuyiBackend>()->phy_segment_map.at(Segment::INPUT)->byte_size;


		// imk fake input
		std::vector<int64_t> output_shape = { 1, CAMERA_H, CAMERA_W,  3 };
		auto tensor_layout = icraft::xir::Layout("NHWC");
		auto output_type = icraft::xrt::TensorType(icraft::xir::IntegerType::UInt8(), output_shape, tensor_layout);
		auto output_tensor = icraft::xrt::Tensor(output_type).mallocOn(icraft::xrt::HostDevice::MemRegion());
		auto img_tensor_list = std::vector<Tensor>{ output_tensor };

		// TSN-net2 fake input
		auto bits = net1_netinfo.ImageMake_->outputs[0]->dtype.getStorageType().bits();
		std::cout << "bits: " << bits << std::endl;
		icraft::xrt::TensorType tsn_net2_input_type;
		if (bits == 8) {
			std::vector<int64_t> tsn_net2_input_shape = { 1,num_segments,64,32 };
			auto tsn_net2_tensor_layout = icraft::xir::Layout("**Cc32");
			tsn_net2_input_type = icraft::xrt::TensorType(icraft::xir::IntegerType::SInt8(), tsn_net2_input_shape, tsn_net2_tensor_layout);
		}
		else {
			std::vector<int64_t> tsn_net2_input_shape = { 1,num_segments,128,16 };
			auto tsn_net2_tensor_layout = icraft::xir::Layout("**Cc16");
			tsn_net2_input_type = icraft::xrt::TensorType(icraft::xir::IntegerType::SInt16(), tsn_net2_input_shape, tsn_net2_tensor_layout);
		}

		auto tsn_net2_input_tensors = std::vector<icraft::xrt::Tensor>(thread_num);
		auto tsn_net2_memchunks = std::vector<MemChunk>(thread_num);
		for (int i = 0; i < thread_num; i++) {
			auto tsn_net2_buyibackend = tsn_net2_sessions[i]->backends[0].cast<BuyiBackend>();
			auto net2_input_chunk = tsn_net2_buyibackend->phy_segment_map.at(Segment::INPUT)->memchunk;
			tsn_net2_memchunks[i] = net2_input_chunk;
			auto tsn_net2_input_tensor = icraft::xrt::Tensor(tsn_net2_input_type, net2_input_chunk);//构造PL tensor
			tsn_net2_input_tensors[i] = tsn_net2_input_tensor;
		}

		// PL端输入必须经过hardResizePL，此处不做任何尺度变换
		auto ratio_bias = preprocess_plin(buyi_device, CAMERA_W, CAMERA_H, CAMERA_W, CAMERA_H, crop_position::center);
		Operation WarpAffine_net1 = net1_netinfo.WarpAffine_;
		auto WarpAffine_M_inversed = WarpAffine_net1->attrs().at("M_inversed").cast<Array<Array<FloatImm>>>();
		float RATIO_W = WarpAffine_M_inversed[0][0];
		float RATIO_H = WarpAffine_M_inversed[1][1];
		int BIAS_W = WarpAffine_M_inversed[0][2];
		int BIAS_H = WarpAffine_M_inversed[1][2];
		std::cout << " RATIO_W:" << RATIO_W << " RATIO_H: " << RATIO_H << " BIAS_W: " << BIAS_W << " BIAS_H:" << BIAS_H << std::endl;


		auto progress_printer = std::make_shared<ProgressPrinter>(1);
		auto FPS_COUNT_NUM = 60;
		auto color = cv::Scalar(128, 128, 128);
		std::atomic<uint64_t> frame_num = 1;
		std::atomic<float> fps = 0.f;

		int8_t* display_data = new int8_t[FRAME_W * FRAME_H * 2];
		bool init_onetime = true;
		std::vector<float> target_pos;
		std::vector<float> target_sz;
		int t_size = net1_netinfo.head_hardop_i_shape[0][2]; // 224
		std::cout << "t_size:" << t_size << std::endl;
		int window_length = num_segments; //动作识别的窗口大小

		// 初始化任务队列
		auto icore_task_queue = std::make_shared<Queue<InputMessageForIcore>>(thread_num);
		auto post_task_queue = std::make_shared<Queue<IcoreMessageForPost>>(thread_num);
		std::vector<bool> buffer_avaiable_flag(thread_num, true);
		std::vector<bool> post_end_flag(thread_num, false);
		std::mutex det_post_mutex;

		auto startfps = std::chrono::steady_clock::now();
		// 线程1：camera->imk取帧
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
					while (!buffer_avaiable_flag[buffer_index]) {
						usleep(0);
					}

					camera.take(camera_buf_group[buffer_index]);

					try {
						msg.image_tensor = imk_sessions[buffer_index].forward(img_tensor_list);//imk前向
					}
					catch (const std::exception& e) {
						msg.error_frame = true;
						icore_task_queue->Push(msg);
						continue;
					}


					if (!camera.wait()) {
						msg.error_frame = true;
						icore_task_queue->Push(msg);
						continue;
					}
					// 将buffer标记为不可用，等后处理完成后再释放
					buffer_avaiable_flag[buffer_index] = false;

					icore_task_queue->Push(msg);

					buffer_index++;
					buffer_index = buffer_index % camera_buf_group.size();
				}
			}
		);

		// 线程2：bytetrack&tsn的icore前向
		auto icore_thread = std::thread(
			[&]() {
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
				cv::Mat mat;
				if (show) {
					camera.get(display_data, camera_buf_group[input_msg.buffer_index]);
					mat = cv::Mat(FRAME_H, FRAME_W, CV_8UC2, display_data);
				}
				std::cout << "==== TSN net1====" << std::endl;
				// tsn_net1：前向推理
				auto tsn_net1_outputs = tsn_net1_sessions[input_msg.buffer_index].forward({ input_msg.image_tensor });
				// 手动同步
				for (auto&& tensor : tsn_net1_outputs) {
					if (!tensor.waitForReady(5000ms))
					{
						std::cout << "timeout!!!!!!!" << std::endl;
					}
				}
				device.reset(1);
				int index = (frame_num - 1) % window_length + 1;
				//std::cout << "#index:" << index << std::endl;
				//PLDDR->PLDDR搬数，将net1的输出的数据搬移至net2的输入chunk
				auto tsn_net2_input_chunk = tsn_net2_memchunks[input_msg.buffer_index];
				auto tsn_net2_dest_base_addr = tsn_net2_input_chunk->begin.addr() + tsn_offset * (index - 1);;
				auto tsn_net2_dest_end_addr = tsn_net2_input_chunk->begin.addr() + tsn_offset * index;
				PLDDRMemRegion::Plddr_memcpy(tsn_net1_src_base_addrs[input_msg.buffer_index], tsn_net1_src_end_addrs[input_msg.buffer_index], tsn_net2_dest_base_addr, tsn_net2_dest_end_addr, device);

				if (frame_num >= num_segments) {
					std::cout << "==== TSN net2====" << std::endl;
					// tsn_net2：前向推理
					auto tsn_net2_outputs = tsn_net2_sessions[input_msg.buffer_index].forward({ tsn_net2_input_tensors[input_msg.buffer_index] });
					std::cout << tsn_net2_outputs[0].dtype()->shape << std::endl;
					// 手动同步
					for (auto&& tensor : tsn_net2_outputs) {
						if (!tensor.waitForReady(5000ms))
						{
							std::cout << "timeout!!!!!!" << std::endl;
						}
					}
					device.reset(1);
					//tsn-net2后处理
					std::string top1_label = plin_tsn_postprocess(tsn_net2_outputs, labelRoot);

					if (show) {
						std::string s = std::string("label: ") + top1_label;
						auto s_size = cv::getTextSize(s, cv::FONT_HERSHEY_COMPLEX, font_scale, thickness, 0);
						cv::putText(mat, s, cv::Point(120, 40), cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
					}

				}
				else {
					if (show) {
						std::string s = std::string("label: None ");
						auto s_size = cv::getTextSize(s, cv::FONT_HERSHEY_COMPLEX, font_scale, thickness, 0);
						cv::putText(mat, s, cv::Point(120, 40), cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
					}
				}

				//-------------------------------------//
				//       帧数计算
				//-------------------------------------//
				std::cout << "frame_num:" << frame_num << std::endl;
				frame_num++;
				if (frame_num % FPS_COUNT_NUM == 0) {
					auto duration = std::chrono::duration_cast<microseconds>
						(std::chrono::steady_clock::now() - startfps) / FPS_COUNT_NUM;
					fps = 1000 / (float(duration.count()) / 1000);
					startfps = std::chrono::steady_clock::now();
				}
				std::cout << "FPS:" << fps << std::endl;
				if (show) {
					drawTextTwoConer(mat, fmt::format("FPS: {:.1f}", fps), MODEL_NAME, cv::Scalar(255, 255, 255));
					display.show(display_data);
				}
				// 后处理结束，将buffer标记为可用
				buffer_avaiable_flag[input_msg.buffer_index] = true;
			}
			}
		);

		input_thread.join();
		icore_thread.join();

		icore_task_queue->Stop();
		Device::Close(device);
	}
	catch (const std::exception& e)
	{
	std::cout << e.what() << std::endl;
	}

	return 0;

}
