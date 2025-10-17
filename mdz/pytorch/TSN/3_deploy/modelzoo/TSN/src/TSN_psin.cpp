
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
using namespace icraft::xrt;
using namespace icraft::xir;

//打印时间
// #define DEBUG_PRINT


int main(int argc, char* argv[])
{
	try {
		YAML::Node config = YAML::LoadFile(argv[1]);
		// icraft模型部署相关参数配置
		auto imodel = config["imodel"];
		// 仿真上板的jrpath配置
		std::string folderPath = imodel["dir"].as<std::string>();
		bool run_sim = imodel["sim"].as<bool>();
		std::string targetFileName;
		std::string JSON_PATH = getJrPath(run_sim, folderPath, imodel["stage"].as<std::string>());
		std::regex rgx3(".json");
		std::string RAW_PATH = std::regex_replace(JSON_PATH, rgx3, ".raw");

		// URL配置
		std::string ip = imodel["ip"].as<std::string>();
		// 可视化配置
		bool show = imodel["show"].as<bool>();
		bool save = imodel["save"].as<bool>();

		auto params = config["param"];
		int num_segments = params["num_segments"].as<int>();

		// 加载network
		Network network = loadNetwork(JSON_PATH, RAW_PATH);

		// 初始化netinfo
		NetInfo netinfo = NetInfo(network);
		//netinfo.ouput_allinfo();

		// 打开device
		bool mmu_open = false; //不能开启MMU
		//bool mmu_open = netinfo.mmu || imodel["mmu"].as<bool>(); //是否需要开启MMU
		Device device = openDevice(run_sim, ip, mmu_open);

		// 选择对网络进行切分
		auto network1_view = network.viewByOpId(272,484);//net1，去除最开始的input算子
		auto network2_view = network.viewByOpId(488,132);//net2，保留最后的cast算子等cpu算子

		// 初始化session
		Session session_1 = initSession(run_sim, network1_view, device, mmu_open, imodel["speedmode"].as<bool>(), imodel["compressFtmp"].as<bool>());
		Session session_2 = initSession(run_sim, network2_view, device, mmu_open, imodel["speedmode"].as<bool>(), imodel["compressFtmp"].as<bool>());

		auto buyi_backend_1 = session_1->backends[0].cast<BuyiBackend>();
		auto buyi_backend_2 = session_2->backends[0].cast<BuyiBackend>();


		// 开启计时功能
		session_1.enableTimeProfile(true);
		session_2.enableTimeProfile(true);

		// session执行前必须进行apply部署操作
		session_1.apply();
		session_2.apply();

		//buyi_backend_1.log();
		//buyi_backend_2.log();

		// 数据集相关参数配置
		auto dataset = config["dataset"];
		std::string imgRoot = dataset["dir"].as<std::string>();
		std::string imgList = dataset["list"].as<std::string>();
		std::string resRoot = dataset["res"].as<std::string>();
		std::string labelRoot = dataset["label_txt"].as<std::string>();
		checkDir(resRoot);

		// 计算net1输出和net2输入在PLDDR上的物理地址，用于搬数
		auto net1_output_segment = buyi_backend_1->phy_segment_map.at(Segment::OUTPUT);
		auto offset = net1_output_segment->byte_size;
		std::cout << "offset:" << offset << std::endl;
		auto src_base_addr = net1_output_segment->memchunk->begin.addr();
		auto src_end_addr = net1_output_segment->memchunk->begin.addr() + offset;
		auto net2_input_chunk = buyi_backend_2->phy_segment_map.at(Segment::INPUT)->memchunk;

		// net2 fake input
		icraft::xrt::TensorType input_type;
		if (netinfo.bit == 8){
			std::vector<int64_t> input_shape = { 1,num_segments,64,32 };
			auto tensor_layout = icraft::xir::Layout("**Cc32");
			input_type = icraft::xrt::TensorType(icraft::xir::IntegerType::SInt8(), input_shape, tensor_layout);
		}
		else{
			std::vector<int64_t> input_shape = { 1,num_segments,128,16 };
			auto tensor_layout = icraft::xir::Layout("**Cc16");
			input_type = icraft::xrt::TensorType(icraft::xir::IntegerType::SInt16(), input_shape, tensor_layout);
		}
		auto input_tensor = icraft::xrt::Tensor(input_type, net2_input_chunk);//构造PL tensor
		

		// 统计图片数量
		auto namevector = toVector(imgList);
		int totalnum = namevector.size();
		int window = num_segments;
		int num = totalnum / window; //动作识别次数
		int cur_img_index;
		std::string img_path;
		for (int i = 0; i < num; i++) {
			progress(i, num);
			// TSN-net1：对每一帧都进行网络前向
			//std::vector<Tensor> net2_inputs;
			int index = 0;
			while (index < window) {
				progress(index, window);
				auto one_start = std::chrono::high_resolution_clock::now();
				cur_img_index = i * window + index;
				img_path = imgRoot + '/' + namevector[cur_img_index];
				std::cout << "imgs: " << img_path << std::endl;
				auto one_read = std::chrono::high_resolution_clock::now();
				auto one_read_dura = std::chrono::duration_cast<std::chrono::microseconds>
					(one_read - one_start);
				// net1前处理
				PicPre img(img_path, cv::IMREAD_COLOR);
				img.Resize({ netinfo.i_cubic[0].h, netinfo.i_cubic[0].w }, PicPre::LONG_SIDE).rPad();
				//std::cout << img.src_img.size << std::endl;
				auto one_resize = std::chrono::high_resolution_clock::now();
				auto one_resize_dura = std::chrono::duration_cast<std::chrono::microseconds>
					(one_resize - one_read);
				Tensor img_tensor = CvMat2Tensor(img.dst_img, network);
				auto one_cvmax2tensor = std::chrono::high_resolution_clock::now();
				auto one_cvmax2tensor_dura = std::chrono::duration_cast<std::chrono::microseconds>
					(one_cvmax2tensor - one_resize);
				//std::cout << "IMK" << netinfo.ImageMake_on << std::endl;
				dmaInit(run_sim, netinfo.ImageMake_on, img_tensor, device);
				auto one_dmainit = std::chrono::high_resolution_clock::now();
				auto one_dmainit_dura = std::chrono::duration_cast<std::chrono::microseconds>
					(one_dmainit - one_cvmax2tensor);
				auto one_preprocess = std::chrono::high_resolution_clock::now();
				auto one_preprocess_dura = std::chrono::duration_cast<std::chrono::microseconds>
					(one_preprocess - one_start);

				auto net1_outputs = session_1.forward({ img_tensor });
				auto one_forward = std::chrono::high_resolution_clock::now();
				auto one_forward_dura = std::chrono::duration_cast<std::chrono::microseconds>
					(one_forward - one_preprocess);
				//std::cout << net1_outputs[0].dtype()->shape << std::endl;
				// 手动同步
				for (auto&& tensor : net1_outputs) {
					tensor.waitForReady(1000ms);
				}
				// 计时
				#ifdef __linux__
					device.reset(1);
					std::string network1_name = session_1->network_view.network()->name + "_net1";
					calctime_detail_ylm(session_1, network1_name);
				#endif

				//PLDDR->PLDDR搬数，将net1的输出直接搬移至net2的输入
				auto dest_base_addr = net2_input_chunk->begin.addr()+ offset * index;
				auto dest_end_addr = net2_input_chunk->begin.addr() + offset * (index + 1);
				PLDDRMemRegion::Plddr_memcpy(src_base_addr, src_end_addr, dest_base_addr, dest_end_addr, device);

				auto one_memcpy = std::chrono::high_resolution_clock::now();
				auto one_memcpy_dura = std::chrono::duration_cast<std::chrono::microseconds>
					(one_memcpy - one_forward);

				if (!run_sim) device.reset(1);

				index++;
				#ifdef DEBUG_PRINT
						spdlog::info("[Net1 Preprocess] read={:.2f}ms, resize={:.2f}ms, cvmax2tensor={:.2f}ms, dmainit={:.2f}ms, preprocess total={:.2f}ms,",
							float(one_read_dura.count()) / 1000,
							float(one_resize_dura.count()) / 1000,
							float(one_cvmax2tensor_dura.count()) / 1000,
							float(one_dmainit_dura.count()) / 1000,
							float(one_preprocess_dura.count()) / 1000

						);
						spdlog::info("[Net1] preprocess={:.2f}ms, forward={:.2f}ms, PLDDR memcpy={:.2f}ms,",
							float(one_preprocess_dura.count()) / 1000,
							float(one_forward_dura.count()) / 1000,
							float(one_memcpy_dura.count()) / 1000
						);
				#endif


			}
			// net2：cls_head前向
			auto net2_start = std::chrono::high_resolution_clock::now();
			auto net2_outputs = session_2.forward({ input_tensor });
			auto net2_forward = std::chrono::high_resolution_clock::now();
			auto net2_forward_dura = std::chrono::duration_cast<std::chrono::microseconds>
				(net2_forward - net2_start);
			std::cout << net2_outputs[0].dtype()->shape << std::endl;
			// 手动同步
			for (auto&& tensor : net2_outputs) {
				tensor.waitForReady(1000ms);
			}
			// 计时
			#ifdef __linux__
				device.reset(1);
				std::string network2_name = session_2->network_view.network()->name + "_net2";
				calctime_detail_ylm(session_2, network2_name);
			#endif
			//net2后处理
			postprocess_tsn(net2_outputs, show, save, labelRoot, resRoot, img_path, namevector[cur_img_index]);
			auto net2_postprocess = std::chrono::high_resolution_clock::now();
			auto net2_postprocess_dura = std::chrono::duration_cast<std::chrono::microseconds>
				(net2_postprocess - net2_forward);
			#ifdef DEBUG_PRINT
						spdlog::info("[Net2] forward={:.2f}ms, postprocess total={:.2f}ms,",
							float(net2_forward_dura.count()) / 1000,
							float(net2_postprocess_dura.count()) / 1000
						);
			#endif
		}
		//关闭设备
		Device::Close(device);
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}

    return 0;
}
