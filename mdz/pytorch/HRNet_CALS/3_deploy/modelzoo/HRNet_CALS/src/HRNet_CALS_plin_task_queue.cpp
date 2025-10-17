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
#include "et_device.hpp"
#include <task_queue.hpp>
#include <NetInfo.hpp>
using namespace icraft::xrt;
using namespace icraft::xir;
struct MessageForPost
{
    int buffer_index;   //
    std::vector<icraft::xrt::Tensor> icore_tensor;
    bool ai;
    bool error_frame = false;
    uint64_t arbase;
    uint64_t last_araddr;
};	
template <typename T>
void process_tensor_data_soft(std::vector<float> &x_list, std::vector<float> &y_list,T* tensor_data, int csize, int valid_csize, int hsize, int wsize) {
    std::vector<int> ind(valid_csize);

    for (int channel = 0; channel < valid_csize; ++channel) {
        float max_value = std::numeric_limits<float>::lowest();
        int max_index = -1;

        for (int spatial_idx = 0; spatial_idx < hsize * wsize; ++spatial_idx) {
            float current_value = tensor_data[channel + spatial_idx * csize];
            if (current_value > max_value) {
                max_value = current_value;
                max_index = spatial_idx;
            }
        }

        ind[channel] = max_index;
    }

    for (int i = 0; i < valid_csize; i++) {
        x_list.emplace_back(ind[i] % hsize);
        y_list.emplace_back(ind[i] / hsize);
    }
	// for (int i = 0; i < valid_csize; i++) {
	//  	std::cout<<"x="<<x_list[i]<<"y="<<y_list[i]<<std::endl;
    // }
}
template <typename T>
void process_tensor_data_hard(std::vector<float> &x_list, std::vector<float> &y_list,T* tensor_data, int csize, int valid_csize, int hsize, int wsize, 
                         Device& device,uint64_t arbase,uint64_t last_araddr) {
    std::vector<int> ind(valid_csize);
    //use fpga_argmax2d_forward_buyibackend
	Tensor index_tensor = fpgaArgmax2d(device,wsize,hsize,valid_csize,csize,arbase,last_araddr);
    auto host_tensor = index_tensor.to(HostDevice::MemRegion());//做 to ps操作
	auto index_data = (uint64_t*)index_tensor.data().cptr();
	for (int i = 0; i < valid_csize; i++) {
        x_list.emplace_back(index_data[i] % wsize);
        y_list.emplace_back(index_data[i] / wsize);
		//std::cout << (uint64_t)index_data[i] << std::endl;
    }
    //check argmax result
	// for (int i = 0; i < valid_csize; i++) {
	// 	std::cout<<"x="<<x_list[i]<<"y="<<y_list[i]<<std::endl;
    // }
}
// smooth
const float ALPHA = 0.5f;
const float SMOOTH_IOU = 0.80f;
using PostResult = std::tuple<std::vector<float>, std::vector<float>>;// x_list, y_list
PostResult post_process_plin(const std::vector<Tensor>& output_tensors, PostResult& last_frame_result,
	Device& device,uint64_t arbase,uint64_t last_araddr,bool fpga_argmax){
	
	auto host_tensor = output_tensors[0].to(HostDevice::MemRegion());// from PL to ps.hostmem region
	int output_tensors_bits = output_tensors[0].dtype()->element_dtype.getStorageType().bits(); //获取位数：8bit or 16bit量化
	int valid_csize = 22;//22为有效通道
	// int ori_hsize = img.src_img.rows;//1080为原图大小
	// int ori_hsize = 1080;//1080为原图大小
	std::vector<int> ind(valid_csize);//argmax结果存放
	std::vector<float> x_list;
	std::vector<float> y_list;
	switch(output_tensors_bits){
		case 8:{
			
			auto tensor_data = (int8_t*)host_tensor.data().cptr(); // pay attention to the data format float or int8_t
			// ------------------POST_PROCESS : reshape + argmax-----------------	
			int hsize = output_tensors[0].dtype()->shape[2]; //320 hsize = wsize 
			int wsize = output_tensors[0].dtype()->shape[3]; //320
			int csize = output_tensors[0].dtype()->shape[4]; //22 or 32
			// std::cout <<" hsize="<<hsize<<" wsize="<<wsize<<" csize="<<csize<<std::endl;
			if(fpga_argmax){
				process_tensor_data_hard(x_list,y_list,(int8_t*)tensor_data,csize,valid_csize,hsize,wsize,device,arbase,last_araddr);
			}else{
				process_tensor_data_soft(x_list,y_list,(int8_t*)tensor_data,csize,valid_csize,hsize,wsize);
			}
			break;
		}
		case 16:{
			auto tensor_data = (int16_t*)host_tensor.data().cptr(); // pay attention to the data format float or int16_t
			
			// ------------------POST_PROCESS : reshape + argmax-----------------	
			int hsize = output_tensors[0].dtype()->shape[2]; //320 hsize = wsize 
			int wsize = output_tensors[0].dtype()->shape[3]; //320
			int csize = output_tensors[0].dtype()->shape[4]; //22 or 32 now it's 16!
			if(fpga_argmax){
				process_tensor_data_hard(x_list,y_list,(int16_t*)tensor_data,csize,valid_csize,hsize,wsize,device,arbase,last_araddr);
			}else{
				process_tensor_data_soft(x_list,y_list,(int16_t*)tensor_data,csize,valid_csize,hsize,wsize);
			}
			

			break;
		}
		default: {
				throw "wrong bits num!";
				exit(EXIT_FAILURE);
		}
	}

	return PostResult { x_list, y_list};
}
int main(int argc, char* argv[])
{
	auto thread_num = 4;
	YAML::Node config = YAML::LoadFile(argv[1]);
	// icraft模型部署相关参数配置
	auto imodel = config["imodel"];
	// 仿真上板的jrpath配置
	std::string folderPath = imodel["dir"].as<std::string>();  
	
	std::string targetFileName;
	std::string JSON_PATH = getJrPath(false,folderPath, imodel["stage"].as<std::string>());
	std::regex rgx3(".json");
	std::string RAW_PATH = std::regex_replace(JSON_PATH, rgx3, ".raw");
	// std::string IMODEL_PATH = std::regex_replace(JSON_PATH, rgx3, ".imodel");
	
	// URL配置
	std::string ip = imodel["ip"].as<std::string>();
	// 可视化配置
	bool run_sim = imodel["sim"].as<bool>();
	// 模型自身相关参数配置
	auto param = config["param"];
	bool fpga_argmax = param["fpga_argmax"].as<bool>();

	// 打开device
	Device device = openDevice(false, "");
	// 关闭mmu模式
	device.cast<BuyiDevice>().mmuModeSwitch(false);

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
	Network network = loadNetwork(JSON_PATH, RAW_PATH); //以load j&r file创建网络
	// auto network = Network::CreateFromMSGFile(IMODEL_PATH);//以load imodel file创建网络,会快一点
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

	// 选择对网络进行切分 仅保留op_id = {2,66}的算子，目的是将cast/Align切分出网络 // auto network_view = network.view(2,66); 
	// 将网络拆分为image make和icore
	auto image_make = network.view(netinfo.inp_shape_opid + 1, netinfo.inp_shape_opid + 2);
	auto icore = network.view(netinfo.inp_shape_opid + 2,-3);
	// 计算复网络的ftmp大小，用于复用相同网络的ftmp
	auto icore_dummy_session = Session::Create<BuyiBackend, HostBackend>(icore, { buyi_device, HostDevice::Default() });
	auto& icore_dummy_backends = icore_dummy_session->backends;
	auto icore_buyi_dummy_backends = icore_dummy_backends[0].cast<BuyiBackend>();

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
		std::cout << "Presentation forward operator ...." << std::endl;
		auto ops = icore_sessions[i].getForwards();
		for (auto&& op : ops) {
			std::cout << "op name:" << std::get<0>(op)->typeKey() << '\n';
		}
		
	}
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
	PostResult post_result;

	auto icore_task_queue = std::make_shared<Queue<InputMessageForIcore>>(thread_num);
	auto post_task_queue = std::make_shared<Queue<MessageForPost>>(thread_num);
	// PL端的resize，需要resize到AI神经网络的尺寸
	// center crop to 1080x1080
	int x0 = 420;
	int y0 = 0;
	int x1 = 1499;
	int y1 = 1079;
	uint64_t base_addr = 0x40080000;
	// hardResizePL to 540x540
	hardResizePL(buyi_device, x0, y0, x1, y1, 2, 2, CAMERA_W, CAMERA_H, base_addr);
	float RATIO_W = 2;
	float RATIO_H = 2;
	int BIAS_W = 420;
	int BIAS_H = 0;
	std::cout<<"NET_W ="<<NET_W<<"NET_H ="<<NET_H<<"RATIO_W ="<<RATIO_W<<" RATIO_H ="<<RATIO_H<<" BIAS_W ="<<BIAS_W<<" BIAS_H ="<<BIAS_H<<std::endl;
	// RATIO_W,RATIO_H,BIAS_W,BIAS_H用于神经网络结果的坐标转换
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

				MessageForPost post_msg;
				post_msg.buffer_index = input_msg.buffer_index;
				post_msg.error_frame = input_msg.error_frame;

				if (input_msg.error_frame) {
					post_task_queue->Push(post_msg);
					continue;
				}

				post_msg.icore_tensor
					= icore_sessions[input_msg.buffer_index].forward(input_msg.image_tensor);
				// 获取arbase和last_araddr,代表output tensor在plddr的起始地址和最后一层 ftmp 的地址 for hard argmax2d
				uint64_t arbase;
				uint64_t last_araddr;
				for (auto& backend : icore_sessions[input_msg.buffer_index]->backends) {
					if (backend.is<BuyiBackend>()) {
						arbase = backend.cast<BuyiBackend>()->phy_segment_map.at(Segment::OUTPUT)->phy_addr;
						// auto last_arbaseadd = arbase +　backend.cast<BuyiBackend>()->phy_segment_map.at(Segment::OUTPUT);
						last_araddr = arbase + backend.cast<BuyiBackend>()->phy_segment_map.at(Segment::OUTPUT)->byte_size;
					}
				}
				post_msg.arbase = arbase;
				post_msg.last_araddr = last_araddr;
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
				MessageForPost post_msg;
				post_task_queue->Pop(post_msg);

				if (post_msg.error_frame) {
					cv::Mat display_mat = cv::Mat::zeros(FRAME_W, FRAME_H, CV_8UC2);
					drawTextTopLeft(display_mat, fmt::format("No input , Please check camera."), cv::Scalar(127, 127, 127));
					display.show(reinterpret_cast<int8_t*>(display_mat.data));
					continue;
				}
				uint64_t arbase;
				uint64_t last_araddr;
				arbase = post_msg.arbase;
				last_araddr = post_msg.last_araddr;
				post_result = post_process_plin(post_msg.icore_tensor, post_result,device,arbase,last_araddr,fpga_argmax);
				std::vector<float> x_list = std::get<0>(post_result);
				std::vector<float> y_list = std::get<1>(post_result);
				
				camera.get(display_data, camera_buf_group[post_msg.buffer_index]);
				buffer_avaiable_flag[post_msg.buffer_index] = true;
				cv::Mat mat = cv::Mat(FRAME_H, FRAME_W, CV_8UC2, display_data);
				// check results
				for (int index = 0; index < x_list.size(); ++index){
					//rescale results 
					float x = x_list[index] * 2* RATIO_W + BIAS_W;
					float y = y_list[index] * 2* RATIO_H + BIAS_H;
					cv::Scalar color = classColor(index);
					double font_scale = 1;
                	int thickness = 1;
					cv::circle(mat, cv::Point(x, y), 1, cv::Scalar(0, 255, 0), 2);
				}

				std::cout<<"fps ="<<fps<<std::endl;
				drawTextTwoConer(mat, fmt::format("FPS: {:.1f}", fps), MODEL_NAME, color);
				display.show(display_data);

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
