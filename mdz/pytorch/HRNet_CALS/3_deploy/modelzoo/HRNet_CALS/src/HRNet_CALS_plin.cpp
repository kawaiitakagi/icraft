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
#include <et_device.hpp>
#include "yaml-cpp/yaml.h"
#include <NetInfo.hpp>
using namespace icraft::xrt;
using namespace icraft::xir;
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

int main(int argc, char* argv[]) {
	try
	{
        YAML::Node config = YAML::LoadFile(argv[1]);
        // icraft模型部署相关参数配置
        auto imodel = config["imodel"];
        
        // 仿真上板的jrpath配置
        std::string folderPath = imodel["dir"].as<std::string>();
        std::string targetFileName;
        std::string JSON_PATH = getJrPath(false, folderPath, imodel["stage"].as<std::string>());
        std::regex rgx3(".json");
        std::string RAW_PATH = std::regex_replace(JSON_PATH, rgx3, ".raw");
        std::cout << "as" << std::endl;

        // 打开device
        Device device = openDevice(false, "", false);
        auto buyi_device = device.cast<BuyiDevice>();

        //-------------------------------------//
        //       加载标签
        //-------------------------------------//
        auto dataset = config["dataset"];
        

        // 模型自身相关参数配置
	    auto param = config["param"];
	    bool fpga_argmax = param["fpga_argmax"].as<bool>();

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
        // 而是在psddr-udmabuf申请了缓存区，进而 take get 都需要指定缓存区。
        auto camera_buf = buyi_device.getMemRegion("udma").malloc(BUFFER_SIZE, false);
        std::cout << "Cam buffer udma addr=" << camera_buf->begin.addr() << '\n';

        // 同样在psddr-udmabuf上申请display缓存区
        const uint64_t DISPLAY_BUFFER_SIZE = FRAME_H * FRAME_W * 2;    // 摄像头输入为RGB565
        auto display_chunck = buyi_device.getMemRegion("udma").malloc(DISPLAY_BUFFER_SIZE, false);
        auto display = Display_pHDMI_RGB565(buyi_device, DISPLAY_BUFFER_SIZE, display_chunck);
        std::cout << "Display buffer udma addr=" << display_chunck->begin.addr() << '\n';


        // 加载network
        Network network = loadNetwork(JSON_PATH, RAW_PATH);
        //初始化netinfo
        NetInfo netinfo = NetInfo(network);
        // 选择对网络进行切分
        // auto network_view = network.view(netinfo.inp_shape_opid + 1);
        // 选择对网络进行切分 仅保留op_id = {2,-3}的算子，目的是将cast/Align切分出网络
        auto network_view = network.view(2,-3);
        // 初始化session
        Session session = initSession(false, network_view, device, false, true, true);
        const std::string MODEL_NAME = session->network_view.network()->name;
        session.apply();
        //获取arbase和last_araddr,代表output tensor在plddr的起始地址和最后一层 ftmp 的地址 for hard argmax2d
		uint64_t arbase;
		uint64_t last_araddr;
		for (auto& backend : session->backends) {
			if (backend.is<BuyiBackend>()) {
				arbase = backend.cast<BuyiBackend>()->phy_segment_map.at(Segment::OUTPUT)->phy_addr;
				// auto last_arbaseadd = arbase +　backend.cast<BuyiBackend>()->phy_segment_map.at(Segment::OUTPUT);
				last_araddr = arbase + backend.cast<BuyiBackend>()->phy_segment_map.at(Segment::OUTPUT)->byte_size;
			}
		}
        std::cout << "Presentation forward operator ...." << std::endl;
        auto ops = session.getForwards();
        for (auto&& op : ops) {
            std::cout << "op name:" << std::get<0>(op)->typeKey() << '\n';
        }
        // PL端图像尺寸，即神经网络网络输入图片尺寸
        int NET_W = netinfo.i_cubic[0].w;
        int NET_H = netinfo.i_cubic[0].h;



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


        // PL端的resize，需要resize到AI神经网络的尺寸
        auto ratio_bias = preprocess_plin(buyi_device, CAMERA_W, CAMERA_H, NET_W, NET_H, crop_position::center);

        // 用于神经网络结果的坐标转换
        float RATIO_W = std::get<0>(ratio_bias);
        float RATIO_H = std::get<1>(ratio_bias);
        int BIAS_W = std::get<2>(ratio_bias);
        int BIAS_H = std::get<3>(ratio_bias);

        int8_t* display_data = new int8_t[FRAME_W * FRAME_H * 2];
        while (true) {
            //-------------------------------------//
            //       取一帧数图像 推理
            //-------------------------------------//
            camera.take(camera_buf);  // 抓取一帧，传到psddr-udmabuf空间上camera_buf处
                                      // 同时启动imk，将PL_resize处理后图像送入PLDDR中
            auto output_tensors = session.forward(img_tensor_list);

            device.reset(1);
            camera.wait();


            //-------------------------------------//
            //       后处理
            //-------------------------------------//
            post_result = post_process_plin(output_tensors,post_result,device,arbase,last_araddr,fpga_argmax);
            std::vector<float> x_list = std::get<0>(post_result);
			std::vector<float> y_list = std::get<1>(post_result);

            camera.get(display_data, camera_buf);  // 将psddr-udmabuf空间camera_buf上数据搬到PSDDR
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
            //如果没有显示器，可以选择存图查看结果是否正确(打开下面三句代码)，但这样会降低fps
            //cv::Mat out;
            //cvtColor(mat, out, cv::COLOR_BGR5652BGR);
            //cv::imwrite("../images/output/_thread_result.jpg", out);

            //-------------------------------------//
            //       帧数计算
            //-------------------------------------//
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
	catch (const std::exception& e)
	{
        std::cout << e.what() << std::endl;
	}


    return 0;
}
