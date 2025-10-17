
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
using namespace icraft::xrt;
using namespace icraft::xir;

template <typename T>
void process_tensor_data_soft(T* tensor_data, int csize, int valid_csize, int hsize, int wsize, 
                         bool save, const std::string& resRoot, const std::string& name, 
                         int ori_hsize, bool show, const cv::Mat& img) {
    std::vector<int> ind(valid_csize);
    std::vector<float> x_list, y_list;
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
    for (int i = 0; i < valid_csize; i++) {
        cv::circle(img, cv::Point(x_list[i] * ori_hsize / hsize, y_list[i] * ori_hsize / hsize), 1, cv::Scalar(0, 255, 0), 2);
    }

    if (save) {
        std::string save_path = resRoot + '/' + name;
        std::regex rgx("\\.(?!.*\\.)");
        std::string RES_PATH = std::regex_replace(save_path, rgx, "_result.");

        #ifdef _WIN32
            std::regex reg(R"(\.(\w*)$)");
            save_path = std::regex_replace(save_path, reg, ".txt");
            std::ofstream outputFile(save_path);
            if (!outputFile.is_open()) {
                std::cout << "Create txt file fail." << std::endl;
            }
            for (int i = 0; i < valid_csize; i++) {
                outputFile << x_list[i] * ori_hsize / hsize << " " << y_list[i] * ori_hsize / hsize << "\n";
            }
            outputFile.close();
        #elif __linux__
            cv::imwrite(RES_PATH, img);
            std::cout << "Save result at:" << RES_PATH << std::endl;
        #endif
    }

    #ifdef _WIN32
    if (show) {
        cv::imshow("results", img);
        cv::waitKey(0);
    }
    #endif
}
template <typename T>
void process_tensor_data_hard(T* tensor_data, int csize, int valid_csize, int hsize, int wsize, 
                         bool save, const std::string& resRoot, const std::string& name, 
                         int ori_hsize, bool show, const cv::Mat& img,Device& device,uint64_t arbase,uint64_t last_araddr) {
    std::vector<int> ind(valid_csize);
    std::vector<float> x_list, y_list;
	//use fpga_argmax2d_forward_buyibackend
	Tensor index_tensor = fpgaArgmax2d(device,wsize,hsize,valid_csize,csize,arbase,last_araddr);
	auto host_tensor = index_tensor.to(HostDevice::MemRegion());//做 to ps操作
	auto index_data = (uint64_t*)host_tensor.data().cptr();
	for (int i = 0; i < valid_csize; i++) {
        x_list.emplace_back(index_data[i] % wsize);
        y_list.emplace_back(index_data[i] / wsize);
    }
	// for (int i = 0; i < valid_csize; i++) {
	// 	std::cout<<"x="<<x_list[i]<<"y="<<y_list[i]<<std::endl;
    // }
    for (int i = 0; i < valid_csize; i++) {
        cv::circle(img, cv::Point(x_list[i] * ori_hsize / hsize, y_list[i] * ori_hsize / hsize), 1, cv::Scalar(0, 255, 0), 2);
    }

    if (save) {
        std::string save_path = resRoot + '/' + name;
        std::regex rgx("\\.(?!.*\\.)");
        std::string RES_PATH = std::regex_replace(save_path, rgx, "_result.");

        #ifdef _WIN32
            std::regex reg(R"(\.(\w*)$)");
            save_path = std::regex_replace(save_path, reg, ".txt");
            std::ofstream outputFile(save_path);
            if (!outputFile.is_open()) {
                std::cout << "Create txt file fail." << std::endl;
            }
            for (int i = 0; i < valid_csize; i++) {
                outputFile << x_list[i] * ori_hsize / hsize << " " << y_list[i] * ori_hsize / hsize << "\n";
            }
            outputFile.close();
        #elif __linux__
            cv::imwrite(RES_PATH, img);
            std::cout << "Save result at:" << RES_PATH << std::endl;
        #endif
    }

    #ifdef _WIN32
    if (show) {
        cv::imshow("results", img);
        cv::waitKey(0);
    }
    #endif
}
void post_process(const std::vector<Tensor>& output_tensors, PicPre& img,
	bool & show, bool & save , std::string &resRoot, std::string & name,
	Device& device,uint64_t arbase,uint64_t last_araddr,bool fpga_argmax,bool& run_sim){
	
	auto host_tensor = output_tensors[0].to(HostDevice::MemRegion());// from PL to ps.hostmem region
	int output_tensors_bits = output_tensors[0].dtype()->element_dtype.getStorageType().bits(); //获取位数：8bit or 16bit量化
	int valid_csize = 22;//22为有效通道
	int ori_hsize = img.src_img.rows;//1080为原图大小
	std::vector<int> ind(valid_csize);//argmax结果存放
	std::vector<float> x_list;
	std::vector<float> y_list;
	switch(output_tensors_bits){
	case 32: {
			auto tensor_data = (float*)host_tensor.data().cptr(); // pay attention to the data format float or int8_t
			// ------------------POST_PROCESS : reshape + argmax-----------------	
			//get output shape
			int hsize = output_tensors[0].dtype()->shape[1]; //320 hsize = wsize 
			int wsize = output_tensors[0].dtype()->shape[2]; //320
			int csize = output_tensors[0].dtype()->shape[3]; //22 or 32
			// std::cout << "hsize =" << hsize << "wsize =" << wsize << "csize =" << csize << std::endl;
			process_tensor_data_soft((float*)tensor_data, csize, valid_csize, hsize, wsize, save, resRoot, name, ori_hsize, show, img.ori_img);
			break;
	}
	case 8:{
			
			auto tensor_data = (int8_t*)host_tensor.data().cptr(); // pay attention to the data format float or int8_t
			// ------------------POST_PROCESS : reshape + argmax-----------------	
			int hsize = output_tensors[0].dtype()->shape[2]; //320 hsize = wsize 
			int wsize = output_tensors[0].dtype()->shape[3]; //320
			int csize = output_tensors[0].dtype()->shape[4]; //22 or 32
			// std::cout << "hsize =" << hsize << "wsize =" << wsize << "csize =" << csize << std::endl;
			if(fpga_argmax && !run_sim){
				process_tensor_data_hard((int8_t*)tensor_data,csize,valid_csize,hsize,wsize,save,resRoot,name,ori_hsize,show,img.ori_img,device,arbase,last_araddr);
			}else{
				process_tensor_data_soft((int8_t*)tensor_data,csize,valid_csize,hsize,wsize,save,resRoot,name,ori_hsize,show,img.ori_img);
			}
			break;
		}
		case 16:{
			auto tensor_data = (int16_t*)host_tensor.data().cptr(); // pay attention to the data format float or int16_t
			
			// ------------------POST_PROCESS : reshape + argmax-----------------	
			int hsize = output_tensors[0].dtype()->shape[2]; //320 hsize = wsize 
			int wsize = output_tensors[0].dtype()->shape[3]; //320
			int csize = output_tensors[0].dtype()->shape[4]; //22 or 32 now it's 16!
			if(fpga_argmax && !run_sim){
				process_tensor_data_hard((int16_t*)tensor_data,csize,valid_csize,hsize,wsize,save,resRoot,name,ori_hsize,show,img.ori_img,device,arbase,last_araddr);
			}else{
				process_tensor_data_soft((int16_t*)tensor_data,csize,valid_csize,hsize,wsize,save,resRoot,name,ori_hsize,show,img.ori_img);
			}
			break;
		}
		default: {
				throw "wrong bits num!";
				exit(EXIT_FAILURE);
		}
	}
	
}

int main(int argc, char* argv[])
{
	YAML::Node config = YAML::LoadFile(argv[1]);
	// icraft模型部署相关参数配置
	auto imodel = config["imodel"];
	// 仿真上板的jrpath配置
	std::string folderPath = imodel["dir"].as<std::string>();  
	bool run_sim = imodel["sim"].as<bool>();
    bool cudamode = imodel["cudamode"].as<bool>();
	std::string targetFileName;
	std::string JSON_PATH = getJrPath(run_sim,folderPath, imodel["stage"].as<std::string>());
	std::regex rgx3(".json");
	std::string RAW_PATH = std::regex_replace(JSON_PATH, rgx3, ".raw");
	std::string IMODEL_PATH = std::regex_replace(JSON_PATH, rgx3, ".imodel");
	// URL配置
	std::string ip = imodel["ip"].as<std::string>();
	// 可视化配置
	bool show = imodel["show"].as<bool>();
	bool save = imodel["save"].as<bool>();
	// 模型自身相关参数配置
	auto param = config["param"];
	bool fpga_argmax = param["fpga_argmax"].as<bool>();
	
	// 加载network
	Network network = loadNetwork(JSON_PATH, RAW_PATH); //以load j&r file创建网络
	// auto network = Network::CreateFromMSGFile(IMODEL_PATH);//以load imodel file创建网络,会快一点
	//初始化netinfo
	NetInfo netinfo = NetInfo(network);
	auto network_view = network.view(2); 
	// std::cout << netinfo.bit<<std::endl;
	// std::cout << network_view->ops[-1]->typeKey()<<std::endl;
	// 打开device
	Device device = openDevice(run_sim, ip, netinfo.mmu || imodel["mmu"].as<bool>(), cudamode);
	// 初始化session
	Session session = initSession(run_sim, network_view, device, netinfo.mmu || imodel["mmu"].as<bool>(), imodel["speedmode"].as<bool>(), imodel["compressFtmp"].as<bool>());
	// 开启计时功能
	session.enableTimeProfile(true);
	// session执行前必须进行apply部署操作
	session.apply();
	//获取arbase和last_araddr,代表output tensor在plddr的起始地址和最后一层 ftmp 的地址
	uint64_t arbase;
	uint64_t last_araddr;
	for (auto& backend : session->backends) {
		if (backend.is<BuyiBackend>()) {
			arbase = backend.cast<BuyiBackend>()->phy_segment_map.at(Segment::OUTPUT)->phy_addr;
			// auto last_arbaseadd = arbase +　backend.cast<BuyiBackend>()->phy_segment_map.at(Segment::OUTPUT);
			last_araddr = arbase + backend.cast<BuyiBackend>()->phy_segment_map.at(Segment::OUTPUT)->byte_size;
		}
	}
	std::cout<<"arbase ="<<arbase<<" last_araddr ="<<last_araddr<<std::endl;
	// 数据集相关参数配置
	auto dataset = config["dataset"];
	std::string imgRoot = dataset["dir"].as<std::string>();
	std::string imgList = dataset["list"].as<std::string>();
	std::string resRoot = dataset["res"].as<std::string>();
	checkDir(resRoot);
	
	// 统计图片数量
	int index = 0;
	auto namevector = toVector(imgList);
	int totalnum = namevector.size();
	for (auto name : namevector) {
		progress(index, totalnum);
		index++;
		std::string img_path = imgRoot + '/' + name;
		// 前处理 - warpaffine
		PicPre img(img_path, cv::IMREAD_COLOR);
		// 设置缩放比例
    	float scale =  float(netinfo.i_cubic[0].h) / float(img.src_img.rows); //640 / 1080
		// 计算缩放矩阵
		cv::Mat trans = (cv::Mat_<float>(2, 3) << scale, 0, 0, 0, scale, 0);
		int h_resized = netinfo.i_cubic[0].h;
    	int w_resized = netinfo.i_cubic[0].w;
		cv::Size size_resized(w_resized, h_resized);
		// 应用仿射变换
    	cv::warpAffine(img.src_img, img.dst_img, trans, size_resized);
		// Convert Matrix to Tensor
		Tensor img_tensor = CvMat2Tensor(img.dst_img, network);
		// 初始化dma
		dmaInit(run_sim, netinfo.ImageMake_on,img_tensor, device);
		//网络前向
		std::vector<Tensor> output_tensors = session.forward({ img_tensor });
		// 调试用
		// for(auto output : output_tensors){
		// 	std::cout << output.dtype()->shape[1] << std::endl;
		// 	std::cout << output.dtype()->shape[2] << std::endl;
		// 	std::cout << output.dtype()->shape[3] << std::endl;
		// }

		if(!run_sim) device.reset(1);
		// 计时
		#ifdef __linux__
		device.reset(1);
		calctime_detail(session);
		#endif	

		post_process(output_tensors,img,show,save,resRoot,name,device,arbase,last_araddr,fpga_argmax,run_sim);
		 		
	}
	//关闭设备
	Device::Close(device);
    return 0;
}
