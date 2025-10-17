#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/buyi_device.h>
#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/cuda/device.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "postprocess_yolov8.hpp"
#include "icraft_utils.hpp"
#include "et_device.hpp"
#include "yaml-cpp/yaml.h"
#include <NetInfo.hpp>
using namespace icraft::xrt;
using namespace icraft::xir;



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
        std::string names_path = dataset["names"].as<std::string>();
        std::vector<std::string> LABELS = toVector(names_path);

        // 模型自身相关参数配置
        auto param = config["param"];
        float conf = param["conf"].as<float>();
        float iou_thresh = param["iou_thresh"].as<float>();
        bool MULTILABEL = param["multilabel"].as<bool>();
        bool fpga_nms = param["fpga_nms"].as<bool>();
        int N_CLASS = param["number_of_class"].as<int>();
        std::vector<std::vector<std::vector<float>>> ANCHORS =
            param["anchors"].as<std::vector<std::vector<std::vector<float>>>>();

        int bbox_info_channel = 64;
	    std::vector<int> ori_out_channles = { N_CLASS,bbox_info_channel };
	    int parts = ori_out_channles.size(); // parts = 2
	    int NOH = param["number_of_head"].as<int>();
        bool run_sim = imodel["sim"].as<bool>();
    bool cudamode = imodel["cudamode"].as<bool>();
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
        // 选择对网络进行切分
        auto network_view = network.view(netinfo.inp_shape_opid + 1);
        // 初始化session
        Session session = initSession(false, network_view, device, false, true, true);
        const std::string MODEL_NAME = session->network_view.network()->name;
        session.apply();

        std::cout << "Presentation forward operator ...." << std::endl;
        auto ops = session.getForwards();
        for (auto&& op : ops) {
            std::cout << "op name:" << std::get<0>(op)->typeKey() << '\n';
        }
        // PL端图像尺寸，即神经网络网络输入图片尺寸
        int NET_W = netinfo.i_cubic[0].w;
        int NET_H = netinfo.i_cubic[0].h;


        //// 从json文件中读取输出层的缩放系数，网络输出的是定点数据，需要该系数来转换为浮点数据
        //std::vector<float> SCALE = getOutputNormratio(network);

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
            camera.take(camera_buf);
            
            auto icore_tensor = session.forward(img_tensor_list);
            
            device.reset(1);
            
            camera.wait();


            //-------------------------------------//
            //       后处理
            //-------------------------------------//
            std::vector<float> normalratio = netinfo.o_scale;
			std::vector<int> real_out_channles = _getReal_out_channles(ori_out_channles, netinfo.detpost_bit, N_CLASS);
			std::vector<std::vector<float>> _norm = set_norm_by_head(NOH, parts, normalratio);
    		std::vector<float> _stride = get_stride(netinfo);
            
            post_results = post_detpost_plin(icore_tensor, post_results, netinfo, conf, iou_thresh, MULTILABEL, fpga_nms,
                N_CLASS, ANCHORS,device,run_sim,_norm,real_out_channles,_stride,bbox_info_channel);
            std::vector<int> id_list = std::get<0>(post_results);
            std::vector<float> socre_list = std::get<1>(post_results);
            std::vector<cv::Rect2f> box_list = std::get<2>(post_results);


            camera.get(display_data, camera_buf);
            cv::Mat mat = cv::Mat(FRAME_H, FRAME_W, CV_8UC2, display_data);
            for (int index = 0; index < box_list.size(); ++index) {
                float x1 = box_list[index].tl().x * RATIO_W + BIAS_W;
                float y1 = box_list[index].tl().y * RATIO_H + BIAS_H;
                float w = box_list[index].width * RATIO_W;
                float h = box_list[index].height * RATIO_H;
                int id = id_list[index];
                cv::Scalar color = classColor(id);
                double font_scale = 1;
                int thickness = 1;
                cv::rectangle(mat, cv::Rect2f(x1, y1, w, h), color, 6, cv::LINE_8, 0);
                std::string s = LABELS[id_list[index]].substr(0, LABELS[id_list[index]].size() - 1) + ":" + std::to_string(int(round(socre_list[index] * 100))) + "%";
                cv::Size s_size = cv::getTextSize(s, cv::FONT_HERSHEY_COMPLEX, font_scale, thickness, 0);
                cv::rectangle(mat, cv::Point(x1, y1 - s_size.height - 6), cv::Point(x1 + s_size.width, y1), color, -1);
                cv::putText(mat, s, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_DUPLEX, font_scale, cv::Scalar(255, 255, 255), thickness);
            }
            std::cout<<"fps ="<<fps<<std::endl;
            drawTextTwoConer(mat, fmt::format("FPS: {:.1f}", fps), MODEL_NAME, color);
            display.show(display_data);
            //如果没有显示器，可以选择存图查看结果是否正确(打开下面三句代码)，但这样会降低fps
            // cv::Mat out;
            // cvtColor(mat, out, cv::COLOR_BGR5652BGR);
            // cv::imwrite("../images/output/_thread_result.jpg", out);
            // std::cout<<"Save!  Done"<<std::endl;
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
