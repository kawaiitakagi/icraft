#include <algorithm>
#include <memory>
#include <string>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>

#include <linux/videodev2.h>
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/buyi_device.h>
#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "postprocess_yolov5.hpp"
#include "icraft_utils.hpp"
#include <et_device.hpp>
#include "yaml-cpp/yaml.h"
#include <NetInfo.hpp>
#include <task_queue.hpp>
using namespace icraft::xrt;
using namespace icraft::xir;



int main(int argc, char* argv[]) {
	try
	{
        auto thread_num = 4;

        YAML::Node config = YAML::LoadFile(argv[1]);
        // icraft模型部署相关参数配置
        auto imodel = config["imodel"];
        // 仿真上板的jrpath配置
        std::string folderPath = imodel["dir"].as<std::string>();
        std::string targetFileName;
        std::string JSON_PATH = getJrPath(false, folderPath, imodel["stage"].as<std::string>());
        std::regex rgx3(".json");
        std::string RAW_PATH = std::regex_replace(JSON_PATH, rgx3, ".raw");
        // 加载network
        Network network = loadNetwork(JSON_PATH, RAW_PATH);
        //初始化netinfo
        NetInfo netinfo = NetInfo(network);
        // 打开device
        
        Device device = openDevice(false, "", netinfo.mmu||imodel["mmu"].as<bool>());
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



        // PL端图像尺寸，即神经网络网络输入图片尺寸
        int NET_W = netinfo.i_cubic[0].w;
        int NET_H = netinfo.i_cubic[0].h;



        // 生成多个session
        auto network_sessions = std::vector<Session>(thread_num);
        auto imk_sessions = std::vector<Session>(thread_num);
        auto icore_sessions = std::vector<Session>(thread_num);
        for (int i = 0; i < thread_num; i++) {
            network_sessions[i] = Session::Create<BuyiBackend, HostBackend>(network, { buyi_device, HostDevice::Default() });
            network_sessions[i].apply();

            imk_sessions[i] = network_sessions[i].sub(netinfo.inp_shape_opid + 1, netinfo.inp_shape_opid + 2);
            icore_sessions[i] = network_sessions[i].sub(netinfo.inp_shape_opid + 2);
            std::cout << "Presentation forward operator ...." << std::endl;
            auto ops = network_sessions[i].getForwards();
            for (auto&& op : ops) {
                std::cout << "op name:" << std::get<0>(op)->typeKey() << '\n';
            }

        }

        // const std::string MODEL_NAME = icore_dummy_session->network_view.network()->name;

        const std::string MODEL_NAME = "yolov5";
        //// 从json文件中读取输出层的缩放系数，网络输出的是定点数据，需要该系数来转换为浮点数据
        //std::vector<float> SCALE = getOutputNormratio(network);

        // fake input
        std::vector<int64_t> output_shape = { 1, NET_W, NET_H, 3 };
        auto tensor_layout = icraft::xir::Layout("NHWC");
        auto output_type = icraft::xrt::TensorType(icraft::xir::IntegerType::UInt8(), output_shape, tensor_layout);
        auto output_tensor = icraft::xrt::Tensor(output_type).mallocOn(icraft::xrt::HostDevice::MemRegion());
        auto img_tensor_list = std::vector<Tensor>{ output_tensor };

        auto icore_task_queue = std::make_shared<Queue<InputMessageForIcore>>(thread_num);
        auto post_task_queue = std::make_shared<Queue<IcoreMessageForPost>>(thread_num);
        auto progress_printer = std::make_shared<ProgressPrinter>(1);
        auto FPS_COUNT_NUM = 30;
        auto color = cv::Scalar(128, 0, 128);
        std::atomic<uint64_t> frame_num = 0;
        std::atomic<float> fps = 0.f;
        auto startfps = std::chrono::steady_clock::now();
        YoloPostResult post_results;

        std::vector<bool> buffer_avaiable_flag(thread_num, true);
        // PL端的resize，需要resize到AI神经网络的尺寸
        auto ratio_bias = preprocess_plin(buyi_device, CAMERA_W, CAMERA_H, NET_W, NET_H, crop_position::center);

        // 用于神经网络结果的坐标转换
        float RATIO_W = std::get<0>(ratio_bias);
        float RATIO_H = std::get<1>(ratio_bias);
        int BIAS_W = std::get<2>(ratio_bias);
        int BIAS_H = std::get<3>(ratio_bias);

        auto input_thread = std::thread(
            [&]()
            {
                std::stringstream ss;
                ss << std::this_thread::get_id();
                uint64_t id = std::stoull(ss.str());
                spdlog::info("[PLin Demo] Input process thread start!, id={}", id);


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
                spdlog::info("[PLin Demo] Icore thread start!, id={}", id);

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
                spdlog::info("[PLin Demo] Post thread start!, id={}", id);
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
                    post_results = post_detpost_plin(post_msg.icore_tensor, post_results, netinfo, conf, iou_thresh, MULTILABEL, fpga_nms,
                        N_CLASS, ANCHORS, device);
                    std::vector<int> id_list = std::get<0>(post_results);
                    std::vector<float> socre_list = std::get<1>(post_results);
                    std::vector<cv::Rect2f> box_list = std::get<2>(post_results);


                    camera.get(display_data, camera_buf_group[post_msg.buffer_index]);
                    buffer_avaiable_flag[post_msg.buffer_index] = true;
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

                    drawTextTwoConer(mat, fmt::format("FPS: {:.1f}", fps), MODEL_NAME, color);
                    display.show(display_data);
                    //如果没有显示器，可以选择存图查看结果是否正确(打开下面三句代码)，但这样会降低fps
                    // cv::Mat out;
                    // cvtColor(mat, out, cv::COLOR_BGR5652BGR);
                    // cv::imwrite("./_result.jpg", out);

                    //-------------------------------------//
                    //       帧数计算
                    //-------------------------------------//
                    // std::cout << "fps:" <<fps <<std::endl;

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

	}
	catch (const std::exception& e)
	{
        std::cout << e.what() << std::endl;
	}


    return 0;
}
