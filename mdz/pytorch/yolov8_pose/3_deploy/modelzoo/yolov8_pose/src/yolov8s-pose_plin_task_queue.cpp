#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/buyi_device.h>
#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/cuda/device.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "postprocess_yolov8_pose.hpp"
#include "yolov8_pose_utils.hpp"
#include "icraft_utils.hpp"
#include <task_queue.hpp>
#include <et_device.hpp>
#include "yaml-cpp/yaml.h"
#include <NetInfo.hpp>
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
        std::cout << "as" << std::endl;

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

        // 在udmabuf(PSDDR)上申请摄像头缓存区 
        auto camera_buf_group = std::vector<MemChunk>(thread_num);
        for (int i = 0; i < thread_num; i++) {
            auto chunck = buyi_device.getMemRegion("udma").malloc(BUFFER_SIZE, false);
            std::cout << "Cam buffer index:" << i
                << " ,udma addr=" << chunck->begin.addr() << '\n';
            camera_buf_group[i] = chunck;//多线程
        }

        // 同样在udmabuf(PSDDR)上申请display缓存区
        const uint64_t DISPLAY_BUFFER_SIZE = FRAME_H * FRAME_W * 2;    // 摄像头输入为RGB565
        auto display_chunck = buyi_device.getMemRegion("udma").malloc(DISPLAY_BUFFER_SIZE, false);
        auto display = Display_pHDMI_RGB565(buyi_device, DISPLAY_BUFFER_SIZE, display_chunck);
        std::cout << "Display buffer udma addr=" << display_chunck->begin.addr() << '\n';

        //-------------------------------------//
        //       相关参数
        //-------------------------------------//
        // labels
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
        int NOP = param["number_of_keypoint"].as<int>();
        std::vector<std::vector<std::vector<float>>> ANCHORS =
            param["anchors"].as<std::vector<std::vector<std::vector<float>>>>();
        
        // 计算real_out_channels
        int NOA = 1;
        if (ANCHORS.size() != 0) {
            NOA = ANCHORS[0].size();
        }

        //-------------------------------------//
        //       加载网络
        //-------------------------------------//
        Network network = loadNetwork(JSON_PATH, RAW_PATH);
        //初始化netinfo
        NetInfo netinfo = NetInfo(network);

        // 获取参数bbox_info_channel、real_out_channles
        int bbox_info_channel = 64;
        std::vector<int> ori_out_channles = { N_CLASS,bbox_info_channel,NOP * 3 };
        std::pair<int, std::vector<int>> anchor_length_real_out_channles =
            _getAnchorLength(ori_out_channles, netinfo.detpost_bit, NOA);
        int anchor_length = anchor_length_real_out_channles.first;
        std::vector<int> real_out_channles = anchor_length_real_out_channles.second;

        // PL端图像尺寸，即神经网络网络输入图片尺寸
        int NET_W = netinfo.i_cubic[0].w ;
        int NET_H = netinfo.i_cubic[0].h ;
        std::cout << NET_W << NET_H <<std::endl;
        // 在PLDDR上申请imagemake缓存区，用来缓存给AI计算的图片
        const uint64_t IMK_OUTPUT_FTMP_SIZE = NET_H * NET_W * 4;
        auto imagemake_buf_group = std::vector<MemChunk>(thread_num);
        for (int i = 0; i < thread_num; i++) {
            auto chunck = buyi_device.getMemRegion("plddr").malloc(IMK_OUTPUT_FTMP_SIZE, false);
            std::cout << "image make buffer index:" << i
                << " ,plddr addr=" << chunck->begin.addr() << '\n';
            imagemake_buf_group[i] = chunck;
        }

        //-------------------------------------//
        //       拆分网络，构建session
        //-------------------------------------//
        // 将网络拆分为imagemake和icore
        auto image_make = network.view(netinfo.inp_shape_opid + 1, netinfo.inp_shape_opid + 2);
        auto icore = network.view(netinfo.inp_shape_opid + 2);
        // 计算复用的icore阶段网络的ftmp大小，用于复用相同网络的ftmp
        auto icore_dummy_session = Session::Create<BuyiBackend, HostBackend>(icore, { buyi_device, HostDevice::Default() });
        auto& icore_dummy_backends = icore_dummy_session->backends;
        auto icore_buyi_dummy_backends = icore_dummy_backends[0].cast<BuyiBackend>(); // 获取绑定icore的buyi后端
        std::cout << "begin to compress ftmp" << std::endl;
        icore_buyi_dummy_backends.compressFtmp();  // 开启中间层压缩功能
        std::cout << "compress ftmp done" << std::endl;
        auto network_ftmp_size = icore_buyi_dummy_backends->phy_segment_map.at(Segment::FTMP)->byte_size; //复用ftmp大小
        std::cout << "after compress network ftmp size=" << network_ftmp_size;
        auto network_ftmp_chunck = buyi_device.getMemRegion("plddr").malloc(network_ftmp_size, false);  // PLDDR上，申请一个压缩后的网络大小的buffer

        const std::string MODEL_NAME = icore_dummy_session->network_view.network()->name;

        // 构建多个session
        auto imk_sessions = std::vector<Session>(thread_num);
        auto icore_sessions = std::vector<Session>(thread_num);
        for (int i = 0; i < thread_num; i++) {
            // 创建session
            imk_sessions[i] = Session::Create<BuyiBackend, HostBackend>(image_make, { buyi_device, HostDevice::Default() });
            icore_sessions[i] = Session::Create<BuyiBackend, HostBackend>(icore, { buyi_device, HostDevice::Default() });

            // 将同一组imagemake和icore的输入输出连接起来
            auto& imk_backends = imk_sessions[i]->backends;
            auto imk_buyi_backend = imk_backends[0].cast<BuyiBackend>();
            imk_buyi_backend.userSetSegment(imagemake_buf_group[i], Segment::OUTPUT); // 输出到imagemake_buf_group[i]

            auto& icore_backends = icore_sessions[i]->backends;
            auto icore_buyi_backend = icore_backends[0].cast<BuyiBackend>();
            icore_buyi_backend.userSetSegment(imagemake_buf_group[i], Segment::INPUT); // 输入为imagemake_buf_group[i]

            // 开启压缩中间层ftmp
            icore_buyi_backend.compressFtmp(); // 中间层ftmp由展开存储变为覆盖存储，压缩空间
            icore_buyi_backend.userSetSegment(network_ftmp_chunck, Segment::FTMP); // 多个相同复用一个ftmp内存，省空间

            // 开启speedmode
            std::cout << "open speed mode" << std::endl;
            icore_buyi_backend.speedMode();

            imk_sessions[i].apply();
            icore_sessions[i].apply();
        }


        // fake input: 用来占位，没有数据
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
        auto startfps = std::chrono::steady_clock::now();  //FPS计算：计时开始
        YoloPostResult post_results;


        // PL端的resize，需要resize到AI神经网络的尺寸
        auto ratio_bias = preprocess_plin(buyi_device, CAMERA_W, CAMERA_H, NET_W, NET_H, crop_position::center);

        // 用于神经网络结果的坐标转换
        float RATIO_W = std::get<0>(ratio_bias);
        float RATIO_H = std::get<1>(ratio_bias);
        int BIAS_W = std::get<2>(ratio_bias);
        int BIAS_H = std::get<3>(ratio_bias);

        int8_t* display_data = new int8_t[FRAME_W * FRAME_H * 2];

        // 初始化任务队列
        auto icore_task_queue = std::make_shared<Queue<InputMessageForIcore>>(thread_num);
        auto post_task_queue = std::make_shared<Queue<IcoreMessageForPost>>(thread_num);
        std::vector<bool> buffer_avaiable_flag(thread_num, true);  // camera_buf_group是否可用


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
                    msg.buffer_index = buffer_index; // camera的index与线程index对齐，保证后处理同一图像
                    auto start = std::chrono::high_resolution_clock::now();
                    while (!buffer_avaiable_flag[buffer_index]) { // camera_buf_group在buffer_index处不可用
                        usleep(0);  // 线程放弃当前时间片，暂停执行
                    }

                    camera.take(camera_buf_group[buffer_index]); // 直接将camera数据送入imagemake中
                    std::cout << "take image sucessful !" << std::endl;

                    try {
                        msg.image_tensor = imk_sessions[buffer_index].forward(img_tensor_list);//imk前向
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
                    // 将buffer标记为不可用，等后处理完成后再释放（camera中此处图像还有用）
                    buffer_avaiable_flag[buffer_index] = false;

                    auto wait_dura = std::chrono::duration_cast<std::chrono::microseconds>
                        (std::chrono::high_resolution_clock::now() - start);
                    icore_task_queue->Push(msg);  // 将imk的msg传递给icore_task_queue

                    buffer_index++;
                    buffer_index = buffer_index % camera_buf_group.size();
                }
            }
        );

        // 线程2：icore前向
        auto icore_thread = std::thread(
            [&]()
            {
                std::stringstream ss;
                ss << std::this_thread::get_id();
                uint64_t id = std::stoull(ss.str());
                spdlog::info("[PLin_Vpu Demo] Icore thread start!, id={}", id);

                while (true) {
                    InputMessageForIcore input_msg;
                    icore_task_queue->Pop(input_msg); // 取出icore_task_queue头部input_msg，队列为空时等待

                    IcoreMessageForPost post_msg;
                    post_msg.buffer_index = input_msg.buffer_index;
                    post_msg.error_frame = input_msg.error_frame;

                    if (input_msg.error_frame) {
                        post_task_queue->Push(post_msg);//跳过错误帧
                        continue;
                    }

                    post_msg.icore_tensor
                        = icore_sessions[input_msg.buffer_index].forward(input_msg.image_tensor);//icore前向

                    device.reset(1);

                    post_task_queue->Push(post_msg); // 将包含icroe输出的post_msg传入post_task_queue
                }
            
            }
        );

        // 线程3：后处理
        auto post_thread = std::thread(
            [&]()
            {
                std::stringstream ss;
                ss << std::this_thread::get_id();
                uint64_t id = std::stoull(ss.str());
                spdlog::info("[PLin_Vpu Demo] Post thread start!, id={}", id);

                int buffer_index = 0;
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
                        N_CLASS,NOP, ANCHORS,device,real_out_channles,bbox_info_channel);
                    std::vector<int> id_list = std::get<0>(post_results);
                    std::vector<float> socre_list = std::get<1>(post_results);
                    std::vector<cv::Rect2f> box_list = std::get<2>(post_results);
                    std::vector<std::vector<std::vector<float>>> key_point_list = std::get<3>(post_results);

                    camera.get(display_data, camera_buf_group[post_msg.buffer_index]);
                    buffer_avaiable_flag[buffer_index] = true;  // 允许

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

                    N_CLASS = 1;
                    for (auto oneobj_kpt : key_point_list) {
                        int k = 0;
                        // 缩放key_point中x,y值
                        for(int i = 0; i < oneobj_kpt.size(); i++){
                            oneobj_kpt[i][0] = oneobj_kpt[i][0]* RATIO_W + BIAS_W;
                            oneobj_kpt[i][1] = oneobj_kpt[i][1]* RATIO_H + BIAS_H;
                        }
                        for (auto one_kpt : oneobj_kpt) {
                            if (one_kpt[2] > 0.5) {
                                cv::Scalar color = cv::Scalar(kColorMap_YOLOV7_POSE[kKptColorIndex[k]][0], kColorMap_YOLOV7_POSE[kKptColorIndex[k]][1], kColorMap_YOLOV7_POSE[kKptColorIndex[k]][2]);
                                cv::circle(mat, { (int)one_kpt[0], (int)one_kpt[1] }, 4, color, -1);
                                k++;
                            }
                        }
                        int ske = 0;
                        for (auto sk_pair : kCocoSkeleton) {
                            int kp1 = sk_pair[0] - 1;
                            int kp2 = sk_pair[1] - 1;
                            if (((oneobj_kpt[kp1][2]) > 0.5) && ((oneobj_kpt[kp2][2]) > 0.5)) {                      
                                cv::Scalar color = cv::Scalar(kColorMap_YOLOV7_POSE[kLimbColorIndex[ske]][0], kColorMap_YOLOV7_POSE[kLimbColorIndex[ske]][1], kColorMap_YOLOV7_POSE[kLimbColorIndex[ske]][2]);
                                cv::line(mat,
                                    { int(oneobj_kpt[kp1][0]), int(oneobj_kpt[kp1][1]) },
                                    { int(oneobj_kpt[kp2][0]), int(oneobj_kpt[kp2][1]) },
                                    color, 2);
                            }
                            ske++;
                        }
                    }

                    drawTextTwoConer(mat, fmt::format("FPS: {:.1f}", fps), MODEL_NAME, color);
                    // 保存图像到文件
                    // std::string filename = "./output_image.png";
                    // bool success = cv::imwrite(filename, mat);

                    // if (success) {
                    //     std::cout << "Image saved successfully as " << filename << std::endl;
                    // } else {
                    //     std::cerr << "Failed to save image." << std::endl;
                    // }
                    display.show(display_data);

                    buffer_index++;
                    buffer_index = buffer_index % camera_buf_group.size();

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
        );
        // 等待子线程结束后才进入主线程
        input_thread.join(); 
        icore_thread.join();
        post_thread.join();

        icore_task_queue->Stop();
        post_task_queue->Stop();
        Device::Close(device);

	}
    
	catch (const std::exception& e)
	{
        std::cout << e.what() << std::endl;
	}
    return 0;
}
