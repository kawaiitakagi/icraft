#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/buyi_device.h>
#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <fstream>
#include <NetInfo.hpp>
#include <opencv2/opencv.hpp>
#include "postprocess_yolov5s+siamfc++.hpp"
#include "icraft_utils.hpp"
#include <et_device.hpp>
#include "yaml-cpp/yaml.h"
#include <task_queue.hpp>
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
auto net_1_json_file = "../imodel/plin_siamfc++_net1/siamfc++-net-1_BY.json";
auto net_1_raw_file = "../imodel/plin_siamfc++_net1/siamfc++-net-1_BY.raw";
auto net_2_json_file = "../imodel/plin_siamfc++_net2/siamfc++-net-2_BY.json";
auto net_2_raw_file = "../imodel/plin_siamfc++_net2/siamfc++-net-2_BY.raw";

const std::string MODEL_NAME = "siamfc++";
bool save = false; //是否保存结果
bool save_init_frame = false; //是否保存初始帧结果
bool show = true; //是否显示结果
std::vector<float> init_rect = {1350, 450, 260, 350 }; // 需要自行指定目标位置：x,y,w,h

int main(int argc, char* argv[]) {
    auto thread_num = 4;

    // 打开设备
    auto device = Device::Open("axi://ql100aiu?npu=0x40000000&dma=0x80000000");
    auto buyi_device = device.cast<BuyiDevice>();

    //关闭MMU
    buyi_device.mmuModeSwitch(false);

    // siamfc++模型参数配置
    float context_amount = 0.5;
    float z_size = 127;
    float x_size = 303;

    // 创建汉宁窗
    float window_influence = 0.21;
    cv::Mat hanning = CreatHannWindow(17, 17);
    cv::Mat window = cv::Mat(289, 1, CV_32F, hanning.data);

    cv::Mat xy_ctr = cv::Mat(289, 2, CV_32F);
    for (int i = 0; i < 17; i++) {
        for (int j = 0; j < 17; j++) {
            xy_ctr.at<float>(17 * i + j, 0) = 87 + 8 * j;
            xy_ctr.at<float>(17 * i + j, 1) = 87 + 8 * i;
        }
    }

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


    // 在psddr-udmabuf上申请display缓存区
    const uint64_t DISPLAY_BUFFER_SIZE = FRAME_H * FRAME_W * 2; 
    auto display_chunk = buyi_device.getMemRegion("udma").malloc(DISPLAY_BUFFER_SIZE, false);
    auto display = Display_pHDMI_RGB565(buyi_device, DISPLAY_BUFFER_SIZE, display_chunk);
    std::cout << "Display buffer udma addr=" << display_chunk->begin.addr() << '\n';

    // 加载网络
    auto network_1 = loadNetwork(net_1_json_file, net_1_raw_file);
    auto network_2 = loadNetwork(net_2_json_file, net_2_raw_file);

    // 去除siamfc++-net1输出部分的Cast&PruneAxis,并连接output<->hardop算子
    removeOutputCast(network_1, false); // false指是否开启MMU
    // 去除siamfc++-net2输入部分的Cast&AlignAxis,并连接input<->hardop算子
    removeInputCast(network_2, false); // false指是否开启MMU
    // 去除siamfc++-net2输出部分的Cast&PruneAxis,并连接output<->hardop算子
    removeOutputCast(network_2, false); // false指是否开启MMU

    // 将网络拆分为imagemake和icore
    auto image_make = network_1.view(1, 2);// imk
    auto net1_icore = network_1.viewExcept({ 0, 72 }); // view掉imk算子
    auto net2_icore = network_2.viewExcept({ 0, 134 }); // view掉imk算子

    // 打印view后的网络结构
    // auto net1_view = net1_icore.toNetwork();
    // auto net2_view = net2_icore.toNetwork();
    // net1_view.dumpJsonToFile("./net1_view.json");
    // net2_view.dumpJsonToFile("./net2_view.json");


    NetInfo net1_netinfo = NetInfo(network_1);
    NetInfo net2_netinfo = NetInfo(network_2);

    // 从json文件中读取输出层的缩放系数，网络输出的是定点数据，需要该系数来转换为浮点数据
    std::vector<float> net2_normratio = getOutputsNormratio(net2_icore);
    std::cout << "net2_normratio:" << std::endl;
    std::cout << net2_normratio[0] << std::endl;
    std::cout << net2_normratio[1] << std::endl;
    std::cout << net2_normratio[2] << std::endl;

    // 初始化net2_session，用于计算内存复用chunk尺寸
    auto sess_net2 = Session::Create<BuyiBackend, HostBackend>(net2_icore, { device, HostDevice::Default() });
    auto buyi_backend_net2 = sess_net2->backends[0].cast<BuyiBackend>();

    const uint64_t imk_output_size= FRAME_H * FRAME_W * 4;

    // 内存复用net2-申请chunk
    auto net2_input_segment = buyi_backend_net2->logic_segment_map.at(Segment::INPUT);
    auto net2_input_size = net2_input_segment->byte_size;
    auto ftmp_id = net2_icore.inputs()[1]->v_id;
    std::cout << "second input ftmp id is: " << ftmp_id << std::endl;
    auto offset = net2_input_segment->info_map.at(ftmp_id)->logic_addr - net2_input_segment->logic_addr;

    // 申请多块PLDDR，用于：①siamfc++net1&net2输入复用imk的输出  ②siamfc++net2其中两个输入复用net1的输出
    auto net2_buf_group = std::vector<MemChunk>(thread_num);
    for (int i = 0; i < thread_num; i++) {
        auto net2_input_chunk = device.defaultMemRegion().malloc(net2_input_size);
        net2_buf_group[i] = net2_input_chunk;
    }

    // 构建多个session
    auto imk_sessions = std::vector<Session>(thread_num);
    auto net1_sessions = std::vector<Session>(thread_num);
    auto net2_sessions = std::vector<Session>(thread_num);

    for (int i = 0; i < thread_num; i++) {
        imk_sessions[i] = Session::Create<BuyiBackend, HostBackend>(image_make, { device, HostDevice::Default() });
        net1_sessions[i] = Session::Create<BuyiBackend, HostBackend>(net1_icore, { device, HostDevice::Default() });
        net2_sessions[i] = Session::Create<BuyiBackend, HostBackend>(net2_icore, { device, HostDevice::Default() });

        auto buyi_backend_imk = imk_sessions[i]->backends[0].cast<BuyiBackend>();
        auto buyi_backend_1 = net1_sessions[i]->backends[0].cast<BuyiBackend>();
        auto buyi_backend_2 = net2_sessions[i]->backends[0].cast<BuyiBackend>();

        // 将同一组imagemake和icore的输入输出连接起来
        buyi_backend_imk.userSetSegment(net2_buf_group[i], Segment::OUTPUT);
        buyi_backend_1.userSetSegment(net2_buf_group[i], Segment::INPUT);
        //buyi_backend_2.userSetSegment(net2_buf_group[i], Segment::INPUT);


        // 将同一组 net1输出和net2输入连接起来
        buyi_backend_1.userSetSegment(net2_buf_group[i], Segment::OUTPUT, offset);
        buyi_backend_2.userSetSegment(net2_buf_group[i], Segment::INPUT, 0);

        imk_sessions[i].enableTimeProfile(true);
        net1_sessions[i].enableTimeProfile(true);
        net2_sessions[i].enableTimeProfile(true);

        buyi_backend_1.speedMode();
        buyi_backend_2.speedMode();

        imk_sessions[i].apply();
        net1_sessions[i].apply();
        net2_sessions[i].apply();
    }

    // fake input
    std::vector<int64_t> output_shape = { 1, CAMERA_W, CAMERA_H, 3 };
    auto tensor_layout = icraft::xir::Layout("NHWC");
    auto output_type = icraft::xrt::TensorType(icraft::xir::IntegerType::UInt8(), output_shape, tensor_layout);
    auto output_tensor = icraft::xrt::Tensor(output_type).mallocOn(icraft::xrt::HostDevice::MemRegion());
    auto img_tensor_list = std::vector<Tensor>{ output_tensor };


    auto progress_printer = std::make_shared<ProgressPrinter>(1);
    auto FPS_COUNT_NUM = 10;
    auto color = cv::Scalar(128, 0, 128);
    std::atomic<uint64_t> frame_num = 0;
    std::atomic<float> fps = 0.f;
    auto startfps = std::chrono::steady_clock::now();

    // PL端输入必须经过hardResizePL，此处不做任何尺度变换
    auto ratio_bias = preprocess_plin(buyi_device, CAMERA_W, CAMERA_H, CAMERA_W, CAMERA_H, crop_position::center);

    int8_t* display_data = new int8_t[FRAME_W * FRAME_H * 2];
    bool find_track_obj = false; 
    bool init_onetime = true;  //确保net2_warpaffine硬算子只initop一次
    std::vector<float> target_pos;
    std::vector<float> target_sz;
    std::vector<std::vector<float>> M_inversed = {    //仿射变换逆矩阵
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}
    };

    icraft::xrt::Tensor data_ptr_1;
    icraft::xrt::Tensor data_ptr_2;
    int index = 0;
    float scale = 1.0;

    // 初始化任务队列
    auto icore_task_queue = std::make_shared<Queue<InputMessageForIcore>>(thread_num);
    std::vector<bool> buffer_avaiable_flag(thread_num, true);

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
                auto start = std::chrono::high_resolution_clock::now();
                while (!buffer_avaiable_flag[buffer_index]) {
                    usleep(0);
                }

                camera.take(camera_buf_group[buffer_index]);

                try {
                    //imk前向
                    msg.image_tensor = imk_sessions[buffer_index].forward(img_tensor_list);
                    // 手动同步
                    for (auto&& tensor : msg.image_tensor) {
                        tensor.waitForReady(1000ms);
                    }
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

                icore_task_queue->Push(msg);

                // 将buffer标记为不可用，等后处理完成后再释放
                buffer_avaiable_flag[buffer_index] = false;

                #ifdef DEBUG_PRINT
                    spdlog::info("[Input] imk={:.2f}ms, buffer={}",
                        float(imk_dura.count()) / 1000,
                        buffer_index
                    );
                #endif
                buffer_index++;
                buffer_index = buffer_index % camera_buf_group.size();
                }

        }
    );

    // 线程2：siamfc++网络 icore前向及后处理
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

                if (input_msg.error_frame) {
                    cv::Mat display_mat = cv::Mat::zeros(FRAME_W, FRAME_H, CV_8UC2);
                    drawTextTopLeft(display_mat, fmt::format("No input , Please check camera."), cv::Scalar(127, 127, 127));
                    display.show(reinterpret_cast<int8_t*>(display_mat.data));
                    continue;
                }

                if (!find_track_obj) {
                    //-------------------------------------//
                    //       第一阶段：处理初始帧-siamfc++net1
                    //-------------------------------------//
                    if (save_init_frame) {                        //保存初始帧图像
                        //获取当前camera cap的图像,搬到ps
                        camera.get(display_data, camera_buf_group[input_msg.buffer_index]);
                        cv::Mat camera_frame = cv::Mat(FRAME_H, FRAME_W, CV_8UC2, display_data);
                        cv::Mat one_frame;
                        cvtColor(camera_frame, one_frame, cv::COLOR_BGR5652BGR);
                        cv::rectangle(one_frame, cv::Rect(init_rect[0], init_rect[1], init_rect[2], init_rect[3]), cv::Scalar(0, 255, 0), 4);
                        cv::imwrite("target_init_frame.jpg", one_frame);
                    }
                    auto net1_start = std::chrono::system_clock::now();
                    //net1前处理
                    target_pos = { (init_rect[0] + (init_rect[2] - 1) / 2),(init_rect[1] + (init_rect[3] - 1) / 2) }; // 计算中心点坐标
                    target_sz = { init_rect[2],init_rect[3] };    // 计算目标尺寸
                    siamfc_preprocess(target_pos, target_sz, M_inversed, scale, context_amount, z_size, z_size);
                    auto net1_preprocess_end = std::chrono::system_clock::now();
                    auto net1_preprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(net1_preprocess_end - net1_start);
                    
                    //初始化Warpaffine
                    Operation WarpAffine_net1 = net1_netinfo.WarpAffine_;
                    net1_sessions[input_msg.buffer_index]->backends[0].cast<BuyiBackend>().initOp(WarpAffine_net1);

                    auto initop_end = std::chrono::system_clock::now();
                    auto initop_duration = std::chrono::duration_cast<std::chrono::microseconds>(initop_end - net1_preprocess_end);

                    // 配置warpaffine寄存器
                    if (net1_netinfo.WarpAffine_on)  fpgaWarpaffine(M_inversed, device);

                    auto warpaffine_end = std::chrono::system_clock::now();
                    auto warpaffine_duration = std::chrono::duration_cast<std::chrono::microseconds>(warpaffine_end - initop_end);

                    //net1前向推理
                    auto net1_tensors = net1_sessions[input_msg.buffer_index].forward(input_msg.image_tensor);

                    auto net1_forward_end = std::chrono::system_clock::now();
                    auto net1_duration = std::chrono::duration_cast<std::chrono::microseconds>(net1_forward_end - warpaffine_end);

                    data_ptr_1 = net1_tensors[0];
                    data_ptr_2 = net1_tensors[1];

                    //内存块复用-同步操作
                    auto net1_out_flag1 = data_ptr_1.waitForReady(1000ms);
                    auto net1_out_flag2 = data_ptr_2.waitForReady(1000ms);
                    data_ptr_1.setReady(net1_out_flag1);
                    data_ptr_2.setReady(net1_out_flag2);

                    //将两个输出复制到其它buffer上，PLDDR->PLDDR
                    std::cout << "offset: " << offset << std::endl;
                    std::cout << "net2_buf_group[input_msg.buffer_index]->begin.addr(): " << net2_buf_group[input_msg.buffer_index]->begin.addr() << std::endl;
                    auto src_base_addr = net2_buf_group[input_msg.buffer_index]->begin.addr() + offset;
                    auto src_end_addr = net2_buf_group[input_msg.buffer_index]->begin.addr() + net2_input_size;
                    uint64_t bytesize = net2_input_size - imk_output_size;
                    for (int i = 1; i < thread_num; i++) {
                        int idx = (input_msg.buffer_index + i) % thread_num;
                        std::cout << "memcpy buffer_index form: " << input_msg.buffer_index << " to: " << idx << std::endl;
                        auto dest_base_addr = net2_buf_group[idx]->begin.addr() + offset;
                        auto dest_end_addr = net2_buf_group[idx]->begin.addr() + net2_input_size;
                        PLDDRMemRegion::Plddr_memcpy(src_base_addr, src_end_addr, dest_base_addr, dest_end_addr, device);
                    }

                    device.reset(1);

                    auto net1_total_end = std::chrono::system_clock::now();
                    auto net1_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(net1_total_end - net1_start);

                    //net2前处理
                    siamfc_preprocess(target_pos, target_sz, M_inversed, scale, context_amount, z_size, x_size);
                    auto net2_preprocess = std::chrono::system_clock::now();
                    auto net2_preprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(net2_preprocess - net1_total_end);

                    buffer_avaiable_flag[input_msg.buffer_index] = true;
                    find_track_obj = true;

                    #ifdef DEBUG_PRINT
                            spdlog::info("[Icore:net1] net1_preprocess={:.2f}ms, net1_initop={:.2f}ms, net1_warpaffine={:.2f}ms, net1_forward={:.2f}ms, net1_total={:.2f}ms, net2_pre={:.2f}ms, buffer={}",
                                float(net1_preprocess_duration.count()) / 1000,
                                float(initop_duration.count()) / 1000,
                                float(warpaffine_duration.count()) / 1000,
                                float(net1_duration.count()) / 1000,
                                float(net1_total_duration.count()) / 1000,
                                float(net2_preprocess_duration.count()) / 1000,
                                input_msg.buffer_index
                            );
                    #endif

                }
                else{
                    //-------------------------------------//
                    //       第二阶段：追踪目标
                    //-------------------------------------//
                    
                    auto net2_start = std::chrono::system_clock::now();
                    //初始化Warpaffine
                    auto net2_initop_duration = std::chrono::duration_cast<std::chrono::microseconds>(net2_start - net2_start);
                    if (init_onetime) {
                        Operation WarpAffine_net2 = net2_netinfo.WarpAffine_;
                        for (int i = 1; i < thread_num; i++) {
                            net2_sessions[i]->backends[0].cast<BuyiBackend>().initOp(WarpAffine_net2);
                        }
                        init_onetime = false;
                        auto net2_initop = std::chrono::system_clock::now();
                        net2_initop_duration = std::chrono::duration_cast<std::chrono::microseconds>(net2_initop - net2_start);
                    }

                    auto net2_warpaffine_start = std::chrono::system_clock::now();
                        
                    // 配置warpaffine寄存器
                    if(net2_netinfo.WarpAffine_on)  fpgaWarpaffine(M_inversed, device);

                    auto net2_warpaffine_end = std::chrono::system_clock::now();
                    auto net2_warpaffine_duration = std::chrono::duration_cast<std::chrono::microseconds>(net2_warpaffine_end - net2_warpaffine_start);
                        
                    // net2：前向推理
                    auto net2_tensors = net2_sessions[input_msg.buffer_index].forward({ input_msg.image_tensor[0], data_ptr_1, data_ptr_2 });

                    auto net2_forward_end = std::chrono::system_clock::now();
                    auto net2_duration = std::chrono::duration_cast<std::chrono::microseconds>(net2_forward_end - net2_warpaffine_end);
                    
                    // 手动同步
                    for (auto&& tensor : net2_tensors) {
                        tensor.waitForReady(1000ms);
                    }
                    device.reset(1);

                    // net2当前帧的后处理
                    net2_postprocess_removeoutputcast(net2_tensors, net2_normratio, target_pos, target_sz, xy_ctr, window, window_influence, x_size, scale, FRAME_W, FRAME_H);
                    auto net2_postprocess_end = std::chrono::system_clock::now();
                    auto net2_postprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(net2_postprocess_end - net2_forward_end);
                    auto net2_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(net2_postprocess_end - net2_start);

                    // net2下一帧的前处理
                    siamfc_preprocess(target_pos, target_sz, M_inversed, scale, context_amount, z_size, x_size);

                    std::cout << "target: " << target_pos[0] << " " << target_pos[1] << " " << target_sz[0] << " " << target_sz[1] << std::endl;
                    float pred_x = target_pos[0] - (target_sz[0] - 1) / 2;
                    float pred_y = target_pos[1] - (target_sz[1] - 1) / 2;
                    auto net2_preprocess_end = std::chrono::system_clock::now();
                    auto net2_preprocess_duration = std::chrono::duration_cast<std::chrono::microseconds>(net2_preprocess_end - net2_postprocess_end);

                    // 获取当前camera cap的图像,搬到ps
                    camera.get(display_data, camera_buf_group[input_msg.buffer_index]);
                    cv::Mat camera_frame = cv::Mat(FRAME_H, FRAME_W, CV_8UC2, display_data);

                    // 屏幕上显示图像
                    if (show) {
                        cv::rectangle(camera_frame, cv::Rect(pred_x, pred_y, target_sz[0], target_sz[1]), cv::Scalar(0, 255, 0), 4);
                        drawTextTwoConer(camera_frame, fmt::format("FPS: {:.1f}", fps), MODEL_NAME, cv::Scalar(0, 255, 0));
                        std::cout << "fps: " << fps << std::endl;
                        display.show(display_data);
                    }

                    // 转换格式并保存图片
                    if (save) {
                        cv::Mat camera_frame = cv::Mat(FRAME_H, FRAME_W, CV_8UC2, display_data);
                        cv::Mat cur_frame;
                        cv::cvtColor(camera_frame, cur_frame, cv::COLOR_BGR5652BGR);
                        cv::rectangle(cur_frame, cv::Rect(pred_x, pred_y, target_sz[0], target_sz[1]), cv::Scalar(0, 255, 0), 4);
                        std::string img_name = "./images/track" + intToString(index) + ".jpg";
                        cv::imwrite(img_name, cur_frame);
                        index++;
                    }

                    buffer_avaiable_flag[input_msg.buffer_index] = true;

                    #ifdef DEBUG_PRINT
                        spdlog::info("[Icore:net2] net2_initop={:.2f}ms, net2_warpaffine={:.2f}ms, net2_forward={:.2f}ms, net2_post={:.2f}ms, net2_total={:.2f}ms, net2_next_pre={:.2f}ms, buffer={}",
                            float(net2_initop_duration.count()) / 1000,
                            float(net2_warpaffine_duration.count()) / 1000,
                            float(net2_duration.count()) / 1000,
                            float(net2_postprocess_duration.count()) / 1000,
                            float(net2_total_duration.count()) / 1000,
                            float(net2_preprocess_duration.count()) / 1000,
                            input_msg.buffer_index
                        );
                    #endif
                       
                }

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

    input_thread.join();
    icore_thread.join();
    icore_task_queue->Stop();
    // 关闭设备
    Device::Close(device);
    return 0;
}

