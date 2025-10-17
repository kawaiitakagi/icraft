#include <iostream>
#include <fstream>
#include <regex>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <icraft-xrt/core/session.h>
#include <icraft-xrt/dev/host_device.h>
#include <icraft-xrt/dev/buyi_device.h>
#include <icraft-backends/buyibackend/buyibackend.h>
#include <icraft-backends/hostbackend/cuda/device.h>
#include <icraft-backends/hostbackend/backend.h>
#include <icraft-backends/hostbackend/utils.h>
#include <random>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include "yaml-cpp/yaml.h"
#include "utils.hpp"
#include "et_device.hpp"
#include "icraft_utils.hpp"
#include <vector>
//#include "post_process_detr.hpp"
using namespace icraft::xrt;
using namespace icraft::xir;
namespace fs = std::filesystem;

int ftmp_read(std::vector<float>& label,std::string file_path) {

    // 以二进制模式打开文件
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cout << "Cannot open file!\n";
    }
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::size_t numfloats = size / sizeof(float);

    // 创建一个足够大的缓冲区来保存文件内容
    label.resize(numfloats);

    // 读取文件内容
    if (!file.read(reinterpret_cast<char*>(label.data()), size)) {
        std::cout << "Failed to read file!\n";
        return 0;
    }
    return 1;
}

std::string replaceUsingRegex(const std::string& str, const std::string& from, const std::string& to) {
    std::regex reg(from);
    return std::regex_replace(str, reg, to);
}

float MAE(const std::vector<std::vector<float>>& pred, const std::vector<std::vector<float>>& true_values) {
    if (pred.size() != true_values.size() || (pred.size() > 0 && pred[0].size() != true_values[0].size())) {
        throw std::invalid_argument("The dimensions of predicted values and true values must be the same.");
    }

    float sum_error = 0.0;
    size_t total_elements = 0;

    // 遍历二维数组
    for (size_t i = 0; i < pred.size(); ++i) {
        for (size_t j = 0; j < pred[i].size(); ++j) {
            sum_error += std::abs(pred[i][j] - true_values[i][j]); // 累加绝对误差
            total_elements++; // 计数元素总数
        }
    }

    return sum_error / total_elements; // 返回平均绝对误差
}

float MSE(const std::vector<std::vector<float>>& pred, const std::vector<std::vector<float>>& true_vals) {
    if (pred.size() != true_vals.size()) {
        throw std::invalid_argument("Outer vectors must be of the same length");
    }

    size_t total_elements = 0;
    float sum = 0.0f;

    for (size_t i = 0; i < pred.size(); ++i) {
        if (pred[i].size() != true_vals[i].size()) {
            throw std::invalid_argument("Inner vectors must be of the same length");
        }

        for (size_t j = 0; j < pred[i].size(); ++j) {
            float diff = pred[i][j] - true_vals[i][j];
            sum += std::pow(diff, 2);
            ++total_elements;
        }
    }

    return sum / total_elements;
}
float cal_accuracy(const std::vector<int>& y_pred, const std::vector<int>& y_true) {
    if (y_pred.size() != y_true.size()) {
        throw std::invalid_argument("Vectors must be of the same length");
    }

    int correct_count = 0;
    for (size_t i = 0; i < y_pred.size(); ++i) {
        if (y_pred[i] == y_true[i]) {
            ++correct_count;
        }
    }

    return static_cast<float>(correct_count) / y_pred.size();
}
struct Metrics {
    float precision;
    float recall;
    float f1;
};

Metrics cal_precision_recall(const std::vector<int>& predictions, const std::vector<int>& trues) {
    if (predictions.size() != trues.size()) {
        throw std::invalid_argument("Vectors must be of the same length");
    }

    int TP = 0, FP = 0, FN = 0;

    for (size_t i = 0; i < predictions.size(); ++i) {
        if (predictions[i] == 1 && trues[i] == 1) {
            ++TP;
        }
        if (predictions[i] == 1 && trues[i] == 0) {
            ++FP;
        }
        if (predictions[i] == 0 && trues[i] == 1) {
            ++FN;
        }
    }

    float precision = TP / static_cast<float>(TP + FP);
    float recall = TP / static_cast<float>(TP + FN);
    float f1 = 2 * (precision * recall) / (precision + recall);

    return Metrics{ precision, recall, f1 };
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
    std::string JSON_PATH = getJrPath(run_sim, folderPath, imodel["stage"].as<std::string>());
    std::regex rgx3(".json");
    std::string RAW_PATH = std::regex_replace(JSON_PATH, rgx3, ".raw");
    // URL配置
    std::string ip = imodel["ip"].as<std::string>();
    // 可视化配置
    bool show = imodel["show"].as<bool>();
    bool save = imodel["save"].as<bool>();

    // 加载network
    Network network = loadNetwork(JSON_PATH, RAW_PATH);
    //初始化netinfo
    NetInfo netinfo = NetInfo(network);
    // 打开device
    Device device = openDevice(run_sim, ip, netinfo.mmu || imodel["mmu"].as<bool>(), cudamode);
    // 初始化session
    Session session = initSession(run_sim, network, device, netinfo.mmu || imodel["mmu"].as<bool>(), imodel["speedmode"].as<bool>(), imodel["compressFtmp"].as<bool>());
    // 开启计时功能
    session.enableTimeProfile(true);
    // session执行前必须进行apply部署操作
    session.apply();
    
    // 数据集相关参数配置
    auto dataset = config["dataset"];
    std::string imgRoot = dataset["data_dir"].as<std::string>();
    std::string label_dir = dataset["label_dir"].as<std::string>();
    std::string imgList = dataset["list"].as<std::string>();

    std::string resRoot = dataset["res"].as<std::string>();
    checkDir(resRoot);

    // 模型自身相关参数配置
    auto param = config["param"];
    int label_len = param["label_len"].as<int>();
    int seq_len = param["seq_len"].as<int>();
    int dec_in = param["dec_in"].as<int>();
    int pre_len = param["pre_len"].as<int>();
    int dec_out = param["dec_out"].as<int>();

    std::string task = param["task"].as<std::string>();
    if (task == "long_term_forecast") {
        //固定除了图片以外的其他输入
        //auto pos_emb_tensor = hostbackend::utils::Ftmp2Tensor("../io/ltf/input/pos_emb.ftmp", network.inputs()[1].tensorType());
        //auto hx_tensor = hostbackend::utils::Ftmp2Tensor("../io/ltf/input/hx.ftmp", network.inputs()[2].tensorType());
        // 统计图片数量
        int index = 0;
        auto namevector = toVector(imgList);
        int totalnum = namevector.size();
        std::vector<std::vector<float>> preds;
        std::vector<std::vector<float>> trues;
        for (auto name : namevector) {
            std::vector<float> input_seq;
            std::vector<float> output_seq;
            std::vector<float> label_seq;
            progress(index, totalnum);
            index++;
            std::string mean_n = replaceUsingRegex(name, ".ftmp", "_mean.ftmp");
            std::string stdev_n = replaceUsingRegex(name, ".ftmp", "_stdev.ftmp");

            std::string img_path = imgRoot + '/' + name;
            std::string _mean_path = imgRoot + '/' + mean_n;
            std::string _stdev_path = imgRoot + '/' + stdev_n;

            std::string label_path = label_dir + '/' + name;
            // 前处理
            auto x_tensor = hostbackend::utils::Ftmp2Tensor(img_path, network.inputs()[0].tensorType());
            auto mean_tensor = hostbackend::utils::Ftmp2Tensor(_mean_path, network.inputs()[1].tensorType());
            auto stdev_tensor = hostbackend::utils::Ftmp2Tensor(_stdev_path, network.inputs()[2].tensorType());

            auto input_ptr = (float*)x_tensor.data().cptr();
            auto mean_ptr = (float*)mean_tensor.data().cptr();
            auto stdev_ptr = (float*)stdev_tensor.data().cptr();

            for (int i = (dec_in - 1)* seq_len; i < seq_len * dec_in; i += 1) {
                input_seq.push_back(*(input_ptr + i)*(*(stdev_ptr + 6))+(*(mean_ptr + 6)));
            }

            //forward
            auto output_tensors = session.forward({ x_tensor, mean_tensor, stdev_tensor });

            if (!run_sim) device.reset(1);
            // 计时
#ifdef __linux__
            device.reset(1);
            calctime_detail(session);
#endif
            //load output tensors to output_seq for show
            auto res_tensor = output_tensors[0].to(HostDevice::MemRegion());
            auto res_ptr = (float*)res_tensor.data().cptr();
            
            for (int i = dec_out - 1; i < pre_len * dec_out; i += dec_out) {
                output_seq.push_back(*(res_ptr + i));
            }
            //load label to label_seq for show
            std::vector<float> label;
            ftmp_read(label,label_path);
            


            for (int i = 0; i < dec_out; i++) {
                std::vector<float> chanel;
                for (int j = 0; j < dec_out* pre_len; j+=dec_out) {
                    int offset = i + j;
                    float ele = *(res_ptr + offset);
                    chanel.push_back(ele);
                }
                preds.push_back(chanel);
            }

            for (int i = 0; i < dec_in; i++) {
                std::vector<float> chanel;
                for (int j = (label_len - pre_len)* dec_out+i; j < dec_out * label_len; j += dec_out) {
                    int offset = j;
                    float ele = *(label.begin() + offset);
                    chanel.push_back(ele);
                }
                trues.push_back(chanel);
            }



            for (auto i = label.begin()+dec_out - 1; i < label.end(); i += dec_out) {
                label_seq.push_back(*i);

            }
            //merge input and label for show
            input_seq.insert(input_seq.end(), label_seq.end()- pre_len, label_seq.end());

            //init show image
            int show_w = (((int)input_seq.size() * 4) / 100 + 1) * 100;
            int show_h = show_w * 0.75;
            cv::Mat img(show_h, show_w, CV_8UC3, cv::Scalar(255, 255, 255));
            std::vector<cv::Point> label_points;
            std::vector<cv::Point> output_points;
            
            //normalize
            auto maxE = std::max_element(input_seq.begin(), input_seq.end());
            auto minE = std::min_element(input_seq.begin(), input_seq.end());
            float scale = show_h*0.8 / (*maxE - *minE);
            float zero_point = (*maxE + *minE) / 2;

            for (int p = 0; p < input_seq.size(); p++) {
                label_points.push_back(cv::Point((p+2) * 4, -(input_seq[p] - zero_point) * scale + show_h/2));
            }
            for (int p = 0; p < output_seq.size(); p++) {
                output_points.push_back(cv::Point((p+2+ seq_len) * 4, -(output_seq[p] - zero_point) * scale + show_h / 2));
            }

            //draw
            for (size_t i = 0; i < label_points.size() - 1; ++i) {
                cv::line(img, label_points[i], label_points[i + 1], cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
            }
            for (size_t i = 0; i < output_points.size() - 1; ++i) {
                cv::line(img, output_points[i], output_points[i + 1], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            }
            // save res image
            std::string id = name.substr(0,name.size()-5);
            cv::imwrite(resRoot+'/'+id+".png", img);

            input_seq.clear();
            output_seq.clear();
            label_seq.clear();
        }
   
        //std::cout << MAE(preds, trues)<<std::endl;
        std::fstream file(resRoot + "/result.txt", std::ios::in | std::ios::out | std::ios::app);
        if (file.is_open()) {
            // 写入文件
            //std::string id = name.substr(0, name.size() - 5);
            file << "mae : " + std::to_string(MAE(preds, trues))<< std::endl;
            file << "mse : " + std::to_string(MSE(preds, trues)) << std::endl;
        }


    }
    if (task == "exp_classification") {
        // 统计图片数量
        int index = 0;
        auto namevector = toVector(imgList);
        int totalnum = namevector.size();
        std::vector<float> input_seq;
        std::vector<float> output_seq;
        std::vector<float> label_seq;

        std::vector<int> preds;
        std::vector<int> trues;


        for (auto name : namevector) {
            progress(index, totalnum);
            index++;
            std::string img_path = imgRoot + '/' + name;
            std::string label_path = label_dir + '/' + name;
            // 前处理
            auto x_tensor = hostbackend::utils::Ftmp2Tensor(img_path, network.inputs()[0].tensorType());
            auto input_ptr = (float*)x_tensor.data().cptr();

            //forward
            auto output_tensors = session.forward({ x_tensor });

            if (!run_sim) device.reset(1);
            // 计时
#ifdef __linux__
            device.reset(1);
            calctime_detail(session);
#endif
            //load output tensors to output_seq for show
            auto res_tensor = output_tensors[0].to(HostDevice::MemRegion());
            auto res_ptr = (float*)res_tensor.data().cptr();
            for (int i = dec_out - 1; i < pre_len * dec_out; i += dec_out) {
                output_seq.push_back(*(res_ptr + i));
            }
            auto it = std::max_element(output_seq.begin(), output_seq.end());

            // 计算索引
            int cls = std::distance(output_seq.begin(), it);
            //load label to label_seq for show
            std::vector<float> label;
            ftmp_read(label, label_path);
            for (auto i = label.begin() + dec_out - 1; i < label.end(); i += dec_out) {
                label_seq.push_back(*i);

            }
            trues.push_back(int(label_seq[0]));
            preds.push_back(cls);
            //std::fstream file(resRoot+"/result.txt", std::ios::in | std::ios::out | std::ios::app);
            //if (file.is_open()) {
            //    // 写入文件
            //    std::string id = name.substr(0, name.size() - 5);
            //    file << id + ":pred " + std::to_string(cls) + ',' + "label " + std::to_string(int(label_seq[0]))
            //        << std::endl;
            //}
            input_seq.clear();
            output_seq.clear();
            label_seq.clear();
        }
        float acc = cal_accuracy(preds, trues);
        Metrics m = cal_precision_recall(preds, trues);
        std::fstream file(resRoot+"/result.txt", std::ios::in | std::ios::out | std::ios::app);
        if (file.is_open()) {
            // 写入文件

            file <<  "acc : " + std::to_string(acc)<< std::endl;
            file << "precision : " + std::to_string(m.precision) << std::endl;
            file << "recall : " + std::to_string(m.recall) << std::endl;
            file << "f1 : " + std::to_string(m.f1) << std::endl;
        }


    }

    //关闭设备
    Device::Close(device);
    return 0;
}
