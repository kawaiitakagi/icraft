#pragma once

#include <icraft-xir/core/network.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include <unordered_set>
#include <cmath>
#include <algorithm>


//根据每个head数，将原本1维的norm分组
std::vector<std::vector<float>> set_norm_by_head(int NOH, int parts, std::vector<float>& normalratio) {
    std::vector<std::vector<float>> _norm;
    for (size_t i = 0; i < NOH; i++)
    {
        std::vector<float> _norm_;
        for (size_t j = 0; j < parts; j++)
        {
            _norm_.push_back(normalratio[i * parts + j]);

        }
        _norm.push_back(_norm_);
    }
    return _norm;
}


class  NetInfo_yolov8 {

public:
    icraft::xir::Network network;
    std::vector<std::vector<int>> i_shape;
    std::vector<std::vector<int>> o_shape;
    std::vector<float> o_scale;
    std::unordered_set<std::string> fpga_op;
    std::vector<Cubic> i_cubic;
    std::vector<Cubic> o_cubic;
    int inp_shape_opid = 0;
    int bit = 8;
    bool resize_on = false;
    bool swaporder_on = false;
    bool ImageMake_on = false;
    bool DetPost_on = false;
    Operation DetPost_op;
    std::unordered_set<std::string> fpgaOPlist(icraft::xir::Network& network) {
        std::unordered_set<std::string> customop_set;
        auto oplist = network->ops;
        for (const auto& op : oplist) {
            if (op->typeKey().find("Resize") != std::string::npos) {
                this->resize_on = true;
            }
            if (op->typeKey().find("SwapOrder") != std::string::npos) {
                this->swaporder_on = true;
            }
            if (op->typeKey().find("ImageMakeNode") != std::string::npos) {
                this->ImageMake_on = true;

            }
            if (op->typeKey().find("DetPostNode") != std::string::npos) {
                this->DetPost_on = true;
                this->DetPost_op = op;

            }
            //std::cout << std::string(op->typeKey()) << std::endl;
            if (op->typeKey().find("customop") != std::string::npos) {
                customop_set.insert(std::string(op->typeKey()));
            }
        }
        if (this->resize_on) this->inp_shape_opid++;
        if (this->swaporder_on) this->inp_shape_opid++;
        return customop_set;
    }
    NetInfo_yolov8(icraft::xir::Network& network) {
        this->network = network;
        this->fpga_op = fpgaOPlist(this->network);

        auto oplist = network->ops;
        for (auto i : oplist[this->inp_shape_opid]->outputs) {
            this->i_shape.push_back(i.tensorType()->shape);
        }
        int count = 0;

        for (auto i : oplist[-1]->inputs) {
            this->o_shape.push_back(i.tensorType()->shape);

        }
        if (this->DetPost_on) {
            for (auto i : this->DetPost_op->inputs) {
                //std::cout << i->dtype.getNormratio().value() << std::endl;
                this->o_scale.emplace_back(i->dtype.getNormratio().value()[0]);

            }
        }
        //std::cout << oplist[-1]->inputs[0]->dtype.getStorageType().bits() << std::endl;
        this->bit = oplist[-1]->inputs[0]->dtype.getStorageType().bits();
        Cubic temp_cubic;
        for (auto i : this->i_shape) {
            if (i.size() == 4) {
                temp_cubic.h = i[1];
                temp_cubic.w = i[2];
                temp_cubic.c = i[3];
                this->i_cubic.emplace_back(temp_cubic);
            }
        }
        for (auto i : this->o_shape) {
            if (i.size() == 4) {
                temp_cubic.h = i[1];
                temp_cubic.w = i[2];
                temp_cubic.c = i[3];
                this->o_cubic.emplace_back(temp_cubic);
            }
        }

        std::cout << "NetInfo init done" << std::endl;
    }
    void ouput_allinfo() {
        std::cout << "ishape" << std::endl;
        for (auto i : i_shape) {
            for (auto ii : i) {
                std::cout << ii << ",";
            }
            std::cout << std::endl;
        }

        std::cout << "o_shape" << std::endl;
        for (auto i : i_shape) {
            for (auto ii : i) {
                std::cout << ii << ",";
            }
            std::cout << std::endl;
        }
        std::cout << "o_scale" << std::endl;
        for (auto i : o_scale) {

            std::cout << i << std::endl;

        }
        std::cout << "fpga_op" << std::endl;

        for (auto it = fpga_op.begin(); it != fpga_op.end(); ++it) {
            std::cout << *it << std::endl;
        }
        std::cout << "i_cubic" << std::endl;
        for (auto i : i_cubic) {
            std::cout << "h：" << i.h << "," << "w：" << i.w << "," << "c：" << i.c << " " << std::endl;
        }
        std::cout << "o_cubic" << std::endl;
        for (auto i : o_cubic) {
            std::cout << "h：" << i.h << "," << "w：" << i.w << "," << "c：" << i.c << " " << std::endl;
        }
        std::cout << "inp_shape_opid" << std::endl;
        std::cout << this->inp_shape_opid << std::endl;

    }


};


void saveScoresToFile(const std::vector<float>& score_list, const std::string& name, const std::string& resRoot) {
    // 去除原始文件名中的 ".JPEG" 后缀
    size_t pos = name.find(".JPEG");
    std::string baseName = (pos != std::string::npos) ? name.substr(0, pos) : name;

    // 构建完整的输出文件路径
    std::string filePath = resRoot + '/' + baseName + ".txt";

    // 打开文件
    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
        std::cerr << "Cann't open the file " << filePath << " \n";
        return;
    }

    // 写入分数到文件
    for (float score : score_list) {
        outFile << score << std::endl;
    }

    // 关闭文件
    outFile.close();
    //std::cout << "File " << filePath << " has been saved. \n";
}

