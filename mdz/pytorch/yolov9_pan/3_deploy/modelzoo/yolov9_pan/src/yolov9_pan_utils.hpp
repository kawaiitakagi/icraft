#pragma once

#include <icraft-xir/core/network.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include "opencv2/opencv.hpp"
#include <unordered_set>


std::vector<int> _getReal_out_channles(std::vector<int>ori_out_channles, int bits, int NOA) {
    int MINC = 0;
    int MAXC = 0;
    if (bits == 8) {

        MAXC = 64;
        MINC = 8;
    }
    else if (bits == 16) {
        MAXC = 32;
        MINC = 4;
    }
    else {
        throw "bits != 8 or 16";
        exit(EXIT_FAILURE);
    }
    //calc for anchor length. the last part should be supplemented with the integral multiple of min channel

    auto _last_c = [&](int ori_c)->int {
        return ceil((float)ori_c / (float)MINC) * MINC + MINC;
    };

    //calc for anchor length. the !last part should be supplemented with the integral multiple of max channel
    auto _mid_c = [&](int ori_c)->int {
        return ceil((float)ori_c / (float)MAXC) * MAXC;
    };


    int anchor_length = 0;
    switch (ori_out_channles.size()) {
    case 1: {
        int oneAnchor = ori_out_channles[0] / NOA;
        int anchor_length = _last_c(oneAnchor);
        return  std::vector<int>{ NOA* anchor_length };
    }
    case 2: {
        anchor_length = _last_c(ori_out_channles[1])
            + _mid_c(ori_out_channles[0]);
        return  std::vector<int>{ _mid_c(ori_out_channles[0]), _last_c(ori_out_channles[1]) };

    }
    case 3: {
        anchor_length = _last_c(ori_out_channles[2])
            + _mid_c(ori_out_channles[1]) + _mid_c(ori_out_channles[0]);
        return  std::vector<int>{ _mid_c(ori_out_channles[0]), _mid_c(ori_out_channles[1]),
            _last_c(ori_out_channles[2]) };
    }
    default: {
        throw "parts > 3, DetPost֧支持1个bbox的信息最多分散在3层中!";
        exit(EXIT_FAILURE);
    }

    }
}


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




std::vector<int> all_instances_ids = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 84, 85, 86, 87, 88, 89, 90,
};

std::vector<int> all_stuff_ids = {
    92, 93, 94, 95, 96, 97, 98, 99, 100,
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
    121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
    131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
    141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
    151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
    161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
    171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
    181, 182, 183, 0,
};

std::vector<int> getCocoIds(const std::string& name = "semantic") {
    if (name == "instances") {
        return all_instances_ids;
    }
    else if (name == "stuff") {
        return all_stuff_ids;
    }
    else if (name == "panoptic") {
        std::vector<int> panoptic_ids = all_instances_ids;
        panoptic_ids.insert(panoptic_ids.end(), all_stuff_ids.begin(), all_stuff_ids.end());
        return panoptic_ids;
    }
    else { // semantic
        std::vector<int> semantic_ids = all_instances_ids;
        semantic_ids.insert(semantic_ids.end(), all_stuff_ids.begin(), all_stuff_ids.end());
        return semantic_ids;
    }
}

int getMappingId(size_t index, const std::string& name = "semantic") {
    std::vector<int> ids = getCocoIds(name);
    if (index >= ids.size()) {
        throw std::out_of_range("Index out of range");
    }
    return ids[index];
}

int getMappingIndex(int id, const std::string& name = "semantic") {
    std::vector<int> ids = getCocoIds(name);
    auto it = std::find(ids.begin(), ids.end(), id);
    if (it == ids.end()) {
        throw std::invalid_argument("ID not found");
    }
    return std::distance(ids.begin(), it);
}