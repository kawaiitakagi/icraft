#pragma once
#include <vector>
#include <string>


std::pair<int, std::vector<int>> _getAnchorLength(std::vector<int>ori_out_channles, int bits, int NOA) {
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
        return std::make_pair(anchor_length, std::vector<int>{ NOA* anchor_length });
    }
    case 2: {
        anchor_length = _last_c(ori_out_channles[1])
            + _mid_c(ori_out_channles[0]);
        return std::make_pair(anchor_length, std::vector<int>{ _mid_c(ori_out_channles[0]), _last_c(ori_out_channles[1]) });

    }
    case 3: {
        anchor_length = _last_c(ori_out_channles[2])
            + _mid_c(ori_out_channles[1]) + _mid_c(ori_out_channles[0]);
        return std::make_pair(anchor_length, std::vector<int>{ _mid_c(ori_out_channles[0]), _mid_c(ori_out_channles[1]),
            _last_c(ori_out_channles[2]) });
    }
    default: {
        throw "parts > 3, DetPost֧支持1个bbox的信息最多分散在3层中!";
        exit(EXIT_FAILURE);
    }

    }
}