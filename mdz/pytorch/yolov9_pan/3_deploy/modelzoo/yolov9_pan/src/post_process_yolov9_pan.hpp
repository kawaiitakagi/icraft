#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <icraft-xrt/core/tensor.h>
#include <icraft-xrt/dev/host_device.h>

#include "yolov9_pan_utils.hpp"

#include <modelzoo_utils.hpp>


using namespace icraft::xrt;
struct Grid {
    uint16_t location_x = 0;
    uint16_t location_y = 0;
    uint16_t anchor_index = 0;
};

std::vector<std::tuple<int, float, cv::Rect2f, std::vector<float>>> nms_soft_mask(std::vector<int>& id_list, std::vector<float>& socre_list, std::vector<cv::Rect2f>& box_list, std::vector<std::vector<float>>& mask_info, float IOU, int max_nms = 3000) {
    std::vector<std::tuple<int, float, cv::Rect2f, std::vector<float>>> filter_res;
    std::vector<std::tuple<int, float, cv::Rect2f, std::vector<float>>> nms_res;
    auto bbox_num = id_list.size();
    for (size_t i = 0; i < bbox_num; i++)
    {
        filter_res.push_back({ id_list[i],socre_list[i],box_list[i],mask_info[i] });
    }

    std::stable_sort(filter_res.begin(), filter_res.end(),
        [](const std::tuple<int, float, cv::Rect2f, std::vector<float>>& tuple1, const std::tuple<int, float, cv::Rect2f, std::vector<float>>& tuple2) {
            return std::get<1>(tuple1) > std::get<1>(tuple2);
        }
    );

    int idx = 0;
    for (auto res : filter_res) {
        bool keep = true;
        for (int k = 0; k < nms_res.size() && keep; ++k) {
            if (std::get<0>(res) == std::get<0>(nms_res[k])) {
                if (jaccardDistance(std::get<2>(res), std::get<2>(nms_res[k])) <= IOU) {
                    keep = false;
                }
                //if (1.f - jaccardDistance(std::get<2>(res), std::get<2>(nms_res[k])) > IOU) {
                //    keep = false;
                //}
            }

        }
        if (keep == true)
            nms_res.emplace_back(res);
        if (idx > max_nms) {
            break;
        }
        idx++;
    }
    return nms_res;
};

/**
 *   nms_hard,ʹ��˵��
 *   ����������������Ϊ500����nms_hard��ʱԼ0.638ms
 *   ����������������Ϊ100����nms_hard��ʱԼ0.297ms
 *   �����ռ������С��30��������£�����nms_soft���nms_hard�ٶȿ졣
 */
std::vector<std::tuple<int, float, cv::Rect2f, std::vector<float>>> nms_hard_mask(std::vector<cv::Rect2f>& box_list, std::vector<float>& socre_list, std::vector<int>& id_list, std::vector<std::vector<float>>& mask_info, const float& conf, const float& iou, icraft::xrt::Device& device, int max_nms = 5000) {
    std::vector<int> nms_indices;
    std::vector<std::pair<float, int> > score_index_vec;
    std::vector<std::tuple<int, float, cv::Rect2f, std::vector<float>>> num_res;
    if (box_list.size() == 0) return num_res;
    for (size_t i = 0; i < socre_list.size(); ++i) {
        if (socre_list[i] > conf) {
            score_index_vec.emplace_back(std::make_pair(socre_list[i], i));
        }
    }
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
        [](const std::pair<float, int>& pair1, const std::pair<float, int>& pair2) {return pair1.first > pair2.first; });
    std::vector<int16_t> nms_pre_data;
    int box_num = score_index_vec.size();
    if (box_num > max_nms) {
        box_num = max_nms;
    }
    for (size_t i = 0; i < box_num; ++i) {
        const int idx = score_index_vec[i].second;
        auto x1 = box_list[idx].tl().x;
        if (x1 < 0) x1 = 0;
        auto y1 = box_list[idx].tl().y;
        if (y1 < 0) y1 = 0;
        auto x2 = box_list[idx].br().x;
        auto y2 = box_list[idx].br().y;
        nms_pre_data.push_back((int16_t)x1);
        nms_pre_data.push_back((int16_t)y1);
        nms_pre_data.push_back((int16_t)x2);
        nms_pre_data.push_back((int16_t)y2);
        nms_pre_data.push_back((int16_t)id_list[i]);

    }
    auto nms_data_cptr = nms_pre_data.data();
    auto uregion_ = device.getMemRegion("udma");
    auto udma_chunk_ = uregion_.malloc(10e6);
    auto mapped_base = udma_chunk_->begin.addr();
    udma_chunk_.write(0, (char*)nms_data_cptr, box_num * 10);
    //hard nms config
    float threshold_f = iou;
    uint64_t arbase = mapped_base;
    uint64_t awbase = mapped_base;
    uint64_t reg_base = 0x100001C00;
    //���Ӳ���İ汾��Ϣ�Ƿ���ȷ������ȷ���׳�����
    if (device.defaultRegRegion().read(reg_base + 0x008, true) != 0x23110200) {
        throw std::runtime_error("ERROR :: No NMS HardWare");
    }
    auto group_num = (uint64_t)ceilf((float)box_num / 16.f);
    if (group_num == 0) throw std::runtime_error("ERROR :: group_num == 0");
    auto last_araddr = arbase + group_num * 160 - 8;
    if (last_araddr < arbase) throw std::runtime_error("ERROR :: last_araddr < arbase");
    auto anchor_hpsize = (uint64_t)ceilf((float)box_num / 64.f);
    if (anchor_hpsize == 0) throw std::runtime_error("ERROR :: anchor_hpsize == 0");
    auto last_awaddr = awbase + anchor_hpsize * 8 - 8;
    if (last_awaddr < awbase) throw std::runtime_error("ERROR :: last_awaddr < awbase");

    auto threshold = (uint16_t)(threshold_f * pow(2, 15));
    //config reg
    device.defaultRegRegion().write(reg_base + 0x014, 1, true);
    device.defaultRegRegion().write(reg_base + 0x014, 0, true);
    device.defaultRegRegion().write(reg_base + 0x01C, arbase, true);
    device.defaultRegRegion().write(reg_base + 0x020, awbase, true);
    device.defaultRegRegion().write(reg_base + 0x024, last_araddr, true);
    device.defaultRegRegion().write(reg_base + 0x028, last_awaddr, true);
    device.defaultRegRegion().write(reg_base + 0x02C, group_num, true);
    device.defaultRegRegion().write(reg_base + 0x030, 1, true); //mode 
    device.defaultRegRegion().write(reg_base + 0x034, threshold, true);
    device.defaultRegRegion().write(reg_base + 0x038, anchor_hpsize, true);

    device.defaultRegRegion().write(reg_base + 0x0, 1, true);  //start
    uint64_t reg_done;
    auto start = std::chrono::steady_clock::now();
    do {
        reg_done = device.defaultRegRegion().read(reg_base + 0x004, true);
        std::chrono::duration<double, std::milli> duration = std::chrono::steady_clock::now() - start;
        if (duration.count() > 1000) {
            throw std::runtime_error("NMS Timeout!!!");
        }
    } while (reg_done == 0);
    uint64_t mask_size = (uint64_t)(ceilf((float)box_num / 8.f));
    char* mask = new char[64000];
    udma_chunk_.read(mask, 0, mask_size);

    for (int i = 0; i < score_index_vec.size(); ++i) {
        const int idx = score_index_vec[i].second;
        int mask_index = i / 8;
        if (i % 8 == 0 && ((mask[mask_index] & (uint8_t)1) != 0))
            nms_indices.emplace_back(idx);
        else if (i % 8 == 1 && ((mask[mask_index] & (uint8_t)2) != 0))
            nms_indices.emplace_back(idx);
        else if (i % 8 == 2 && ((mask[mask_index] & (uint8_t)4) != 0))
            nms_indices.emplace_back(idx);
        else if (i % 8 == 3 && ((mask[mask_index] & (uint8_t)8) != 0))
            nms_indices.emplace_back(idx);
        else if (i % 8 == 4 && ((mask[mask_index] & (uint8_t)16) != 0))
            nms_indices.emplace_back(idx);
        else if (i % 8 == 5 && ((mask[mask_index] & (uint8_t)32) != 0))
            nms_indices.emplace_back(idx);
        else if (i % 8 == 6 && ((mask[mask_index] & (uint8_t)64) != 0))
            nms_indices.emplace_back(idx);
        else if (i % 8 == 7 && ((mask[mask_index] & (uint8_t)128) != 0))
            nms_indices.emplace_back(idx);
    }
    delete mask;
    for (auto idx : nms_indices) {
        num_res.push_back({ id_list[idx],socre_list[idx],box_list[idx],mask_info[idx] });
    }
    return num_res;
}


std::tuple<std::vector<std::vector<float>>, cv::Mat> coordTrans_point(std::vector<std::tuple<int, float, cv::Rect2f, std::vector<float> >>& nms_res, cv::Mat proto_mat_float, cv::Mat psemask_mat_float, PicPre& img, bool check_border = true) {
    std::vector<std::vector<float>> output_data;
    int left_pad = img.getPad().first;
    int top_pad = img.getPad().second;
    float ratio = img.getRatio().first;

    struct MaskInfo {
        int class_id;
        float score;
        cv::Mat mask;
    };


    auto dst_size = (img.ori_img.cols >= img.ori_img.rows) ? img.ori_img.cols : img.ori_img.rows;
    auto scale_seg = (float)dst_size / 160;
    cv::Mat reshaped_mat = psemask_mat_float.reshape(173, 160);
    //std::cout << "reshaped rows: " << reshaped_mat.rows << std::endl;
    //std::cout << "reshaped cols: " << reshaped_mat.cols << std::endl;
    //std::cout << "reshaped channels: " << reshaped_mat.channels() << std::endl;

    cv::resize(reshaped_mat, reshaped_mat, cv::Size(160 * scale_seg, 160 * scale_seg));
    //std::cout << "resized rows: " << reshaped_mat.rows << std::endl;
    //std::cout << "resized cols: " << reshaped_mat.cols << std::endl;
    //std::cout << "resized channels: " << reshaped_mat.channels() << std::endl;

    cv::Mat final_reshaped_mat = reshaped_mat.reshape(1, dst_size* dst_size);
    //std::cout << "final rows: " << final_reshaped_mat.rows << std::endl;
    //std::cout << "final cols: " << final_reshaped_mat.cols << std::endl;
    //std::cout << "final channels: " << final_reshaped_mat.channels() << std::endl;


    cv::Mat max_idx(final_reshaped_mat.rows, 1, CV_32S);

    for (int i = 0; i < final_reshaped_mat.rows; ++i) {
        cv::Mat row = final_reshaped_mat.row(i);
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(row, &minVal, &maxVal, &minLoc, &maxLoc);
        max_idx.at<int>(i, 0) = maxLoc.x;
    }

    std::set<int> unique_labels_set;
    for (int i = 0; i < final_reshaped_mat.rows; ++i) {
        unique_labels_set.insert(max_idx.at<int>(i));
    }
    std::vector<int> unique_labels(unique_labels_set.begin(), unique_labels_set.end());
    //// 将 max_idx 转换为 (160, 160) 形状
    //cv::Mat max_idx_reshaped = max_idx.reshape(1, 160);

    //std::cout << "Part-1 " << std::endl;
    //std::cout << max_idx.rows << std::endl;
    //std::cout << max_idx.cols << std::endl;

    cv::Mat semask = max_idx.reshape(1, dst_size);
    semask.convertTo(semask, CV_8UC1); // shape: 640 x 640


    float gain = std::min((float)semask.rows / img.ori_img.rows, (float)semask.cols / img.ori_img.cols);
    float pad_y = (semask.rows - img.ori_img.rows * gain) / 2;
    float pad_x = (semask.cols - img.ori_img.cols * gain) / 2;

    int top = static_cast<int>(pad_y);
    int left = static_cast<int>(pad_x);
    int bottom = static_cast<int>(semask.rows - pad_y);
    int right = static_cast<int>(semask.cols - pad_x);

    top = std::max(0, top);
    left = std::max(0, left);
    //bottom = std::min(semask.rows - 1, bottom);
    //right = std::min(semask.cols - 1, right);

    cv::Mat semask_2;
    semask_2 = semask(cv::Range(top, bottom), cv::Range(left, right));


    cv::Mat panoptic(semask_2.size(), CV_32SC3, cv::Scalar(0, 0, 0));
    cv::Mat stuff = cv::Mat::zeros(semask_2.size(), CV_32S);
    cv::Mat stuff2 = cv::Mat::zeros(semask_2.size(), CV_32S);
 

    for (int _cls : unique_labels) {
        cv::Mat mask = (semask_2 == _cls);
        int area = cv::countNonZero(mask);

        if (area < 64 * 64) {
            stuff.setTo(255, mask);
            stuff2.setTo(255, mask);
        }
        else {
            if (_cls < 80) {
                stuff.setTo(255, mask);
                stuff2.setTo(255, mask);
            }
            else {
                stuff.setTo(getMappingId(_cls, "semantic"), mask);
                stuff2.setTo(getMappingId(_cls, "semantic"), mask);
            }
        }
    }
    //std::cout << "Part1 " << std::endl;

    std::vector<cv::Mat> channels(3);
    channels[0] = stuff;// 红色通道
    channels[1] = cv::Mat::zeros(stuff.size(), stuff.type()); // 绿色通道
    channels[2] = stuff2; // 蓝色通道

    //cv::merge(channels, panoptic);
    //std::vector<cv::Mat> channels(3);
    //cv::split(panoptic, channels);
    

    int inst_id = 0;

//////////////////////////////////////////////////////////////////////////////

    //size_t numArrays = nms_res.size();
    //std::cout << "Number of arrays in nms_res: " << numArrays << std::endl;

    std::vector<MaskInfo> masks_info_container;

    for (auto&& res : nms_res) {
        float class_id = std::get<0>(res);
        float score = std::get<1>(res);
        auto box = std::get<2>(res);
        float x1 = (box.tl().x - left_pad) / ratio;
        float y1 = (box.tl().y - top_pad) / ratio;
        float x2 = (box.br().x - left_pad) / ratio;
        float y2 = (box.br().y - top_pad) / ratio;
        if (check_border) {
            x1 = checkBorder(x1, 0.f, (float)img.src_img.cols);
            y1 = checkBorder(y1, 0.f, (float)img.src_img.rows);
            x2 = checkBorder(x2, 0.f, (float)img.src_img.cols);
            y2 = checkBorder(y2, 0.f, (float)img.src_img.rows);
        }
        float w = x2 - x1;
        float h = y2 - y1;

        //std::cout << "bbox: " << box << std::endl;
        //bbox：左上角点和wh

        if (w > 0 && h > 0) {

            std::default_random_engine e;
            std::uniform_int_distribution<unsigned> u(10, 200);
            cv::Mat mask = cv::Mat::zeros(cv::Size(img.src_img.cols, img.src_img.rows), img.src_img.type());
            cv::Mat rgb_mask = cv::Mat::zeros(cv::Size(img.src_img.cols, img.src_img.rows), img.src_img.type());

            cv::Scalar color_ = cv::Scalar(u(e), u(e), u(e));

            std::vector<float> outdata = { class_id, x1, y1, w, h, score };

            cv::Mat obj_mask_mat = cv::Mat(32, 1, CV_32F, std::get<3>(res).data());
            //std::cout << "Shape: " << obj_mask_mat.rows << "x" << obj_mask_mat.cols << std::endl;
            //std::cout << "Shape: " << proto_mat_float.rows << "x" << proto_mat_float.cols << std::endl;

            cv::Mat out_mask = proto_mat_float * obj_mask_mat * (-1); // (160*160 x 32)*(32 x 1) = (25600 x 1)
            cv::Mat out_mask_exp;
            cv::exp(out_mask, out_mask_exp);
            cv::Mat out_mask_sigmoid = (1.0 / (1.0 + out_mask_exp));
            cv::Mat masks_1 = out_mask_sigmoid.reshape(1, 160); // (160 x 160)

            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            //cv::resize(masks_1, masks_1, cv::Size(img.ori_img.cols, img.ori_img.rows)); // (dst_size x dst_size)
            cv::resize(masks_1, masks_1, cv::Size(160 * scale_seg, 160 * scale_seg));
            cv::Mat binary_mask = masks_1 > 0.5;
            binary_mask.convertTo(binary_mask, CV_8UC1);


            cv::Mat masks_2;
            masks_2 = binary_mask(cv::Range(top, bottom), cv::Range(left, right));
            //std::cout << "rows: " << masks_2.rows << std::endl;
            //std::cout << "cols: " << masks_2.cols << std::endl;
            //cv::resize(masks_2, masks_2, cv::Size(img.ori_img.cols, img.ori_img.rows));

            cv::Mat cropped_mask = cv::Mat::zeros(masks_2.size(), masks_2.type());
            cv::Rect roi(x1, y1, w, h);
            //std::cout << "Rect: " << roi << std::endl;
            masks_2(roi).copyTo(cropped_mask(roi));
            masks_2 = cropped_mask;
            //std::cout << "masks2 rows: " << masks_2.rows << std::endl;
            //std::cout << "masks2 cols: " << masks_2.cols << std::endl;
            //std::cout << "masks2 channels: " << masks_2.channels() << std::endl;
            MaskInfo mask_info = { class_id, score, masks_2 };
            masks_info_container.push_back(mask_info);
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////


            std::vector <std::vector<cv::Point>> points;
            cv::findContours(masks_2, points, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point());
            int id = 0;
            int points_number = 0;
            for (int i = 0; i < points.size(); i++) {
                if (points[i].size() > points_number) {
                    points_number = points[i].size();
                    id = i;
                }
            }
            for (int s = 0; s < points_number; s++) {
                outdata.push_back(points[id][s].x);
                outdata.push_back(points[id][s].y);
                //std::cout<< points[id][s].x <<std::endl;
                //std::cout << points[id][s].y << std::endl;
                //cv::circle(mask, cv::Point(points[id][s].x, points[id][s].y), 1, (255, 255, 255), 6);
            }
            //cv::imshow("coordTrans", mask);
            //cv::waitKey();
            output_data.emplace_back(outdata);
        }
    }

    // 对 masks_info_container 进行排序，按 mask 的非零元素数量降序排序
    std::sort(masks_info_container.begin(), masks_info_container.end(),
        [](const MaskInfo& a, const MaskInfo& b) {
            return cv::countNonZero(a.mask) > cv::countNonZero(b.mask);
        });

    for (auto& mask_info : masks_info_container) {


        int mapped_id = getMappingId(mask_info.class_id, "instances") * 1000 + inst_id;
        int ori_id = getMappingId(mask_info.class_id, "instances");
        //std::cout <<"inner output" << ori_id << std::endl;
        //cv::Mat inst_mask = cv::Mat::zeros(semask_2.size(), CV_32S);
        cv::Mat inst_mask0 = (mask_info.mask != 0);


        channels[0].setTo(ori_id, inst_mask0);
        //channels[2].setTo(mapped_id, masks_2 == 1);
        channels[1].setTo(mapped_id, inst_mask0);

        //cv::merge(channels, panoptic);
        inst_id += 1;
    
    }
  

    cv::Mat null_mask0 = (channels[0] == 255);
    cv::Mat null_mask2 = (channels[2] == 255);
    //null_mask0.convertTo(null_mask0, CV_32S);
    //null_mask2.convertTo(null_mask0, CV_32S);
    channels[0].setTo(0, null_mask0 );
    channels[2].setTo(0, null_mask2 );


    cv::Mat temp;
    cv::divide(channels[1], 1000, temp, 1, CV_32S);
    //channels[1] = temp;
    temp.copyTo(channels[1]);
    //double minVal, maxVal;
    //cv::minMaxLoc(channels[0], &minVal, &maxVal);
    //std::cout << "Min value in channel0: " << minVal << std::endl;
    //std::cout << "Max value in channel0: " << maxVal << std::endl;

    cv::merge(channels, panoptic);
    panoptic.convertTo(panoptic, CV_8UC3);

return std::make_tuple(output_data, panoptic);
}


void visualize_point(std::vector<std::vector<float>>& output_res, std::vector<std::tuple<int, float, cv::Rect2f, std::vector<float> >>& nms_res, cv::Mat proto_mat_float, const cv::Mat& img, const std::string resRoot, const std::string name, const std::vector<std::string>& names) {
    std::default_random_engine e;
    std::uniform_int_distribution<unsigned> u(10, 200);
    int index = 0;
    cv::Mat mask = cv::Mat::zeros(cv::Size(img.cols, img.rows), img.type());
    cv::Mat rgb_mask = cv::Mat::zeros(cv::Size(img.cols, img.rows), img.type());

    for (auto res : output_res) {
        int class_id = (int)res[0];
        float x1 = res[1];
        float y1 = res[2];
        float w = res[3];
        float h = res[4];
        float score = res[5];
        cv::Scalar color_ = cv::Scalar(u(e), u(e), u(e));
        cv::rectangle(img, cv::Rect2f(x1, y1, w, h), color_, 2);
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << score;
        std::string s = std::to_string(class_id) + "_" + names[class_id] + " " + ss.str();
        auto s_size = cv::getTextSize(s, cv::FONT_HERSHEY_DUPLEX, 0.5, 1, 0);
        cv::rectangle(img, cv::Point2f(x1 - 1, y1 - s_size.height - 7), cv::Point2f(x1 + s_size.width, y1 - 2), color_, -1);
        cv::putText(img, s, cv::Point2f(x1, y1 - 2), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 255, 255), 0.2);

        std::vector<cv::Point> contour;
        if (res.size() < 6) {
            throw std::runtime_error("Error: The masks input is None (minimum 6 required).");
        }
        else
        {
            for (size_t i = 6; i < res.size(); i += 2) {
            float x = res[i];
            float y = res[i + 1];
            contour.emplace_back(cv::Point(x, y));
            }
        }

        cv::Mat mask_temp = cv::Mat::zeros(img.rows, img.cols, CV_8UC3); // for colors
        std::vector<std::vector<cv::Point>> contours = { contour };
        cv::fillPoly(mask_temp, contours, color_);
        //cv::Mat obj_mask_mat = cv::Mat(32, 1, CV_32F, std::get<3>(nms_res[index]).data());
        //index++;
        //cv::Mat out_mask = proto_mat_float * obj_mask_mat * (-1);

        //cv::Mat out_mask_exp;

        //cv::exp(out_mask, out_mask_exp);

        //cv::Mat out_mask_sigmoid = (1.0 / (1.0 + out_mask_exp));

        //cv::Mat masks_1 = out_mask_sigmoid.reshape(1, 160); // (160 x 160)

        ////cv::Rect bbox(x1, y1, w, h);
        ////img is original image
        //auto dst_size = (img.cols >= img.rows) ? img.cols : img.rows;
        //auto scale_seg = (float)dst_size / 160;

        ////bbox.x = bbox.x / scale_seg;
        ////bbox.width = bbox.width / scale_seg;
        ////bbox.y = bbox.y / scale_seg;
        ////bbox.height = bbox.height / scale_seg;
      

        //// 实例分割可视化

        //cv::Mat masks_2;

        //auto dst_size = (img.cols >= img.rows) ? img.cols : img.rows;

        //auto scale_seg = (float)dst_size / 160;

        //cv::Mat  masks_1 = out_mask_sigmoid.reshape(1, 160);

        //cv::resize(masks_1, masks_1, cv::Size(160 * scale_seg, 160 * scale_seg));

        //masks_2 = masks_1(cv::Range(0, img.rows), cv::Range(0, img.cols));

        //cv::Mat mask_tmp = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8UC1);

        //// 将masked目标从原图上裁剪下来
        //cv::Mat masks_3 = masks_2(cv::Range(y1, y1 + h), cv::Range(x1, x1 + w));

        //// 将目标区域粘贴到原图的黑色背景上
        //masks_3.copyTo(mask_tmp(cv::Range(y1, y1 + h), cv::Range(x1, x1 + w)));

        //cv::Mat rgb_mask_test = cv::Mat::zeros(cv::Size(img.cols, img.rows), img.type());
        //add(rgb_mask_test, color_, rgb_mask_test, mask_tmp);
        //mask = mask + rgb_mask_test;
        mask = mask + mask_temp;
    }
    cv::Mat out = img*0.9 + mask * 0.6;
#ifdef _WIN32
    cv::imshow("results", out);
    cv::waitKey(0);
#elif __linux__
    std::string save_path = resRoot + '/' + name;
    std::regex rgx("\\.(?!.*\\.)"); // 匹配最后一个点号（.）之前的位置，且该点号后面没有其他点号
    std::string RES_PATH = std::regex_replace(save_path, rgx, "_result.");
    cv::imwrite(RES_PATH, out);
#endif

}

// output : { class_id, x1, y1, w, h, score, contours } in 640 * 640 image
void saveRes_point(std::vector<std::vector<float>>& output_res, std::string resRoot, std::string name) {
    std::string save_path = resRoot + '/' + name;
    std::regex reg(R"(\.(\w*)$)");
    save_path = std::regex_replace(save_path, reg, ".txt");
    std::ofstream outputFile(save_path);
    if (!outputFile.is_open()) {
        std::cout << "Create txt file fail." << std::endl;
    }
    outputFile << std::fixed << std::setprecision(6);
    for (auto i : output_res) {
        for (auto j : i) {
            if (j == static_cast<int>(j)) {
                outputFile << static_cast<int>(j) << " ";
            }
            else {
                outputFile << j << " ";
            }
        }
        outputFile << "\n";
    }
    outputFile.close();
}

void savePanopticImage(cv::Mat panoptic, std::string resRoot, std::string name) {
    std::string save_path = resRoot + '/' + name;
    std::regex reg(R"(\.(\w*)$)");
    save_path = std::regex_replace(save_path, reg, ".png");
    cv::imwrite(save_path, panoptic);
}

std::tuple<std::vector<std::vector<float>>, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> coordTrans_mask(const Tensor& icraft_results, std::vector<std::tuple<int, float, cv::Rect2f, std::vector<float>>>& nms_res, PicPre& img,
    int protoh, int protow, int mask_channel, int bit, bool check_border = true) {
    std::vector<std::vector<float>> output_data;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mask_res;
    int left_pad = img.getPad().first;
    int top_pad = img.getPad().second;
    float ratio = img.getRatio().first;
    auto proto = icraft_results.to(HostDevice::MemRegion());
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mask_proto;

    auto mask_datasize = protoh * protow;
    auto proto_ = (float*)proto.data().cptr();
    mask_proto = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>
        (proto_, mask_channel, mask_datasize);
    //if (bit == 8) {
    //    auto proto_8 = (int8_t*)proto.data().cptr();
    //    mask_proto = Eigen::Map<Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>>
    //        (proto_8, mask_channel, mask_datasize).cast<float>();
    //}
    //else if (bit == 16) {
    //    auto proto_16 = (int16_t*)proto.data().cptr();
    //    Eigen::TensorMap<Eigen::Tensor<int16_t, 3, Eigen::RowMajor>, Eigen::Aligned> tensor_map(proto_16, 2, 25600, 16);
    //    Eigen::array<Eigen::Index, 3> permute_dims({ 0, 2, 1 });
    //    Eigen::Tensor<int16_t, 3, Eigen::RowMajor> transposed = tensor_map.shuffle(permute_dims);
    //    Eigen::array<Eigen::Index, 2> new_dims({ 25600, 32 });
    //    Eigen::TensorMap<Eigen::Tensor<int16_t, 2, Eigen::RowMajor>, Eigen::Aligned> reshaped(transposed.data(), new_dims);
    //    mask_proto = Eigen::Map<Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>>
    //        (reshaped.data(), mask_datasize, mask_channel).transpose().cast<float>();

    //}
    //else {
    //    std::cerr << "yolo_seg coordTrans wrong bits!";
    //}
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mask_kernel;
    //std::cout <<"ratio: " << ratio << std::endl;
    int objnum = nms_res.size();
    mask_kernel.resize(objnum, mask_channel);
    int obj_idn = 0;
    for (auto&& res : nms_res) {
        float class_id = std::get<0>(res);
        float score = std::get<1>(res);
        auto box = std::get<2>(res);
        float x1 = (box.tl().x - left_pad) / ratio;
        float y1 = (box.tl().y - top_pad) / ratio;
        float x2 = (box.br().x - left_pad) / ratio;
        float y2 = (box.br().y - top_pad) / ratio;
        if (check_border) {
            x1 = checkBorder(x1, 0.f, (float)img.src_img.cols);
            y1 = checkBorder(y1, 0.f, (float)img.src_img.rows);
            x2 = checkBorder(x2, 0.f, (float)img.src_img.cols);
            y2 = checkBorder(y2, 0.f, (float)img.src_img.rows);
        }
        float w = x2 - x1;
        float h = y2 - y1;
        //bbox�����Ͻǵ��wh
        output_data.emplace_back(std::vector<float>({ class_id, x1, y1, w, h, score }));
        std::vector<float> masks_in = std::get<3>(res);
        Eigen::VectorXf vec(mask_channel);
        vec = Eigen::Map<Eigen::VectorXf>(masks_in.data(), mask_channel);
        mask_kernel.row(obj_idn) = vec;
        obj_idn++;
    }
    mask_res.resize(objnum, mask_datasize);
    mask_res = mask_kernel * mask_proto;
    _sigmoid(mask_res.data(), mask_datasize * objnum, true);
    return { output_data, mask_res };
}


Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Trans_mask_plin_woc(const Tensor& icraft_results, std::vector<std::tuple<int, float, cv::Rect2f, std::vector<float>>>& nms_res,
    int protoh, int protow, int mask_channel, int bit, float mask_normratio, bool check_border = true) {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mask_res;
    //auto forward_start001 = std::chrono::system_clock::now();

    auto proto = icraft_results.to(HostDevice::MemRegion());
    //auto forward_start002 = std::chrono::system_clock::now();
    //std::cout << "icraft_results.to(HostDevice::MemRegion()) = " << double(std::chrono::duration_cast<std::chrono::microseconds>(forward_start002 - forward_start001).count() / 1000.f) << "ms" << '\n';

    Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic> mask_proto_int8_t;
    auto sp = icraft_results.dtype()->shape;
    auto mask_datasize = protoh * protow;
    //auto proto_ = (float*)proto.data().cptr();
    auto proto_ = (int8_t*)proto.data().cptr();

    mask_proto_int8_t = Eigen::Map<Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>>
        (proto_, mask_channel, mask_datasize);
    //auto forward_start010 = std::chrono::system_clock::now();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mask_proto = mask_proto_int8_t.cast<float>();
    //auto forward_start020 = std::chrono::system_clock::now();
    //std::cout << "cast = " << double(std::chrono::duration_cast<std::chrono::microseconds>(forward_start020 - forward_start010).count() / 1000.f) << "ms" << '\n';

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mask_kernel;
    //std::cout <<"ratio: " << ratio << std::endl;
    int objnum = nms_res.size();
    mask_kernel.resize(objnum, mask_channel);
    int obj_idn = 0;
    for (auto&& res : nms_res) { // 2-6ms
        std::vector<float> masks_in = std::get<3>(res);
        Eigen::VectorXf vec(mask_channel);
        vec = Eigen::Map<Eigen::VectorXf>(masks_in.data(), mask_channel);
        mask_kernel.row(obj_idn) = vec;
        obj_idn++;
    }
    //auto forward_start03 = std::chrono::system_clock::now();
    //std::cout << "res : nms_res = " << double(std::chrono::duration_cast<std::chrono::microseconds>(forward_start03 - forward_start020).count() / 1000.f) << "ms" << '\n';

    mask_res.resize(objnum, mask_datasize);// 0.088ms

    //auto forward_start003 = std::chrono::system_clock::now();

    mask_res = mask_kernel * mask_proto * mask_normratio;

    //auto forward_start04 = std::chrono::system_clock::now();
    //std::cout << "mask_kernel * mask_proto * mask_normratio = " << double(std::chrono::duration_cast<std::chrono::microseconds>(forward_start04 - forward_start003).count() / 1000.f) << "ms" << '\n';
    return  mask_res;
}
void visualize_mask(std::vector<std::vector<float>>& output_res, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& mask_res, const cv::Mat& img, const std::string resRoot, const std::string name, const std::vector<std::string>& names, int protoh, int protow) {
    std::default_random_engine e;
    std::uniform_int_distribution<unsigned> u(10, 200);
    int index = 0;
    cv::Mat mask = cv::Mat::zeros(cv::Size(img.cols, img.rows), img.type());
    cv::Mat rgb_mask = cv::Mat::zeros(cv::Size(img.cols, img.rows), img.type());
    auto dst_size = (img.cols >= img.rows) ? img.cols : img.rows;
    auto mask_size = (protoh >= protow) ? protoh : protow;
    auto ratio = (float)dst_size / (float)mask_size;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mask_single;
    for (auto res : output_res) {
        int class_id = (int)res[0];
        float x1 = res[1];
        float y1 = res[2];
        float w = res[3];
        float h = res[4];
        float score = res[5];
        cv::Scalar color_ = cv::Scalar(u(e), u(e), u(e));
        cv::rectangle(img, cv::Rect2f(x1, y1, w, h), color_, 2);
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << score;
        std::string s = std::to_string(class_id) + "_" + names[class_id] + " " + ss.str();
        auto s_size = cv::getTextSize(s, cv::FONT_HERSHEY_DUPLEX, 0.5, 1, 0);
        cv::rectangle(img, cv::Point2f(x1 - 1, y1 - s_size.height - 7), cv::Point2f(x1 + s_size.width, y1 - 2), color_, -1);
        cv::putText(img, s, cv::Point2f(x1, y1 - 2), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 255, 255), 0.2);
        //��ȡÿ�������mask
        auto pic_size = protow * protoh;
        Eigen::VectorXf v(pic_size);
        v = mask_res.row(index);
        mask_single = Eigen::Map<Eigen::MatrixXf>(v.data(), protoh, protow).transpose();
        cv::Mat masks_1;
        cv::eigen2cv(mask_single, masks_1);
        cv::resize(masks_1, masks_1, cv::Size(protow * ratio, protoh * ratio));
        // ��maskedĿ���ԭͼ�ϲü�����
        cv::Mat masks_3 = masks_1(cv::Range(y1, y1 + h), cv::Range(x1, x1 + w));
        //���ɱ�������ճ������mask
        cv::Mat mask_tmp = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8UC1);
        masks_3.copyTo(mask_tmp(cv::Range(y1, y1 + h), cv::Range(x1, x1 + w)));
        //����colorճ��mask��
        add(mask, color_, mask, mask_tmp);
        index += 1;
    }
    cv::Mat out = img + mask * 0.5;
#ifdef _WIN32
    cv::imshow("results", out);
    cv::waitKey(0);
#elif __linux__
    std::string save_path = resRoot + '/' + name;
    std::regex rgx("\\.(?!.*\\.)"); // ƥ�����һ����ţ�.��֮ǰ��λ�ã��Ҹõ�ź���û���������
    std::string RES_PATH = std::regex_replace(save_path, rgx, "_result.");
    cv::imwrite(RES_PATH, out);
#endif

}

void saveRes_mask(std::vector<std::vector<float>>& output_res, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& mask_res, std::string resRoot, std::string name, int protoh, int protow) {
    auto box_path = resRoot + "\\box";
    auto mask_path = resRoot + "\\mask";
    checkDir(box_path);
    checkDir(mask_path);

    std::string box_save_path = box_path + "\\" + name;
    std::string mask_save_path = mask_path + "\\" + name;

    std::regex reg(R"(\.(\w*)$)");
    box_save_path = std::regex_replace(box_save_path, reg, ".txt");
    mask_save_path = std::regex_replace(mask_save_path, reg, ".bin");

    std::ofstream outputFileB(box_save_path);
    std::ofstream outputFileM(mask_save_path, std::ios::out | std::ios::binary);

    if (!outputFileB.is_open()) {
        std::cout << "Create BOX txt file fail." << std::endl;
    }
    if (!outputFileM.is_open()) {
        std::cout << "Create MASK txt file fail." << std::endl;
    }

    int index = 0;
    for (auto i : output_res) {
        for (auto j : i) {
            outputFileB << j << " ";
        }
        outputFileB << "\n";

        auto pic_size = protow * protoh;
        Eigen::VectorXf v(pic_size);
        v = mask_res.row(index);
        outputFileM.write((const char*)(v.data()), pic_size * sizeof(float));
        index++;
    }
    outputFileB.close();
    outputFileM.close();
}


std::vector<float> get_stride(NetInfo& netinfo) {
    std::vector<float> stride;
    for (auto i : netinfo.o_cubic) {
        stride.emplace_back(netinfo.i_cubic[0].h / i.h);
    }
    return stride;
};

template <typename T>
Grid get_grid(int bits, T* tensor_data, int base_addr, int anchor_length) {
    Grid grid;
    uint16_t anchor_index;
    uint16_t location_y;
    uint16_t location_x;
    if (bits == 8)
    {
        anchor_index = (((uint16_t)tensor_data[base_addr + anchor_length - 1]) << 8) + (uint8_t)tensor_data[base_addr + anchor_length - 2];
        location_y = (((uint16_t)tensor_data[base_addr + anchor_length - 3]) << 8) + (uint8_t)tensor_data[base_addr + anchor_length - 4];
        location_x = (((uint16_t)tensor_data[base_addr + anchor_length - 5]) << 8) + (uint8_t)tensor_data[base_addr + anchor_length - 6];

    }
    else if (bits == 16)
    {
        anchor_index = (uint16_t)tensor_data[base_addr + anchor_length - 1];
        location_y = (uint16_t)tensor_data[base_addr + anchor_length - 2];
        location_x = (uint16_t)tensor_data[base_addr + anchor_length - 3];
    }
    grid.location_x = location_x;
    grid.location_y = location_y;
    grid.anchor_index = anchor_index;
    return grid;
}

template <typename T>
void get_cls_bbox_maskInfo(std::vector<int>& id_list, std::vector<float>& socre_list, std::vector<cv::Rect2f>& box_list, std::vector<std::vector<float>>& mask_info, T* tensor_data, int base_addr,
    Grid& grid, std::vector<float>& SCALE, std::vector<int>& real_out_channles, int bbox_info_channel, int stride, int N_CLASS, float THR_F, bool MULTILABEL, int mask_channel) {
    if (!MULTILABEL) {

        auto class_ptr_start = tensor_data + base_addr;
        auto max_prob_ptr = std::max_element(class_ptr_start, class_ptr_start + N_CLASS);
        int max_index = std::distance(class_ptr_start, max_prob_ptr);
        auto _prob_ = sigmoid(*max_prob_ptr * SCALE[0]);
        auto realscore = _prob_;
        if (realscore > THR_F) {
            //getBbox
            std::vector<float> ltrb = dfl(tensor_data,
                SCALE[1], base_addr + real_out_channles[0], bbox_info_channel);
            float x1 = grid.location_x + 0.5 - ltrb[0];
            float y1 = grid.location_y + 0.5 - ltrb[1];
            float x2 = grid.location_x + 0.5 + ltrb[2];
            float y2 = grid.location_y + 0.5 + ltrb[3];

            float x = ((x2 + x1) / 2.f) * stride;
            float y = ((y2 + y1) / 2.f) * stride;
            float w = (x2 - x1) * stride;
            float h = (y2 - y1) * stride;
            std::vector<float> xywh = { x,y,w,h };
            int seg_ptr_start = base_addr + real_out_channles[0] + real_out_channles[1];
            std::vector<float> one_obj_mask_info;
            for (size_t j = 0; j < mask_channel; j++)
            {
                one_obj_mask_info.emplace_back(tensor_data[seg_ptr_start + j] * SCALE[2]);
            }


            id_list.emplace_back(max_index);
            socre_list.emplace_back(realscore);
            box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),
                (xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
            mask_info.emplace_back(one_obj_mask_info);
        }
    }
    else {
        for (size_t cls_idx = 0; cls_idx < N_CLASS; cls_idx++) {
            //auto realscore = this->getRealScore(tensor_data, obj_ptr_start, norm, i);

            auto _prob_ = sigmoid(tensor_data[base_addr + cls_idx] * SCALE[0]);
            auto realscore = _prob_;
            if (realscore > THR_F) {
                //getBbox
                std::vector<float> ltrb = dfl(tensor_data,
                    SCALE[1], base_addr + real_out_channles[0], bbox_info_channel);
                float x1 = grid.location_x + 0.5 - ltrb[0];
                float y1 = grid.location_y + 0.5 - ltrb[1];
                float x2 = grid.location_x + 0.5 + ltrb[2];
                float y2 = grid.location_y + 0.5 + ltrb[3];

                float x = ((x2 + x1) / 2.f) * stride;
                float y = ((y2 + y1) / 2.f) * stride;
                float w = (x2 - x1) * stride;
                float h = (y2 - y1) * stride;
                std::vector<float> xywh = { x,y,w,h };
                int seg_ptr_start = base_addr + real_out_channles[0] + real_out_channles[1];
                std::vector<float> one_obj_mask_info;
                for (size_t j = 0; j < mask_channel; j++)
                {
                    one_obj_mask_info.emplace_back(tensor_data[seg_ptr_start + j] * SCALE[2]);
                }


                id_list.emplace_back(cls_idx);
                socre_list.emplace_back(realscore);
                box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),
                    (xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
                mask_info.emplace_back(one_obj_mask_info);
            }
        }
    }
}

template <typename T>
void get_cls_bbox(std::vector<int>& id_list, std::vector<float>& socre_list, std::vector<cv::Rect2f>& box_list, T* tensor_data, int base_addr,
    Grid& grid, float& SCALE, int stride,
    std::vector<float> anchor, int N_CLASS, float THR_F, bool MULTILABEL) {
    if (!MULTILABEL) {
        auto _score_ = sigmoid(tensor_data[base_addr + 4] * SCALE);
        auto class_ptr_start = tensor_data + base_addr + 5;
        auto max_prob_ptr = std::max_element(class_ptr_start, class_ptr_start + N_CLASS);
        int max_index = std::distance(class_ptr_start, max_prob_ptr);
        auto _prob_ = sigmoid(*max_prob_ptr * SCALE);
        auto realscore = _score_ * _prob_;
        if (realscore > THR_F) {
            std::vector<float> xywh = sigmoid(tensor_data, SCALE, base_addr, 4);


            xywh[0] = (2 * xywh[0] + grid.location_x - 0.5) * stride;
            xywh[1] = (2 * xywh[1] + grid.location_y - 0.5) * stride;

            xywh[2] = 4 * powf(xywh[2], 2) * anchor[0];
            xywh[3] = 4 * powf(xywh[3], 2) * anchor[1];
            id_list.emplace_back(max_index);
            socre_list.emplace_back(realscore);
            box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),
                (xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
        }
    }
    else {
        for (size_t cls_idx = 0; cls_idx < N_CLASS; cls_idx++) {
            //auto realscore = this->getRealScore(tensor_data, obj_ptr_start, norm, i);

            auto _score_ = sigmoid(tensor_data[base_addr + 4] * SCALE);
            auto _prob_ = sigmoid(tensor_data[base_addr + 5 + cls_idx] * SCALE);
            auto realscore = _score_ * _prob_;
            if (realscore > THR_F) {
                //auto bbox = this->getBbox(tensor_data, norm, obj_ptr_start, location_x, location_y, stride, anchor);
                std::vector<float> xywh = sigmoid(tensor_data, SCALE, base_addr, 4);

                xywh[0] = (2 * xywh[0] + grid.location_x - 0.5) * stride;
                xywh[1] = (2 * xywh[1] + grid.location_y - 0.5) * stride;

                xywh[2] = 4 * powf(xywh[2], 2) * anchor[0];
                xywh[3] = 4 * powf(xywh[3], 2) * anchor[1];

                id_list.emplace_back(cls_idx);
                socre_list.emplace_back(realscore);
                box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),
                    (xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
            }
        }
    }
}
void post_detpost_hard(const std::vector<Tensor>& output_tensors, PicPre& img, NetInfo& netinfo,
    float conf, float iou_thresh, bool MULTILABEL, bool fpga_nms, int N_CLASS, std::vector<std::string>& LABELS,
    bool& show, bool& save, std::string& resRoot, std::string& name, icraft::xrt::Device device, bool& run_sim,
    int& mask_channel, int& protoh, int& protow, std::vector<std::vector<float>>& SCALE, std::vector<int>& real_out_channles, int bbox_info_channel) {

    std::vector<int> id_list;
    std::vector<float> socre_list;
    std::vector<cv::Rect2f> box_list;
    std::vector<std::vector<float>> mask_info;

    std::vector<float> stride = get_stride(netinfo);
    for (size_t i = 0; i < output_tensors.size() - 2; i++) {

        auto host_tensor = output_tensors[i].to(HostDevice::MemRegion());
        int output_tensors_bits = output_tensors[i].dtype()->element_dtype.getStorageType().bits();

        int obj_num = output_tensors[i].dtype()->shape[2];
        int anchor_length = output_tensors[i].dtype()->shape[3];
        if (output_tensors_bits == 16) {
            auto tensor_data = (int16_t*)host_tensor.data().cptr();
            for (size_t obj = 0; obj < obj_num; obj++) {
                int base_addr = obj * anchor_length;
                Grid grid = get_grid(output_tensors_bits, tensor_data, base_addr, anchor_length);
                get_cls_bbox_maskInfo(id_list, socre_list, box_list, mask_info, tensor_data, base_addr, grid, SCALE[i], real_out_channles, bbox_info_channel, stride[i], N_CLASS, conf, MULTILABEL, mask_channel);
            }
        }
        else {
            auto tensor_data = (int8_t*)host_tensor.data().cptr();
            for (size_t obj = 0; obj < obj_num; obj++) {
                int base_addr = obj * anchor_length;
                Grid grid = get_grid(output_tensors_bits, tensor_data, base_addr, anchor_length);
                get_cls_bbox_maskInfo(id_list, socre_list, box_list, mask_info, tensor_data, base_addr, grid, SCALE[i], real_out_channles, bbox_info_channel, stride[i], N_CLASS, conf, MULTILABEL, mask_channel);



            }
        }
    }
    std::vector<std::tuple<int, float, cv::Rect2f, std::vector<float>>> nms_res;
    //std::cout << "number of results before nms = " << id_list.size() << '\n';
    //auto post_start = std::chrono::system_clock::now();
    if (fpga_nms && !run_sim) {
        nms_res = nms_hard_mask(box_list, socre_list, id_list, mask_info, conf, iou_thresh, device);
    }
    else {
        nms_res = nms_soft_mask(id_list, socre_list, box_list, mask_info, iou_thresh);   // ���� ֮ NMS

    }
    auto proto_ = output_tensors[3].to(HostDevice::MemRegion());
    auto dims = proto_.dtype()->shape;
    //std::cout << proto_.dtype()->shape << std::endl;
    auto proto = (float*)proto_.data().cptr();

    auto h = dims[1];
    auto w = dims[2];
    auto c = dims[3];

    cv::Mat proto_mat_float = cv::Mat(h * w, c, CV_32F, proto);

    auto psemask_ = output_tensors[4].to(HostDevice::MemRegion());
    auto dimsp = psemask_.dtype()->shape;
    //std::cout << psemask_.dtype()->shape << std::endl;
    auto psemask = (float*)psemask_.data().cptr();
    auto hp = dimsp[1];
    auto wp = dimsp[2];
    auto cp = dimsp[3];

    cv::Mat psemask_mat_float = cv::Mat(hp * wp, cp, CV_32F, psemask);
    //std::cout << "psemasks dims: "<< dimsp << std::endl;
    std::vector<std::vector<float>> output_res;
    cv::Mat panoptic;
    //std::vector<std::vector<float>> output_res = coordTrans_point(nms_res, proto_mat_float, psemask_mat_float, img);
    std::tie(output_res, panoptic) = coordTrans_point(nms_res, proto_mat_float, psemask_mat_float, img);

#ifdef _WIN32
    if (show) {
        visualize_point(output_res, nms_res, proto_mat_float, img.ori_img, resRoot, name, LABELS);
        //visualize_mask(std::get<0>(output_res), std::get<1>(output_res), img.ori_img, resRoot, name, LABELS, protoh, protow);

    }
#endif
    if (save) {
#ifdef _WIN32
        //saveRes_point(output_res, resRoot, name);
        savePanopticImage(panoptic, resRoot, name);
        //saveRes_mask(std::get<0>(output_res), std::get<1>(output_res), resRoot, name, protoh, protow);

#elif __linux__
        visualize_point(output_res, nms_res, proto_mat_float, img.ori_img, resRoot, name, LABELS);
        //visualize_mask(std::get<0>(output_res), std::get<1>(output_res), img.ori_img, resRoot, name, LABELS, protoh, protow);

#endif
    }


}

void post_detpost_soft(const std::vector<Tensor>& output_tensors, PicPre& img, std::vector<std::string>& LABELS,
    std::vector<std::vector<std::vector<float>>>& ANCHORS, NetInfo& netinfo, int N_CLASS, float conf, float iou_thresh, bool& fpga_nms, icraft::xrt::Device device, bool& run_sim,
    bool& MULTILABEL, bool& show, bool& save, std::string& resRoot, std::string& name, int& mask_channel, int& protoh, int& protow) {

    std::vector<int> id_list;
    std::vector<float> socre_list;
    std::vector<cv::Rect2f> box_list;
    std::vector<std::vector<float>> mask_info;
    std::vector<float> stride = get_stride(netinfo);
    for (int yolo = 0; yolo < output_tensors.size() - 2; yolo = yolo + 3) {
        int _H = output_tensors[yolo].dtype()->shape[1];
        int _W = output_tensors[yolo].dtype()->shape[2];

        auto host_tensor_class = output_tensors[yolo + 0].to(HostDevice::MemRegion());
        auto host_tensor_box = output_tensors[yolo + 1].to(HostDevice::MemRegion());
        auto host_tensor_mask = output_tensors[yolo + 2].to(HostDevice::MemRegion());
        auto tensor_data_class = (float*)host_tensor_class.data().cptr();
        auto tensor_data_box = (float*)host_tensor_box.data().cptr();
        auto tensor_data_mask = (float*)host_tensor_mask.data().cptr();

        for (size_t h = 0; h < _H; h++) {
            for (size_t w = 0; w < _W; w++) {

                auto one_head_stride = stride[yolo];

                auto classPtr = tensor_data_class +
                    h * _W * N_CLASS +
                    w * N_CLASS;
                auto boxPtr = tensor_data_box +
                    h * _W * 64 +
                    w * 64;

                auto maskPtr = tensor_data_mask +
                    h * _W * mask_channel +
                    w * mask_channel;
                if (!MULTILABEL) {

                    float* max_prob_ptr = std::max_element(classPtr, classPtr + N_CLASS);
                    int max_index = std::distance(classPtr, max_prob_ptr);
                    auto _prob_ = sigmoid(*max_prob_ptr);
                    auto realscore = _prob_;
                    if (realscore > conf) {

                        //getBbox
                        std::vector<float> ltrb = dfl(boxPtr,
                            1.0, 0, 64);
                        float x1 = w + 0.5 - ltrb[0];
                        float y1 = h + 0.5 - ltrb[1];
                        float x2 = w + 0.5 + ltrb[2];
                        float y2 = h + 0.5 + ltrb[3];

                        float x_ = ((x2 + x1) / 2.f) * one_head_stride;
                        float y_ = ((y2 + y1) / 2.f) * one_head_stride;
                        float w_ = (x2 - x1) * one_head_stride;
                        float h_ = (y2 - y1) * one_head_stride;
                        std::vector<float> xywh = { x_,y_,w_,h_ };

                        // mask 
                        std::vector<float> one_obj_mask_info;
                        for (size_t j = 0; j < mask_channel; j++)
                        {
                            one_obj_mask_info.emplace_back(*(maskPtr + j));
                        }

                        id_list.emplace_back(max_index);
                        socre_list.emplace_back(realscore);
                        box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),
                            (xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
                        mask_info.emplace_back(one_obj_mask_info);
                    }
                }
                else {
                    for (size_t cls_idx = 0; cls_idx < N_CLASS; cls_idx++) {
                        //auto realscore = this->getRealScore(tensor_data, obj_ptr_start, norm, i);

                        auto _prob_ = sigmoid(*(classPtr + cls_idx));
                        auto realscore = _prob_;
                        if (realscore > conf) {

                            //getBbox
                            std::vector<float> ltrb = dfl(boxPtr,
                                1.0, 0, 64);
                            float x1 = w + 0.5 - ltrb[0];
                            float y1 = h + 0.5 - ltrb[1];
                            float x2 = w + 0.5 + ltrb[2];
                            float y2 = h + 0.5 + ltrb[3];

                            float x_ = ((x2 + x1) / 2.f) * one_head_stride;
                            float y_ = ((y2 + y1) / 2.f) * one_head_stride;
                            float w_ = (x2 - x1) * one_head_stride;
                            float h_ = (y2 - y1) * one_head_stride;
                            std::vector<float> xywh = { x_,y_,w_,h_ };

                            // mask 
                            std::vector<float> one_obj_mask_info;
                            for (size_t j = 0; j < mask_channel; j++)
                            {
                                one_obj_mask_info.emplace_back(*(maskPtr + j));
                            }

                            id_list.emplace_back(cls_idx);
                            socre_list.emplace_back(realscore);
                            box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),
                                (xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
                            mask_info.emplace_back(one_obj_mask_info);
                        }
                    }
                }
            }
        }
    }

    std::vector<std::tuple<int, float, cv::Rect2f, std::vector<float>>> nms_res;
    //std::cout << "number of results before nms = " << id_list.size() << '\n';
    //auto post_start = std::chrono::system_clock::now();
    if (fpga_nms && !run_sim) {
        nms_res = nms_hard_mask(box_list, socre_list, id_list, mask_info, conf, iou_thresh, device);
    }
    else {
        nms_res = nms_soft_mask(id_list, socre_list, box_list, mask_info, iou_thresh);   // ���� ֮ NMS

    }
    auto proto_ = output_tensors[9].to(HostDevice::MemRegion());
    auto dims = proto_.dtype()->shape;
    //std::cout << proto_.dtype()->shape << std::endl;
    auto proto = (float*)proto_.data().cptr();

    auto h = dims[1];
    auto w = dims[2];
    auto c = dims[3];


    cv::Mat proto_mat_float = cv::Mat(h * w, c, CV_32F, proto);

    auto psemask_ = output_tensors[10].to(HostDevice::MemRegion());
    auto dimsp = psemask_.dtype()->shape;
    auto psemask = (float*)psemask_.data().cptr();

    auto hp = dimsp[1];
    auto wp = dimsp[2];
    auto cp = dimsp[3];

    cv::Mat psemask_mat_float = cv::Mat(hp * wp, cp, CV_32F, psemask);

    std::vector<std::vector<float>> output_res;
    cv::Mat panoptic;
    //std::vector<std::vector<float>> output_res = coordTrans_point(nms_res, proto_mat_float, psemask_mat_float, img);
    std::tie(output_res, panoptic) = coordTrans_point(nms_res, proto_mat_float, psemask_mat_float, img);
#ifdef _WIN32
    if (show) {
        visualize_point(output_res, nms_res, proto_mat_float, img.ori_img, resRoot, name, LABELS);
        //visualize_mask(std::get<0>(output_res), std::get<1>(output_res), img.ori_img, resRoot, name, LABELS, protoh, protow);

    }
#endif
    if (save) {
#ifdef _WIN32
        //saveRes_point(output_res, resRoot, name);
        savePanopticImage(panoptic, resRoot, name);
        //saveRes_mask(std::get<0>(output_res), std::get<1>(output_res), resRoot, name, protoh, protow);

#elif __linux__
        visualize_point(output_res, nms_res, proto_mat_float, img.ori_img, resRoot, name, LABELS);
        //visualize_mask(std::get<0>(output_res), std::get<1>(output_res), img.ori_img, resRoot, name, LABELS, protoh, protow);

#endif
    }


}

void post_detpost_hard_old(const std::vector<Tensor>& output_tensors, PicPre& img, NetInfo& netinfo,
    float conf, float iou_thresh, bool MULTILABEL, bool fpga_nms, int N_CLASS, std::vector<std::string>& LABELS,
    bool& show, bool& save, std::string& resRoot, std::string& name, icraft::xrt::Device device, bool& run_sim,
    int& mask_channel, int& protoh, int& protow, std::vector<std::vector<float>>& SCALE, std::vector<int>& real_out_channles, int bbox_info_channel) {

    std::vector<int> id_list;
    std::vector<float> socre_list;
    std::vector<cv::Rect2f> box_list;
    std::vector<std::vector<float>> mask_info;

    std::vector<float> stride = get_stride(netinfo);
    for (size_t i = 0; i < output_tensors.size() - 1; i++) {

        auto host_tensor = output_tensors[i].to(HostDevice::MemRegion());
        int output_tensors_bits = output_tensors[i].dtype()->element_dtype.getStorageType().bits();

        int obj_num = output_tensors[i].dtype()->shape[2];
        int anchor_length = output_tensors[i].dtype()->shape[3];
        if (output_tensors_bits == 16) {
            auto tensor_data = (int16_t*)host_tensor.data().cptr();
            for (size_t obj = 0; obj < obj_num; obj++) {
                int base_addr = obj * anchor_length;
                Grid grid = get_grid(output_tensors_bits, tensor_data, base_addr, anchor_length);
                get_cls_bbox_maskInfo(id_list, socre_list, box_list, mask_info, tensor_data, base_addr, grid, SCALE[i], real_out_channles, bbox_info_channel, stride[i], N_CLASS, conf, MULTILABEL, mask_channel);
            }
        }
        else {
            auto tensor_data = (int8_t*)host_tensor.data().cptr();
            for (size_t obj = 0; obj < obj_num; obj++) {
                int base_addr = obj * anchor_length;
                Grid grid = get_grid(output_tensors_bits, tensor_data, base_addr, anchor_length);
                get_cls_bbox_maskInfo(id_list, socre_list, box_list, mask_info, tensor_data, base_addr, grid, SCALE[i], real_out_channles, bbox_info_channel, stride[i], N_CLASS, conf, MULTILABEL, mask_channel);



            }
        }
    }
    std::vector<std::tuple<int, float, cv::Rect2f, std::vector<float>>> nms_res;
    //std::cout << "number of results before nms = " << id_list.size() << '\n';
    //auto post_start = std::chrono::system_clock::now();
    if (fpga_nms && !run_sim) {
        nms_res = nms_hard_mask(box_list, socre_list, id_list, mask_info, conf, iou_thresh, device);
    }
    else {
        nms_res = nms_soft_mask(id_list, socre_list, box_list, mask_info, iou_thresh);   // ���� ֮ NMS

    }
    //auto post_end = std::chrono::system_clock::now();
    //auto post_time = std::chrono::duration_cast<std::chrono::microseconds>(post_end - post_start);
    //std::cout << "nms_time = " << double(post_time.count() / 1000.f) << '\n';
    //std::cout << "number of results after nms = " << nms_res.size() << '\n';
    //std::vector<std::vector<float>> output_res = coordTrans(nms_res, img);
    std::tuple<std::vector<std::vector<float>>, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_res =
        coordTrans_mask(output_tensors[3], nms_res, img, protoh, protow, mask_channel, netinfo.detpost_bit);
#ifdef _WIN32
    if (show) {
        visualize_mask(std::get<0>(output_res), std::get<1>(output_res), img.ori_img, resRoot, name, LABELS, protoh, protow);

    }
#endif
    if (save) {
#ifdef _WIN32
        saveRes_mask(std::get<0>(output_res), std::get<1>(output_res), resRoot, name, protoh, protow);
#elif __linux__
        visualize_mask(std::get<0>(output_res), std::get<1>(output_res), img.ori_img, resRoot, name, LABELS, protoh, protow);

#endif
    }


}

void post_detpost_soft_old(const std::vector<Tensor>& output_tensors, PicPre& img, std::vector<std::string>& LABELS,
    std::vector<std::vector<std::vector<float>>>& ANCHORS, NetInfo& netinfo, int N_CLASS, float conf, float iou_thresh, bool& fpga_nms, icraft::xrt::Device device, bool& run_sim,
    bool& MULTILABEL, bool& show, bool& save, std::string& resRoot, std::string& name, int& mask_channel, int& protoh, int& protow) {

    std::vector<int> id_list;
    std::vector<float> socre_list;
    std::vector<cv::Rect2f> box_list;
    std::vector<std::vector<float>> mask_info;
    std::vector<float> stride = get_stride(netinfo);
    for (int yolo = 0; yolo < output_tensors.size() - 1; yolo = yolo + 3) {
        int _H = output_tensors[yolo].dtype()->shape[1];
        int _W = output_tensors[yolo].dtype()->shape[2];

        auto host_tensor_class = output_tensors[yolo + 0].to(HostDevice::MemRegion());
        auto host_tensor_box = output_tensors[yolo + 1].to(HostDevice::MemRegion());
        auto host_tensor_mask = output_tensors[yolo + 2].to(HostDevice::MemRegion());
        auto tensor_data_class = (float*)host_tensor_class.data().cptr();
        auto tensor_data_box = (float*)host_tensor_box.data().cptr();
        auto tensor_data_mask = (float*)host_tensor_mask.data().cptr();

        for (size_t h = 0; h < _H; h++) {
            for (size_t w = 0; w < _W; w++) {

                auto one_head_stride = stride[yolo];

                auto classPtr = tensor_data_class +
                    h * _W * N_CLASS +
                    w * N_CLASS;
                auto boxPtr = tensor_data_box +
                    h * _W * 64 +
                    w * 64;

                auto maskPtr = tensor_data_mask +
                    h * _W * mask_channel +
                    w * mask_channel;
                if (!MULTILABEL) {

                    float* max_prob_ptr = std::max_element(classPtr, classPtr + N_CLASS);
                    int max_index = std::distance(classPtr, max_prob_ptr);
                    auto _prob_ = sigmoid(*max_prob_ptr);
                    auto realscore = _prob_;
                    if (realscore > conf) {

                        //getBbox
                        std::vector<float> ltrb = dfl(boxPtr,
                            1.0, 0, 64);
                        float x1 = w + 0.5 - ltrb[0];
                        float y1 = h + 0.5 - ltrb[1];
                        float x2 = w + 0.5 + ltrb[2];
                        float y2 = h + 0.5 + ltrb[3];

                        float x_ = ((x2 + x1) / 2.f) * one_head_stride;
                        float y_ = ((y2 + y1) / 2.f) * one_head_stride;
                        float w_ = (x2 - x1) * one_head_stride;
                        float h_ = (y2 - y1) * one_head_stride;
                        std::vector<float> xywh = { x_,y_,w_,h_ };

                        // mask 
                        std::vector<float> one_obj_mask_info;
                        for (size_t j = 0; j < mask_channel; j++)
                        {
                            one_obj_mask_info.emplace_back(*(maskPtr + j));
                        }

                        id_list.emplace_back(max_index);
                        socre_list.emplace_back(realscore);
                        box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),
                            (xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
                        mask_info.emplace_back(one_obj_mask_info);
                    }
                }
                else {
                    for (size_t cls_idx = 0; cls_idx < N_CLASS; cls_idx++) {
                        //auto realscore = this->getRealScore(tensor_data, obj_ptr_start, norm, i);

                        auto _prob_ = sigmoid(*(classPtr + cls_idx));
                        auto realscore = _prob_;
                        if (realscore > conf) {

                            //getBbox
                            std::vector<float> ltrb = dfl(boxPtr,
                                1.0, 0, 64);
                            float x1 = w + 0.5 - ltrb[0];
                            float y1 = h + 0.5 - ltrb[1];
                            float x2 = w + 0.5 + ltrb[2];
                            float y2 = h + 0.5 + ltrb[3];

                            float x_ = ((x2 + x1) / 2.f) * one_head_stride;
                            float y_ = ((y2 + y1) / 2.f) * one_head_stride;
                            float w_ = (x2 - x1) * one_head_stride;
                            float h_ = (y2 - y1) * one_head_stride;
                            std::vector<float> xywh = { x_,y_,w_,h_ };

                            // mask 
                            std::vector<float> one_obj_mask_info;
                            for (size_t j = 0; j < mask_channel; j++)
                            {
                                one_obj_mask_info.emplace_back(*(maskPtr + j));
                            }

                            id_list.emplace_back(cls_idx);
                            socre_list.emplace_back(realscore);
                            box_list.emplace_back(cv::Rect2f((xywh[0] - xywh[2] / 2),
                                (xywh[1] - xywh[3] / 2), xywh[2], xywh[3]));
                            mask_info.emplace_back(one_obj_mask_info);
                        }
                    }
                }
            }
        }
    }

    std::vector<std::tuple<int, float, cv::Rect2f, std::vector<float>>> nms_res;
    //std::cout << "number of results before nms = " << id_list.size() << '\n';
    //auto post_start = std::chrono::system_clock::now();
    if (fpga_nms && !run_sim) {
        nms_res = nms_hard_mask(box_list, socre_list, id_list, mask_info, conf, iou_thresh, device);
    }
    else {
        nms_res = nms_soft_mask(id_list, socre_list, box_list, mask_info, iou_thresh);   // ���� ֮ NMS

    }
    //auto post_end = std::chrono::system_clock::now();
    //auto post_time = std::chrono::duration_cast<std::chrono::microseconds>(post_end - post_start);
    //std::cout << "nms_time = " << double(post_time.count() / 1000.f) << '\n';
    //std::cout << "number of results after nms = " << nms_res.size() << '\n';
    //std::vector<std::vector<float>> output_res = coordTrans(nms_res, img);
    std::tuple<std::vector<std::vector<float>>, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> output_res =
        coordTrans_mask(output_tensors[9], nms_res, img, protoh, protow, mask_channel, netinfo.detpost_bit);
#ifdef _WIN32
    if (show) {
        visualize_mask(std::get<0>(output_res), std::get<1>(output_res), img.ori_img, resRoot, name, LABELS, protoh, protow);

    }
#endif
    if (save) {
#ifdef _WIN32
        saveRes_mask(std::get<0>(output_res), std::get<1>(output_res), resRoot, name, protoh, protow);
#elif __linux__
        visualize_mask(std::get<0>(output_res), std::get<1>(output_res), img.ori_img, resRoot, name, LABELS, protoh, protow);

#endif
    }


}

// smooth
const float ALPHA = 0.5f;
const float SMOOTH_IOU = 0.80f;

using YoloPostResult = std::tuple<std::vector<int>, std::vector<float>, std::vector<cv::Rect2f>, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>; // box_list, id_list, score_list ,mask_info

YoloPostResult post_detpost_plin_woc(const std::vector<Tensor>& output_tensors, YoloPostResult& last_frame_result, NetInfo& netinfo,
    float conf, float iou_thresh, bool MULTILABEL, bool fpga_nms, int N_CLASS,
    icraft::xrt::Device device, int& mask_channel, int& protoh, int& protow,
    std::vector<std::vector<float>>& SCALE, std::vector<int>& real_out_channles, int bbox_info_channel, float mask_normratio) {

    std::vector<int> id_list;
    std::vector<float> socre_list;
    std::vector<cv::Rect2f> box_list;
    std::vector<std::vector<float>> mask_info;
    //auto forward_start = std::chrono::system_clock::now();
    std::vector<float> stride = get_stride(netinfo);
    for (size_t i = 0; i < output_tensors.size() - 1; i++) {

        auto host_tensor = output_tensors[i + 1].to(HostDevice::MemRegion());
        int output_tensors_bits = output_tensors[i + 1].dtype()->element_dtype.getStorageType().bits();

        int obj_num = output_tensors[i + 1].dtype()->shape[2];
        int anchor_length = output_tensors[i + 1].dtype()->shape[3];
        if (output_tensors_bits == 16) {
            auto tensor_data = (int16_t*)host_tensor.data().cptr();
            for (size_t obj = 0; obj < obj_num; obj++) {
                int base_addr = obj * anchor_length;
                Grid grid = get_grid(output_tensors_bits, tensor_data, base_addr, anchor_length);
                get_cls_bbox_maskInfo(id_list, socre_list, box_list, mask_info, tensor_data, base_addr, grid, SCALE[i], real_out_channles, bbox_info_channel, stride[i], N_CLASS, conf, MULTILABEL, mask_channel);
            }
        }
        else {
            auto tensor_data = (int8_t*)host_tensor.data().cptr();
            for (size_t obj = 0; obj < obj_num; obj++) {
                int base_addr = obj * anchor_length;
                Grid grid = get_grid(output_tensors_bits, tensor_data, base_addr, anchor_length);
                get_cls_bbox_maskInfo(id_list, socre_list, box_list, mask_info, tensor_data, base_addr, grid, SCALE[i], real_out_channles, bbox_info_channel, stride[i], N_CLASS, conf, MULTILABEL, mask_channel);



            }
        }
    }

    std::vector<std::tuple<int, float, cv::Rect2f, std::vector<float>>> nms_res;
    //std::cout << "number of results before nms = " << id_list.size() << '\n';
    //auto post_start = std::chrono::system_clock::now();
    if (fpga_nms) {
        nms_res = nms_hard_mask(box_list, socre_list, id_list, mask_info, conf, iou_thresh, device);
    }
    else {
        nms_res = nms_soft_mask(id_list, socre_list, box_list, mask_info, iou_thresh);   // ���� ֮ NMS

    }
    std::vector<std::tuple<int, float, cv::Rect2f, std::vector<float>>>num_res_3;

    for (int i = 0; i < 3 && i < nms_res.size(); ++i) {
        num_res_3.push_back(nms_res[i]);
    }
    //auto forward_start0 = std::chrono::system_clock::now();
    //auto forward_time_ = std::chrono::duration_cast<std::chrono::microseconds>(forward_start0 - forward_start);
    //std::cout << "cls&bbox&maskinfo post time = " << double(forward_time_.count() / 1000.f) << "ms" << '\n';
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> output_res_mask =
        Trans_mask_plin_woc(output_tensors[0], num_res_3, protoh, protow, mask_channel, mask_normratio, netinfo.detpost_bit);
    //auto forward_start1 = std::chrono::system_clock::now();
    //auto forward_time = std::chrono::duration_cast<std::chrono::microseconds>(forward_start1 - forward_start0);
    //std::cout << "mask proto post time = " << double(forward_time.count() / 1000.f) << "ms" << '\n';
    // // ��ǰ��֡�Ľ����ƽ��
    auto id_list_last_frame = std::get<0>(last_frame_result);
    auto score_list_last_frame = std::get<1>(last_frame_result);
    auto box_list_last_frame = std::get<2>(last_frame_result);

    for (auto idx_score_bbox : num_res_3) {
        for (size_t i = 0; i < box_list_last_frame.size(); ++i) {
            if ((1.f - cv::jaccardDistance(std::get<2>(idx_score_bbox), box_list_last_frame[i])) > SMOOTH_IOU && (std::get<0>(idx_score_bbox) == id_list_last_frame[i])) {

                std::get<2>(idx_score_bbox).x = box_list_last_frame[i].x * ALPHA + std::get<2>(idx_score_bbox).x * (1.0f - ALPHA);
                std::get<2>(idx_score_bbox).y = box_list_last_frame[i].y * ALPHA + std::get<2>(idx_score_bbox).y * (1.0f - ALPHA);
                std::get<2>(idx_score_bbox).width = box_list_last_frame[i].width * ALPHA + std::get<2>(idx_score_bbox).width * (1.0f - ALPHA);
                std::get<2>(idx_score_bbox).height = box_list_last_frame[i].height * ALPHA + std::get<2>(idx_score_bbox).height * (1.0f - ALPHA);
                std::get<1>(idx_score_bbox) = score_list_last_frame[i] * ALPHA + std::get<1>(idx_score_bbox) * (1.0f - ALPHA);
                break;
            }
        }
    }
    std::vector<int> id_list_ret;   //id_list_ret.reserve(nms_indices.size());
    std::vector<float> score_list_ret; //score_list_ret.reserve(nms_indices.size());
    std::vector<cv::Rect2f> box_list_ret; //box_list_ret.reserve(nms_indices.size());
    for (auto idx_score_bbox : num_res_3) {
        // store 
        id_list_ret.emplace_back(std::get<0>(idx_score_bbox));
        score_list_ret.emplace_back(std::get<1>(idx_score_bbox));
        box_list_ret.emplace_back(std::get<2>(idx_score_bbox));
    }
    return YoloPostResult{ id_list_ret, score_list_ret, box_list_ret,output_res_mask };

}
