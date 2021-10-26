#ifndef UTIL_H
#define UTIL_H

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <torch/torch.h>
#include "common.h"

namespace M
{
    static float iou(const bbox_t &bb_test, const bbox_t &bb_gt)
    {
        float inter_l = bb_test.x > bb_gt.x ? bb_test.x : bb_gt.x;
        float inter_t = bb_test.y > bb_gt.y ? bb_test.y : bb_gt.y;
        float inter_r = bb_test.x + bb_test.w < bb_gt.x + bb_gt.w ? bb_test.x + bb_test.w : bb_gt.x + bb_gt.w;
        float inter_b = bb_test.y + bb_test.h < bb_gt.y + bb_gt.h ? bb_test.y + bb_test.h : bb_gt.y + bb_gt.h;
        if (inter_b < inter_t || inter_r < inter_l)
            return 0;
        float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
        float union_area = bb_test.w * bb_test.h + bb_gt.w * bb_gt.h - inter_area;
        if (union_area == 0)
            return 0;
        else
            return inter_area / union_area;
    }

    static torch::Tensor iou_dist(const std::vector<bbox_t> &dets, const std::vector<bbox_t> &trks)
    {
        auto trk_num = trks.size();
        auto det_num = dets.size();
        auto dist = torch::empty({int64_t(trk_num), int64_t(det_num)});
        for (size_t i = 0; i < trk_num; i++) // compute iou matrix as a distance matrix
        {
            for (size_t j = 0; j < det_num; j++)
            {
                dist[i][j] = 1 - iou(trks[i], dets[j]);
            }
        }
        return dist;
    }
}

namespace
{
    bbox_t pad_rect(bbox_t rect, float padding)
    {
        rect.x = std::max(0.0f, rect.x - rect.w * padding);
        rect.y = std::max(0.0f, rect.y - rect.h * padding);
        rect.w = std::min(1 - rect.x, rect.w * (1 + 2 * padding));
        rect.h = std::min(1 - rect.y, rect.h * (1 + 2 * padding));

        return rect;
    }

    bbox_t normalize_rect(bbox_t rect, float w, float h)
    {
        rect.x /= w;
        rect.y /= h;
        rect.w /= w;
        rect.h /= h;
        return rect;
    }

    bbox_t unnormalize_rect(bbox_t rect, float w, float h)
    {
        rect.x *= w;
        rect.y *= h;
        rect.w *= w;
        rect.h *= h;
        return rect;
    }

    cv::Scalar color_map(int64_t n)
    {
        auto bit_get = [](int64_t x, int64_t i)
        { return x & (1 << i); };

        int64_t r = 0, g = 0, b = 0;
        int64_t i = n;
        for (int64_t j = 7; j >= 0; --j)
        {
            r |= bit_get(i, 0) << j;
            g |= bit_get(i, 1) << j;
            b |= bit_get(i, 2) << j;
            i >>= 3;
        }
        return cv::Scalar(b, g, r);
    }

    void draw_text(cv::Mat &img, const std::string &str,
                   const cv::Scalar &color, cv::Point pos, bool reverse = false)
    {
        auto t_size = cv::getTextSize(str, cv::FONT_HERSHEY_PLAIN, 1, 1, nullptr);
        cv::Point bottom_left, upper_right;
        if (reverse)
        {
            upper_right = pos;
            bottom_left = cv::Point(upper_right.x - t_size.width, upper_right.y + t_size.height);
        }
        else
        {
            bottom_left = pos;
            upper_right = cv::Point(bottom_left.x + t_size.width, bottom_left.y - t_size.height);
        }

        cv::rectangle(img, bottom_left, upper_right, color, -1);
        cv::putText(img, str, bottom_left, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255) - color);
    }

    void draw_bbox(cv::Mat &img, bbox_t bbox,
                   const std::string &label = "", const cv::Scalar &color = {0, 0, 0})
    {
        cv::Rect2f rect(bbox.x, bbox.y, bbox.w, bbox.h);
        cv::rectangle(img, rect, color);
        if (!label.empty())
        {
            draw_text(img, label, color, rect.tl());
        }
    }

    void draw_trajectories(cv::Mat &img, const std::map<int, bbox_t> &traj,
                           const cv::Scalar &color = {0, 0, 0})
    {
        if (traj.size() < 2)
            return;

        bbox_t box = traj.begin()->second;
        cv::Rect2f cur(box.x, box.y, box.w, box.h);
        cv::Point2f pt1 = cur.br();
        pt1.x -= cur.width / 2;
        pt1.x *= img.cols;
        pt1.y *= img.rows;

        for (std::map<int, bbox_t>::const_iterator it = ++traj.begin(); it != traj.end(); ++it)
        {
            box = it->second;
            cur = cv::Rect2f(box.x, box.y, box.w, box.h);
            auto pt2 = cur.br();
            pt2.x -= cur.width / 2;
            pt2.x *= img.cols;
            pt2.y *= img.rows;
            cv::line(img, pt1, pt2, color);
            pt1 = pt2;
        }
    }
}

#endif // UTIL_H
