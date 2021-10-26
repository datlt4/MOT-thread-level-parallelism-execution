// #ifndef _COMMON_H
// #define _COMMON_H
#pragma once
#include <opencv2/opencv.hpp>
#include <sstream>
#include <iostream>
#include <chrono>
#include <sstream>

#define USE_MOT

namespace M
{
    template <typename T>
    static std::string cout_vec(std::vector<T> v)
    {
        std::ostringstream oss;
        oss << "[";
        for (int i = 0; i < v.size(); ++i)
        {
            oss << v[i];
            if (i != v.size() - 1)
            {
                oss << ", ";
            }
        }
        oss << "]";
        return oss.str();
    }

    template <typename T>
    static std::string cout_vec(std::vector<std::tuple<int, int>> v)
    {
        std::ostringstream oss;
        oss << "[";
        for (int i = 0; i < v.size(); ++i)
        {
            oss << "(" << std::get<0>(v[i]) << ", " << std::get<1>(v[i]) << ")";
            if (i != v.size() - 1)
            {
                oss << ", ";
            }
        }
        oss << "]";
        return oss.str();
    }
}

struct bbox_t
{
    int obj_id;       // class of object - from range [0, classes-1]
    int track_id;     // tracking id for video (0 - untracked, 1 - inf - tracked object)
    float xc;         // x_center
    float yc;         // y_center
    float x, y, w, h; // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;

    bbox_t() : track_id(-1)
    {
    }

    bbox_t(float x, float y, float w, float h) : track_id{-1}, x{x}, y{y}, w{w}, h{h}, xc{x + w / 2}, yc{y + w / 2} {}
    std::string str()
    {
        std::ostringstream os;
        os << "\tx : " << x << "\ty : " << y << "\tw : " << w << "\th : " << h << "\txc : " << xc << "\tyc : " << yc;
        return os.str();
    }

    void update(bbox_t &box)
    {
        x = box.x;
        y = box.y;
        w = box.w;
        h = box.h;
        xc = box.xc;
        yc = box.yc;
    }

    void update(float bx, float by, float bw, float bh)
    {
        x = bx;
        y = by;
        w = bw;
        h = bh;
        xc = x + bw / 2;
        yc = y + bh / 2;
    }
};

inline std::string bbox_str(bbox_t box)
{
    std::ostringstream os;
    os << "\tx : " << box.x << "\ty : " << box.y << "\tw : " << box.w << "\th : " << box.h << "\txc : " << box.xc << "\tyc : " << box.yc;
    return os.str();
}

static bool contains(bbox_t box, cv::Point2f pt)
{
    if ((box.x < pt.x) && (pt.x < box.x + box.w) && (box.y < pt.y) && (pt.y < box.y + box.h))
        return true;
    else
        return false;
}

static float area(bbox_t box)
{
    return static_cast<float>(box.w) * static_cast<float>(box.h);
}
// #endif // _COMMON_H
