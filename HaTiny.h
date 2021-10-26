#ifndef HA_TINY_H
#define HA_TINY_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include "TrackerManager.h"
#include "common.h"
#include "util.h"

namespace M
{
    struct TrackData
    {
        KalmanTracker kalman;
    };
}

template <typename T>
class TrackerManager;

class HaTiny
{
public:
    explicit HaTiny();
    ~HaTiny();
    std::vector<bbox_t> update(const std::vector<bbox_t> &detections, cv::Mat ori_img);

private:
    std::unique_ptr<TrackerManager<M::TrackData>> manager;
    std::vector<M::TrackData> data;
};

#endif // HA_TINY_H
