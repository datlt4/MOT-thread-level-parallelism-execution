#ifndef HA_TINY_H
#define HA_TINY_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include "TrackerManager.h"
#include "HistogramFeature.h"
#include "common.h"
#include "util.h"

namespace M
{
    struct TrackData
    {
        KalmanTracker kalman;
        FeatureBundle feats;
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
    std::unique_ptr<FeatureMetric<M::TrackData>> feat_metric;
    std::vector<M::TrackData> data;
};

#endif // HA_TINY_H
