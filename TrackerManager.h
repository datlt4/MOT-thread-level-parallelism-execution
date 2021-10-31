#ifndef TRACKER_H
#define TRACKER_H

#include <vector>
#include <torch/torch.h>
#include <tuple>

#include "KalmanTracker.h"
#include "common.h"
#include "KalmanTracker.h"
#include "util.h"

#define INVALID_DIST 1E6f

using DistanceMetricFunc = std::function<torch::Tensor(const std::vector<int> &trk_ids, const std::vector<int> &det_ids)>;

void associate_detections_to_trackers_idx(const DistanceMetricFunc &metric,
                                          std::vector<int> &unmatched_trks,
                                          std::vector<int> &unmatched_dets,
                                          std::vector<std::tuple<int, int>> &matched);

template <typename TrackDataTemplate>
class TrackerManager
{
public:
    explicit TrackerManager(std::vector<TrackDataTemplate> &data, const std::array<int64_t, 2> &dim) : data(data), img_box(0, 0, dim[1], dim[0]) {}

    void kalman_predict()
    {
        for (TrackDataTemplate &t : data)
        {
            t.kalman.predict();
        }
    }

    void remove_nan()
    {
        data.erase(std::remove_if(data.begin(), data.end(),
                                  [](TrackDataTemplate &t)
                                  {
                                      bbox_t bbox{t.kalman.bbox()};
                                      return std::isnan(bbox.x) || std::isnan(bbox.y) || std::isnan(bbox.w) || std::isnan(bbox.h);
                                  }),
                   data.end());
    }

    void remove_deleted()
    {
        data.erase(std::remove_if(data.begin(), data.end(),
                                  [this](const TrackDataTemplate &t)
                                  {
                                      return t.kalman.state() == TrackState::Deleted;
                                  }),
                   data.end());
    }

    std::vector<std::tuple<int, int>> update(const std::vector<bbox_t> &dets,
                                             const DistanceMetricFunc &iou_matching_metric,
                                             const DistanceMetricFunc &centroid_matching_metric,
                                             const DistanceMetricFunc &histogram_matching_metric)
    {
        // * M: unmatched_trks = [i0, i1, ...]
        std::vector<int> unmatched_trks;
        for (size_t i = 0; i < data.size(); ++i)
        {
            if (data[i].kalman.state() == TrackState::Confirmed)
            {
                unmatched_trks.emplace_back(i);
            }
        }

        for (size_t i = 0; i < data.size(); ++i)
        {
            if (data[i].kalman.state() == TrackState::Tentative)
            {
                unmatched_trks.emplace_back(i);
            }
        }

        // * M: unmatched_dets = [0, 1, ...]
        std::vector<int> unmatched_dets(dets.size());
        iota(unmatched_dets.begin(), unmatched_dets.end(), 0);

        // * Declare matched
        std::vector<std::tuple<int, int>> matched;

        // * IOU Matching
        associate_detections_to_trackers_idx(iou_matching_metric, unmatched_trks, unmatched_dets, matched);
        // auto centroid_cost = centroid_matching_metric(unmatched_trks, unmatched_dets);
        // std::cout << "[ CENTROID ][ COST ]: " << centroid_cost << std::endl;
        associate_detections_to_trackers_idx(histogram_matching_metric, unmatched_trks, unmatched_dets, matched);

        for (int i : unmatched_trks)
        {
            data[i].kalman.miss();
        }

        for (auto [x, y] : matched)
        {
            data[x].kalman.update(dets[y]);
        }

        // create and initialise new trackers for unmatched detections
        for (auto umd : unmatched_dets)
        {
            matched.emplace_back(data.size(), umd);
            TrackDataTemplate t{};
            t.kalman.init(dets[umd]);
            data.emplace_back(t);
        }
        return matched;
    }

    std::vector<bbox_t> visible_tracks()
    {
        std::vector<bbox_t> ret;
        for (auto &t : data)
        {
            bbox_t box{t.kalman.bbox()};
            cv::Rect2f rect(box.x, box.y, box.w, box.h);
            // if (t.kalman.state() == TrackState::Confirmed && contains(img_box, rect.tl()) && contains(img_box, rect.br()))
            if (t.kalman.state() == TrackState::Confirmed
            {
                ret.push_back(t.kalman.bbox());
            }
        }
        return ret;
    }

private:
    std::vector<TrackDataTemplate> &data;
    const bbox_t img_box;
};

#endif // TRACKER_H
