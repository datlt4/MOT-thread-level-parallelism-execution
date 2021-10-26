#include "HaTiny.h"

HaTiny::HaTiny() : manager(make_unique<TrackerManager<M::TrackData>>(data))
{
}

HaTiny::~HaTiny() = default;

std::vector<bbox_t> HaTiny::update(const std::vector<bbox_t> &detections, cv::Mat ori_img)
{
    manager->kalman_predict();
    manager->remove_nan();

    std::vector<int> targets;
    std::vector<std::tuple<int, int>> matched = manager->update(
        detections,
        [this, &detections](const std::vector<int> &trk_ids, const std::vector<int> &det_ids)
        {
            std::vector<bbox_t> trks;
            for (int t : trk_ids)
            {
                trks.push_back(data[t].kalman.bbox());
            }
            std::vector<bbox_t> dets;
            for (int d : det_ids)
            {
                dets.push_back(detections[d]);
            }
            auto iou_mat = M::iou_dist(dets, trks);
            iou_mat.masked_fill_(iou_mat > 0.9f, INVALID_DIST);
            return iou_mat;
        });

    // for (auto [x, y] : matched)
    // {
    //     targets.emplace_back(x);
    //     cv::Rect2f roi(detections[y].x, detections[y].y, detections[y].w, detections[y].h);
    //     boxes.emplace_back(ori_img(roi));
    // }

    manager->remove_deleted();
    return manager->visible_tracks();
}
