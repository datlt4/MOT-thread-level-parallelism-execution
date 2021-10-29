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
        [this, &detections, &ori_img](const std::vector<int> &trk_ids, const std::vector<int> &det_ids)
        {
            std::vector<bbox_t> trks;
            for (int t : trk_ids)
            {
                trks.push_back(data[t].kalman.bbox());
            }
            std::vector<bbox_t> dets;
            std::vector<cv::Mat> boxes;
            for (int d : det_ids)
            {
                dets.push_back(detections[d]);
                cv::Rect2f roi(detections[d].x, detections[d].y, detections[d].w, detections[d].h);
                boxes.push_back(ori_img(roi));
            }
            auto iou_mat = M::iou_dist(dets, trks);
            std::cout << M::extract(boxes) << std::endl;
            // auto feat_mat = feat_metric->distance(M::extract(boxes), trk_ids);
            iou_mat.masked_fill_(iou_mat > 0.9f, INVALID_DIST);
            return iou_mat;
        },
        [this, &detections](const std::vector<int> &trk_ids, const std::vector<int> &det_ids)
        {
            std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> trks;
            for (int t : trk_ids)
            {
                trks.push_back(data[t].kalman.meanVarCovStateKF());
            }
            std::vector<torch::Tensor> dets;
            for (int d : det_ids)
            {
                torch::Tensor _d = torch::ones({4});
                dets.emplace_back(torch::ones({4}));
                dets[dets.size() - 1][0] = detections[d].xc;
                dets[dets.size() - 1][1] = detections[d].yc;
                dets[dets.size() - 1][2] = detections[d].w * detections[d].h;
                dets[dets.size() - 1][3] = detections[d].w;
                // dets.push_back(detections[d]);
            }
            auto iou_mat = M::mahalanobis_dist(dets, trks);
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
