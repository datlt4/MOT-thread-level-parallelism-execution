#ifndef KALMAN_H
#define KALMAN_H

#include "opencv2/video/tracking.hpp"
#include "common.h"
#include <torch/torch.h>

#define MAX_AGE 10
#define N_INIT 3

enum class TrackState
{
    Tentative,
    Confirmed,
    Deleted
};

// This class represents the internel state of individual tracked objects observed as bounding box.
class KalmanTracker
{
public:
    KalmanTracker();

    explicit KalmanTracker(bbox_t initRect) : KalmanTracker() { init(initRect); }

    void init(bbox_t initRect);

    void predict();

    void update(bbox_t stateMat);

    void miss();

    cv::Rect2f rect();

    bbox_t bbox();

    TrackState state() const { return _state; }

    int id() const { return box.track_id; }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> meanVarCovStateKF();

private:
    static const int max_age = MAX_AGE;
    static const int n_init = N_INIT;

    static int count;

    TrackState _state = TrackState::Tentative;

    bbox_t box;

    int time_since_update = 0;
    int hits = 0;

    cv::KalmanFilter kf;
    cv::Mat measurement;

    std::vector<torch::Tensor> stateKF;
};

#endif // KALMAN_H
