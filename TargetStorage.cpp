#include <fstream>
#include <iomanip>
#include <iostream>

#include "TargetStorage.h"

using namespace std;

TargetStorage::TargetStorage()
{
}

void TargetStorage::update(const vector<bbox_t> &trks, int frame, const cv::Mat &image)
{
    for (auto box : trks)
    {
        // save normalized boxes
        box = normalize_rect(box, image.cols, image.rows);

        auto &t = targets[box.track_id];
        t.trajectories.emplace(frame, box);

    }
}
