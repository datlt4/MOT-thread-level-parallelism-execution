#ifndef TARGET_H
#define TARGET_H

#include <map>
#include <array>
#include <utility>
#include <opencv2/opencv.hpp>

#include "util.h"
#include "common.h"

class TargetStorage
{
public:
    explicit TargetStorage();

    virtual ~TargetStorage() {}

    void update(const std::vector<bbox_t> &trks,
                int frame, const cv::Mat &image);

    struct Target
    {
        std::map<int, bbox_t> trajectories;
    };

    const std::map<int, Target> &get() const { return targets; }

private:
    std::map<int, Target> targets;
};

#endif //TARGET_H
