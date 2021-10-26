#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "HaTiny.h"
#include "Yolov4TensorRT.h"
#include "TargetStorage.h"
#include "util.h"
#include "common.h"
#include <thread>
#include <atomic>
#include <exception>
#include <signal.h>
#include <stdlib.h>

namespace M
{
    class InterruptException : public std::exception
    {
    public:
        InterruptException(int s) : S(s) {}
        int S;
    };

    void sig_to_exception(int s)
    {
        throw InterruptException(s);
    }

    // TODO: Minimize, optimize this
    struct pipeline_data
    {
        cv::Mat cap_frame;
        std::vector<bbox_t> dets;
        std::vector<bbox_t> trks;
        pipeline_data(){};
        pipeline_data(cv::Mat frame) : cap_frame(frame){};
    };

    template <typename T>
    class send_one_replaceable_object
    {
        std::atomic<T *> a_ptr = {nullptr};

    public:
        void send(T const &_obj)
        {
            T *new_ptr = new T;
            *new_ptr = _obj;
            // TODO: The `unique_ptr` prevents a scary memory leak, why?
            std::unique_ptr<T> old_ptr(a_ptr.exchange(new_ptr));
        }

        T receive()
        {
            std::unique_ptr<T> ptr;
            do
            {
                while (!a_ptr)
                    std::this_thread::sleep_for(std::chrono::milliseconds(3));
                ptr.reset(a_ptr.exchange(nullptr));
            } while (!ptr);
            return *ptr;
        }
    };

}

#endif // MAIN_H
