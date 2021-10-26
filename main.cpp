#include "main.h"

/**
 * ./mot AVG-TownCentre
 * ./mot djiphantom
 * ./mot drone
 * ./mot helofast
 * ./mot oxford-street
 * ./mot police-drone
 * ./mot Thermal
 * ./mot many-people 30
 */

int main(int argc, const char *argv[])
{
    // * Taken from answer to "How can I catch a ctrl-c event? (C++)"
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = M::sig_to_exception;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    std::string xx{argv[1]};
    int sleep = std::atoi(argv[2]);

    std::string inputPath = "/mnt/4B323B9107F693E2/TensorRT/samples/vizgard-drone/" + xx + ".mp4";
    std::string outputPath = "./" + xx + ".mp4";

    cv::VideoCapture cap(inputPath, cv::CAP_GSTREAMER);
    cv::VideoWriter writer(outputPath, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), cap.get(cv::CAP_PROP_FPS), cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    if (!cap.isOpened())
    {
        throw std::runtime_error("Cannot open cv::VideoCapture");
    }

    std::array<int64_t, 2> origDim{int64_t(cap.get(cv::CAP_PROP_FRAME_HEIGHT)), int64_t(cap.get(cv::CAP_PROP_FRAME_WIDTH))};

    Config *cfg = new Config();
    cfg->BATCH_SIZE = 1;
    cfg->INPUT_CHANNEL = 3;
    cfg->engine_file = "/mnt/4B323B9107F693E2/TensorRT/model-zoo/vizgard/yolov4-vizgard-512.engine";
    cfg->labels_file = "/mnt/4B323B9107F693E2/TensorRT/model-zoo/vizgard/yolov4-vizgard-512.names";
    cfg->IMAGE_WIDTH = 512;
    cfg->IMAGE_HEIGHT = 512;
    cfg->model = std::string("yolo");
    cfg->iou_with_distance = true;
    cfg->obj_threshold = 0.6;
    cfg->nms_threshold = 0.45;
    cfg->strides = std::vector<int>{8, 16, 32};
    cfg->num_anchors = std::vector<int>{3, 3, 3};
    cfg->anchors = std::vector<std::vector<int>>{{12, 16}, {19, 36}, {40, 28}, {36, 75}, {76, 55}, {72, 146}, {142, 110}, {192, 243}, {459, 401}};

    YOLOv4 *yolo = new YOLOv4(cfg);
    yolo->LoadEngine();
#ifdef USE_MOT
    HaTiny tracker{};
    TargetStorage repo{};
#endif // USE_MOT
    // TODO: Thread communication
    M::send_one_replaceable_object<M::pipeline_data> stream2detect;
#ifdef USE_MOT
    M::send_one_replaceable_object<M::pipeline_data> detect2track, track2show;
#else
    M::send_one_replaceable_object<M::pipeline_data> detect2show;
#endif // USE_MOT

    // TODO: Flags
    bool exitProgramFlag = false;

    // TODO: start Threads
    std::thread retrivedFrameThead(
        [&]()
        {
            while (!exitProgramFlag)
            {
                M::pipeline_data pData;
                if (!cap.read(pData.cap_frame))
                {
                    cap.release();
                    cap = cv::VideoCapture(inputPath, cv::CAP_GSTREAMER);
                    cap.read(pData.cap_frame);
                }
                stream2detect.send(pData);
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep));
            }
            M::pipeline_data pData{cv::Mat()};
            stream2detect.send(pData);
            std::cout << "[ LOG ][ EXIT ] retrivedFrameThead" << std::endl;
        });

    std::thread detectThead(
        [&]()
        {
            while (!exitProgramFlag)
            {
                M::pipeline_data pData;
                pData = stream2detect.receive();
                if (!pData.cap_frame.empty())
                {
                    pData.dets = yolo->EngineInference(pData.cap_frame);
                }
#ifdef USE_MOT
                detect2track.send(pData);
#else
                detect2show.send(pData);
#endif // USE_MOT
            }
            std::cout << "[ LOG ][ EXIT ] detectThead" << std::endl;
        });

#ifdef USE_MOT
    std::thread trackingThead(
        [&]()
        {
            while (!exitProgramFlag)
            {
                M::pipeline_data pData;
                pData = detect2track.receive();
                if (!pData.cap_frame.empty())
                {
                    pData.trks = tracker.update(pData.dets, pData.cap_frame);
                }
                track2show.send(pData);
            }
            std::cout << "[ LOG ][ EXIT ] trackingThead" << std::endl;
        });
#endif // USE_MOT

    // TODO: FPS & flags
    int fps = 1;
    std::atomic<int> fps_counter(0), current_fps(0);
    std::chrono::steady_clock::time_point fps_count_start;
    try
    {
        while (true)
        {
            M::pipeline_data pData;
#ifdef USE_MOT
            pData = track2show.receive();
            auto frameProcessed = static_cast<uint32_t>(cap.get(cv::CAP_PROP_POS_FRAMES)) - 1;
            repo.update(pData.trks, frameProcessed, pData.cap_frame);
#else
            pData = detect2show.receive();
#endif // USE_MOT
            for (auto &d : pData.dets)
            {
                draw_bbox(pData.cap_frame, d);
            }
#ifdef USE_MOT
            for (auto &t : pData.trks)
            {
                draw_bbox(pData.cap_frame, t, std::to_string(t.track_id), color_map(t.track_id));
                draw_trajectories(pData.cap_frame, repo.get().at(t.track_id).trajectories, color_map(t.track_id));
            }
#endif // USE_MOT

            // TODO: Improve FPS formula?
            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            float time_sec = std::chrono::duration<double>(now - fps_count_start).count();
            if (time_sec >= 1)
            {
                current_fps = fps_counter / time_sec;
                fps_count_start = now;
                fps_counter = 0;
            }
            else
            {
                ++fps_counter;
            }
            std::string info_msg = "FPS: " + std::to_string(current_fps);
            cv::putText(pData.cap_frame, info_msg, cv::Point2f(10, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar{50, 255, 0}, 2);

            writer.write(pData.cap_frame);
            cv::imshow("EMoi", pData.cap_frame);
            // Press ESC on keyboard to exit
            char c = (char)cv::waitKey(25);
            if (c == 27)
            {
                exitProgramFlag = true;
                break;
            }
        }
    }
    catch (M::InterruptException &e)
    {
        exitProgramFlag = true;
        std::cout << "Caught signal " << e.S << std::endl;
    }
    if (retrivedFrameThead.joinable())
        retrivedFrameThead.join();
    if (detectThead.joinable())
        detectThead.join();
#ifdef USE_MOT
    if (trackingThead.joinable())
        trackingThead.join();
#endif // USE_MOT
    writer.release();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
