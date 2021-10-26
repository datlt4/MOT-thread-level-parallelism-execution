#include "Yolov4TensorRT.h"

YOLOv4::YOLOv4(Config *config)
{
    engine_file = config->engine_file;
    labels_file = config->labels_file;
    BATCH_SIZE = config->BATCH_SIZE;
    INPUT_CHANNEL = config->INPUT_CHANNEL;
    IMAGE_WIDTH = config->IMAGE_WIDTH;
    IMAGE_HEIGHT = config->IMAGE_HEIGHT;
    obj_threshold = config->obj_threshold;
    nms_threshold = config->nms_threshold;
    model_name = std::string(config->model);
    strides = config->strides;
    num_anchors = config->num_anchors;
    anchors = config->anchors;
    if (this->model_name == std::string("csp"))
    {
        this->iou_with_distance = true;
    }
    else
    {
        this->iou_with_distance = false;
    }

    detect_labels = readCOCOLabel(labels_file);
    CATEGORY = detect_labels.size();
    int index = 0;
    for (const int &stride : strides)
    {
        grids.push_back({num_anchors[index], int(IMAGE_HEIGHT / stride), int(IMAGE_WIDTH / stride)});
        index++;
    }
    refer_rows = 0;
    refer_cols = 6;
    for (const std::vector<int> &grid : grids)
    {
        refer_rows += std::accumulate(grid.begin(), grid.end(), 1, std::multiplies<int>());
    }
    GenerateReferMatrix();
    class_colors.resize(CATEGORY);
    srand((int)time(nullptr));
    for (cv::Scalar &class_color : class_colors)
        class_color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
}

YOLOv4::~YOLOv4()
{
    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    // destroy the engine
    context->destroy();
    engine->destroy();
}

void YOLOv4::LoadEngine()
{
    // create and load engine
    std::fstream existEngine;
    existEngine.open(engine_file, std::ios::in);
    if (existEngine)
    {
        readTrtFile(engine_file, engine, gLogger);
        assert(engine != nullptr);
    }
    else
    {
        assert(engine != nullptr);
    }

    //get context
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);

    //get buffers
    assert(engine->getNbBindings() == 2);
    int nbBindings = engine->getNbBindings();
    bufferSize.resize(nbBindings);

    for (int i = 0; i < nbBindings; ++i)
    {
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
        bufferSize[i] = totalSize;
        cudaMalloc(&buffers[i], totalSize);
    }

    //get stream
    cudaStreamCreate(&stream);
    outSize = bufferSize[1] / sizeof(float) / BATCH_SIZE;
}

std::vector<bbox_t> YOLOv4::EngineInference(cv::Mat &image)
{
    std::vector<bbox_t> boxes;
    std::vector<float> curInput = prepareImage(image);
    if (!curInput.data())
    {
        std::cout << "prepare images ERROR!" << std::endl;
        return boxes;
    }

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    cudaMemcpyAsync(buffers[0], curInput.data(), bufferSize[0], cudaMemcpyHostToDevice, stream);
    context->execute(BATCH_SIZE, buffers);
    auto *out = new float[outSize * BATCH_SIZE];
    // float out[outSize * BATCH_SIZE];
    cudaMemcpyAsync(out, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    boxes = postProcess(image, out, outSize);
    // cv::cvtColor(org_img, org_cvimg, cv::COLOR_BGR2RGB);
    delete[] out;
    std::sort(boxes.begin(), boxes.end(),
              [](bbox_t A, bbox_t B)
              {
                  return ((static_cast<float>(A.w) * static_cast<float>(A.h)) > (static_cast<float>(B.h) * static_cast<float>(B.w)));
              });
    return boxes;
}

std::vector<float> YOLOv4::prepareImage(cv::Mat &img)
{
    std::vector<float> result(long(BATCH_SIZE * IMAGE_WIDTH * IMAGE_HEIGHT * INPUT_CHANNEL));
    float *data = result.data();
    int index = 0;
    if (!img.data)
        return result;

    cv::Mat flt_img;
    if (this->model_name == std::string("csp")) // letter_box = True
    {
        float ratio = std::min(float(IMAGE_WIDTH) / float(img.cols), float(IMAGE_HEIGHT) / float(img.rows));
        flt_img = cv::Mat(cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), CV_32FC3, 0.5);
        cv::Mat rsz_img;
        cv::resize(img, rsz_img, cv::Size(), ratio, ratio);
        rsz_img.convertTo(rsz_img, CV_32FC3, 1.0 / 255);
        x_offset = (IMAGE_WIDTH - rsz_img.cols) / 2;
        y_offset = (IMAGE_HEIGHT - rsz_img.rows) / 2;
        rsz_img.copyTo(flt_img(cv::Rect(x_offset, y_offset, rsz_img.cols, rsz_img.rows)));
    }
    else
    {
        cv::resize(img, flt_img, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));
        flt_img.convertTo(flt_img, CV_32FC3, 1.0 / 255);
    }

    //HWC TO CHW
    int channelLength = IMAGE_WIDTH * IMAGE_HEIGHT;
    std::vector<cv::Mat> split_img = {
        cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, data + channelLength * 2),
        cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, data + channelLength),
        cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1, data)};
    cv::split(flt_img, split_img);
    return result;
}

std::vector<bbox_t> YOLOv4::postProcess(cv::Mat &src_img, float *output, int &outSize)
{
    std::vector<bbox_t> result;
    float *out = output;
    cv::Mat result_matrix = cv::Mat(refer_rows, CATEGORY + 5, CV_32FC1, out);
    for (int row_num = 0; row_num < refer_rows; row_num++)
    {
        bbox_t box;
        auto *row = result_matrix.ptr<float>(row_num);
        auto max_pos = std::max_element(row + 5, row + CATEGORY + 5);
        box.prob = sigmoid(row[4]) * sigmoid(row[max_pos - row]);
#ifdef INFERENCE_ALPHAPOSE_TORCH
#ifdef INFERENCE_TABULAR_TORCH
        memcpy((void *)box.feature, (void *)(row + 5), CATEGORY * sizeof(float));
#endif // INFERENCE_TABULAR_TORCH
#endif // INFERENCE_ALPHAPOSE_TORCH
        if (box.prob < obj_threshold)
            continue;
        box.obj_id = static_cast<unsigned int>(max_pos - row - 5);
        auto *anchor = refer_matrix.ptr<float>(row_num);
        float xc, yc, x, y, w, h;
        if (this->model_name == std::string("csp"))
        {
            float ratio = std::max(float(src_img.cols) / float(IMAGE_WIDTH), float(src_img.rows) / float(IMAGE_HEIGHT));
            xc = (float)((row[0] * 2 - 0.5 + anchor[0]) / anchor[1] * (float)IMAGE_WIDTH - (float)x_offset) * ratio;
            yc = (float)((row[1] * 2 - 0.5 + anchor[2]) / anchor[3] * (float)IMAGE_HEIGHT - (float)y_offset) * ratio;
            w = (float)pow(row[2] * 2, 2) * anchor[4] * ratio; // ratio_w;
            h = (float)pow(row[3] * 2, 2) * anchor[5] * ratio; // ratio_h;
        }
        else
        {
            xc = (sigmoid(row[0]) + anchor[0]) / anchor[1] * (float)src_img.cols;
            yc = (sigmoid(row[1]) + anchor[2]) / anchor[3] * (float)src_img.rows;
            w = exp(row[2]) * anchor[4] / (float)IMAGE_WIDTH * (float)src_img.cols;
            h = exp(row[3]) * anchor[5] / (float)IMAGE_HEIGHT * (float)src_img.rows;
        }
        box.x = xc - w / 2;
        box.y = yc - h / 2;
        box.x = (box.x < 0) ? 0.0 : box.x;
        box.y = (box.y < 0) ? 0.0 : box.y;
        box.w = (w < 0) ? 0.0 : w;
        box.w = ((box.x + box.w) > src_img.cols) ? static_cast<float>(src_img.cols) - box.x : box.w;
        box.h = (h < 0) ? 0.0 : h;
        box.h = ((box.y + box.h) > src_img.rows) ? static_cast<float>(src_img.rows) - box.y : box.h;
        box.xc = box.x + box.w / 2;
        box.yc = box.y + box.h / 2;
        result.push_back(box);
    }
    NmsDetect(result);
    return result;
}

void YOLOv4::GenerateReferMatrix()
{
    refer_matrix = cv::Mat(refer_rows, refer_cols, CV_32FC1);
    int position = 0;

    for (int n = 0; n < (int)grids.size(); n++)
    {
        for (int c = 0; c < grids[n][0]; c++)
        {
            std::vector<int> anchor = anchors[n * grids[n][0] + c];
            for (int h = 0; h < grids[n][1]; h++)
                for (int w = 0; w < grids[n][2]; w++)
                {
                    float *row = refer_matrix.ptr<float>(position);
                    row[0] = w;
                    row[1] = grids[n][2];
                    row[2] = h;
                    row[3] = grids[n][1];
                    row[4] = anchor[0];
                    row[5] = anchor[1];
                    position++;
                }
        }
    }
}

void YOLOv4::NmsDetect(std::vector<bbox_t> &detections)
{
    sort(detections.begin(), detections.end(), [=](const bbox_t &left, const bbox_t &right)
         { return left.prob > right.prob; });

    for (int i = 0; i < (int)detections.size(); i++)
        for (int j = i + 1; j < (int)detections.size(); j++)
        {
            if (detections[i].obj_id == detections[j].obj_id)
            {
                float iou = IOUCalculate(detections[i], detections[j]);
                if (iou > nms_threshold)
                    detections[j].prob = 0;
            }
        }

    detections.erase(std::remove_if(detections.begin(), detections.end(), [](const bbox_t &det)
                                    { return det.prob == 0; }),
                     detections.end());
}

float YOLOv4::IOUCalculate(const bbox_t &det_a, const bbox_t &det_b)
{
    cv::Point2f center_a(det_a.xc, det_a.yc);
    cv::Point2f center_b(det_b.xc, det_b.yc);
    cv::Point2f left_up(std::min(det_a.x, det_b.x), std::min(det_a.y, det_b.y));
    cv::Point2f right_down(std::max(det_a.x + det_a.w, det_b.x + det_b.w), std::max(det_a.y + det_a.h, det_b.y + det_b.h));
    float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
    float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
    float inter_l = det_a.x > det_b.x ? det_a.x : det_b.x;
    float inter_t = det_a.y > det_b.y ? det_a.y : det_b.y;
    float inter_r = det_a.x + det_a.w < det_b.x + det_b.w ? det_a.x + det_a.w : det_b.x + det_b.w;
    float inter_b = det_a.y + det_a.h < det_b.y + det_b.h ? det_a.y + det_a.h : det_b.y + det_b.h;
    if (inter_b < inter_t || inter_r < inter_l)
        return 0;
    float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
    float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
    if (union_area == 0)
        return 0;
    else if (this->iou_with_distance)
        return inter_area / union_area - distance_d / distance_c;
    else
        return inter_area / union_area;
}

float YOLOv4::sigmoid(float in)
{
    if (this->model_name == std::string("csp"))
        return in;
    else
        return 1.f / (1.f + exp(-in));
}

std::map<int, std::string> readCOCOLabel(const std::string &fileName)
{
    std::map<int, std::string> coco_label;
    std::ifstream file(fileName);
    if (!file.is_open())
    {
#ifdef DEBUG
        std::cout << "[DEBUG] "
                  << "read file error: " << fileName << std::endl;
#endif // DEBUG
    }
    std::string strLine;
    int index = 0;
    while (getline(file, strLine))
    {
        coco_label.insert({index, strLine});
        index++;
    }
    file.close();
    return coco_label;
}

bool readTrtFile(const std::string &engineFile, nvinfer1::ICudaEngine *&engine, Logger &gLogger)
{
    std::string cached_engine;
    std::fstream file;
    nvinfer1::IRuntime *trtRuntime;
    file.open(engineFile, std::ios::binary | std::ios::in);

    if (!file.is_open())
    {
        cached_engine = "";
    }

    while (file.peek() != EOF)
    {
        std::stringstream buffer;
        buffer << file.rdbuf();
        cached_engine.append(buffer.str());
    }
    file.close();

    trtRuntime = nvinfer1::createInferRuntime(gLogger.getTRTLogger());
    engine = trtRuntime->deserializeCudaEngine(cached_engine.data(), cached_engine.size(), nullptr);

    return true;
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
        return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int64_t volume(const nvinfer1::Dims &d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}
