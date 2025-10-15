
#include <opencv2/opencv.hpp>
#include <vector>

bool readVideoFrame(cv::VideoCapture& cap, cv::Mat& frame, int delay_ms = 30);
std::vector<cv::RotatedRect> processFrame(const cv::Mat& frame);
std::vector<std::pair<int, int>> findSimilarRectPairsOptimized(
    const std::vector<cv::RotatedRect>& rects,
    float angle_thresh = 5.0f,        
    float y_thresh = 0.1f,           
    float area_ratio_thresh = 2.0f,   
    float aspect_min = 3.50f,
    float aspect_max = 10.0f
);
void drawCombinedRectangles(cv::Mat& image,
                            const std::vector<cv::RotatedRect>& rects,
                            const std::vector<std::pair<int, int>>& pairs);


