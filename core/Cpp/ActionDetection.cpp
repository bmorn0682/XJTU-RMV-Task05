
#include <algorithm>
#include <cstdlib>
#include "ActionDetection.hpp"//同名头文件

bool readVideoFrame(cv::VideoCapture& cap, cv::Mat& frame, int delay_ms)
{
    cap >> frame;
    if (frame.empty() || cv::waitKey(delay_ms) >= 0)
        return false;
    return true;
}


std::vector<cv::RotatedRect> processFrame(const cv::Mat& frame)
{
    std::vector<cv::RotatedRect> rects;

    cv::Mat gray, dst;
    cv::Mat src = frame.clone();
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    //cv::medianBlur(gray, gray, 11);
    cv::threshold(gray, dst, 150, 255, cv::THRESH_BINARY);
    //cv::adaptiveThreshold(gray, dst, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11,5);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(dst, dst, cv::MORPH_OPEN, kernel);
    cv::dilate(dst, dst, kernel);

    // cv::namedWindow("frame", cv::WINDOW_NORMAL);
    // cv::imshow("frame",dst);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dst, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours)
    {
        if (cv::contourArea(contour) <= 100)
            continue;

        cv::RotatedRect rect = cv::minAreaRect(contour);
        double area = cv::contourArea(contour);
        if (area / rect.size.area() > 0.6)
            rects.push_back(rect);
    }

    return rects;
}


std::vector<std::pair<int, int>> findSimilarRectPairsOptimized(
    const std::vector<cv::RotatedRect>& rects,
    float angle_thresh,       
    float y_thresh ,          
    float area_ratio_thresh,   
    float aspect_min,
    float aspect_max
) 
{
    struct RectFeature {
        int index;
        float angle;
        float x;
        float y;
        float area;
        cv::RotatedRect rect;
    };

    std::vector<std::pair<int, int>> matched_pairs;
    std::vector<RectFeature> feats;

    feats.reserve(rects.size());

    for (int i = 0; i < (int)rects.size(); ++i) {
        float w = rects[i].size.width;
        float h = rects[i].size.height;
        float area = w * h;
        float aspect = std::max(w, h) / std::min(w, h);
        float angle = rects[i].angle;
        if (w < h) angle += 90.f;

        //std::cout << "aspect= " << aspect <<std::endl; 

        if (aspect < aspect_min || aspect > aspect_max)
            continue;
        
        feats.push_back({i, angle, rects[i].center.x,rects[i].center.y, area, rects[i]});
    }
    //std::cout << "feats size = " << feats.size() << std::endl;  

    std::sort(feats.begin(), feats.end(),
              [](const RectFeature &a, const RectFeature &b) {
                  return a.y < b.y;
              });

    for (size_t i = 0; i < feats.size(); ++i) {
        const auto &r1 = feats[i];
        //std::cout<< "feasts[" << i << "]: angle= " << feats[i].angle << "center.x: " << rects[i].center.x <<" center.y= " << rects[i].center.y << " area= "<< feats[i].area << std::endl;   
        for (size_t j = i + 1; j < feats.size(); ++j) {
            const auto &r2 = feats[j];
            if (abs(r2.x - r1.x )> 5 * std::max(r1.rect.size.height, r1.rect.size.width) ) 
                break;
            if ((r2.y - r1.y)/std::max(r1.y, r2.y) > y_thresh * r1.rect.size.height)
                break; 

            float d_angle = std::fabs(r1.angle - r2.angle);
            float area_ratio = std::max(r1.area, r2.area) / std::min(r1.area, r2.area);

            if (d_angle <= angle_thresh &&
                area_ratio <= area_ratio_thresh)
            {
                matched_pairs.emplace_back(r1.index, r2.index);
            }
        }
    }

    return matched_pairs;
}


void drawCombinedRectangles(cv::Mat& image,
                            const std::vector<cv::RotatedRect>& rects,
                            const std::vector<std::pair<int, int>>& pairs)
{
    if (image.empty() || rects.empty() || pairs.empty()) return;

    float factor = 0.7;
    auto randomColor = []() -> cv::Scalar {
        return cv::Scalar(0,0,255);
    };
    for (const auto& p : pairs)
    {
        size_t i = p.first;
        size_t j = p.second;
        if (i >= rects.size() || j >= rects.size()) continue;

        cv::Scalar color = randomColor();

        cv::Point2f ptsA[4], ptsB[4];
        rects[i].points(ptsA);
        rects[j].points(ptsB);


        auto sortCorners = [](cv::Point2f pts[4]) -> std::array<cv::Point2f, 4> {
            std::vector<cv::Point2f> v(pts, pts + 4);
            std::sort(v.begin(), v.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
                return (a.y < b.y) || (a.y == b.y && a.x < b.x);
            });
            cv::Point2f tl = v[0].x < v[1].x ? v[0] : v[1];
            cv::Point2f tr = v[0].x < v[1].x ? v[1] : v[0];
            cv::Point2f bl = v[2].x < v[3].x ? v[2] : v[3];
            cv::Point2f br = v[2].x < v[3].x ? v[3] : v[2];
            return {tl, tr, br, bl};
        };

        auto cornersA = sortCorners(ptsA);
        auto cornersB = sortCorners(ptsB);


        cv::Point2f heightVec = cornersA[3] - cornersA[0];
        cv::Point2f expandVec = heightVec * static_cast<float>(factor);  

        cornersA[0] -= expandVec; 
        cornersA[3] += expandVec;  


        cornersB[1] -= expandVec;  
        cornersB[2] += expandVec; 
        
        std::vector<cv::Point2f> combo = {
            cornersA[0], 
            cornersB[1], 
            cornersB[2], 
            cornersA[3]  
        };

    
        for (int k = 0; k < 4; ++k)
            cv::line(image, combo[k], combo[(k + 1) % 4], color, 2, cv::LINE_AA);

        //cv::Point mid = (rects[i].center + rects[j].center) * 0.5;
        //cv::putText(image, std::to_string(i) + "-" + std::to_string(j),
        //            mid, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
}
