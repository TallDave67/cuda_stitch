#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <Eigen/Dense>

#define PI 3.14285714286

using namespace Eigen;
using namespace std;
using namespace std::chrono;

struct imageData {
    std::string imageName = "";
    double latitude = 0;
    double longitude = 0;
    double altitudeFeet = 0;
    double altitudeMeter = 0;
    double roll = 0;
    double pitch = 0;
    double yaw = 0;
};

struct imageSize2d {
    float width = 0.;
    float height = 0.;
    float scale = 0.;

    float getNewWidth() {
        if (scale == 0)
        {
            return 0.;
        }
        else
        {
            return width / scale;
        }
    }

    float getNewHeight() {
        if (scale == 0)
        {
            return 0.;
        }
        else
        {
            return height / scale;
        }
    }
};

inline ostream& operator<<(ostream& oss, const imageSize2d& other)
{
    oss << "width: " << other.width << ", height: " << other.height << ", scale: " << other.scale;
    return oss;
}

struct imageRange2d {
    int max_x = 0;
    int min_x = 0;
    int max_y = 0;
    int min_y = 0;

    int getWidth() {
        int w = max_x - min_x;
        //std::cout << "width = " << w << std::endl;
        return w;
    }

    int getHeight() {
        int h = max_y - min_y;
        //std::cout << "height = " << h << std::endl;
        return h;
    }
};

inline ostream& operator<<(ostream& oss, const imageRange2d& other)
{
    oss << "min_x: " << other.min_x << ", max_x: " << other.max_x << ", min_y: " << other.min_y << ", max_y: " << other.max_y;
    return oss;
}

void printMat(cv::Mat& mat) {
    int rows = mat.rows;
    int cols = mat.cols;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << mat.at<double>(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void getTransformationToOriginPlane (imageData& pose, cv::Mat& transformation) {
    
    // get our 3 rotation angles
    double h = (pose.yaw * PI) / 180;
    double p = (pose.pitch * PI) / 180;
    double b = (pose.roll * PI) / 180;

    // compute the rotation matrix per axis
    Matrix3d Rz;
    Rz << cos (h), -sin (h), 0,
          sin (h), cos (h), 0,
          0, 0, 1;
    Matrix3d Ry;
    Ry << cos (p), 0, sin (p),
          0, 1, 0,
          -sin (p), 0, cos (p);
    Matrix3d Rx;
    Rx << 1, 0, 0,
          0, cos (b), -sin (b),
          0, sin (b), cos (b);

    // rotate back to origin plane in roll-pitch-yaw order
    Matrix3d R = Rz * (Ry * Rx); 

    // do not adjust any rotations made around the z-axis
    R(0, 2) = 0;
    R(1, 2) = 0;
    R(2, 2) = 1;

    // create the final transformation
    Matrix3d Rtrans = R.transpose ();
    Matrix3d InvR = Rtrans.inverse ();
    transformation = (cv::Mat_<double>(3,3) << InvR(0,0), InvR(0,1), InvR(0,2),
                                               InvR(1,0), InvR(1,1), InvR(1,2),
                                               InvR(2,0), InvR(2,1), InvR(2,2));
}

void getBoundedRangesFromCorners(std::vector <std::vector<cv::Point2f>>& corners_collection, imageRange2d& range2d) {
    // match the ranges to just include all the corners of all the rectangles in the collection
    // but decrease the ranges if any go outside min/max bounds
    float max_x = 1e9, min_x = -1e9;
    float max_y = 1e9, min_y = -1e9;
    float max_x_warp = 0., min_x_warp = 0.;
    float max_y_warp = 0., min_y_warp = 0.;

    // do the loop
    bool seeded_min = false;
    bool seeded_max = false;
    for (auto & rect : corners_collection) {
        // min is seeded?
        if (!seeded_min)
        {
            min_x_warp = rect[0].x;
            min_y_warp = rect[0].y;
            seeded_min = true;
        }

        // origin corner tells us the min
        if (rect[0].x < min_x || rect[0].x > max_x)
        {
            min_x_warp = min_x;
        }
        if (rect[0].y < min_y || rect[0].y > max_y)
        {
            min_y_warp = min_y;
        }


        // max is seeded?
        if (!seeded_max)
        {
            max_x_warp = rect[2].x;
            max_y_warp = rect[2].y;
            seeded_max = true;
        }

        // corner diagonal from the origin corner tells us the max
        if (rect[2].x < min_x || rect[2].x > max_x)
        {
            max_x_warp = max_x;
        }
        if (rect[2].y < min_y || rect[2].y > max_y)
        {
            max_y_warp = max_y;
        }
    }

    // round floats to appropriate integer value
    range2d.max_x = static_cast<int>(max_x_warp - 0.5);
    range2d.min_x = static_cast<int>(min_x_warp + 0.5);
    range2d.max_y = static_cast<int>(max_y_warp - 0.5);
    range2d.min_y = static_cast<int>(min_y_warp + 0.5);
    //std::cout << "getBoundedRangesFromCorners: range2d = " << range2d << std::endl;

}

void getScaledPaddedTransformation(cv::Mat& transformation, imageSize2d& size2d, cv::Mat& padded_transformation, imageRange2d& range2d) {
    // translate corners according to the scale factor
    std::vector<cv::Point2f> corners { 
            cv::Point2f(0.,0.), 
            cv::Point2f(0.,size2d.height/ size2d.scale),
            cv::Point2f(size2d.width/ size2d.scale,size2d.height/ size2d.scale),
            cv::Point2f(size2d.width/ size2d.scale,0.)
    };

    // apply the transformation to get the warped corners
    std::vector<cv::Point2f> warpedCorners;
    cv::perspectiveTransform(corners, warpedCorners, transformation);
    //std::cout << "warpedCorners = " << warpedCorners << std::endl;
    //std::cout << "transformation = " << transformation << std::endl;

    //get x & y ranges for the warped corners that do not go outside of min/max bounds
    std::vector <std::vector<cv::Point2f>> corners_collection;
    corners_collection.push_back(warpedCorners);
    getBoundedRangesFromCorners(corners_collection, range2d);
}

cv::Mat warpPerspectiveWithPadding (const cv::Mat& image, cv::Mat& transformation) {
    // reduce the individual image size so we have enough room to create the final stitched tapestry 
    imageSize2d size2d{ static_cast<float>(image.rows), static_cast<float>(image.rows), 2.0 };
    //std::cout << "size2d = " << size2d << std::endl;
    cv::Mat small_img;
    cv::resize (image, small_img, cv::Size (size2d.getNewWidth(), size2d.getNewHeight()));

    // get a padded transformation scaled to to our reduced image size
    cv::Mat padded_transformation;
    imageRange2d range2d;
    getScaledPaddedTransformation(transformation, size2d, padded_transformation, range2d);
    cv::Mat translation = (cv::Mat_<double>(3,3) << 1, 0, range2d.min_x, 0, 1, range2d.min_y, 0, 0, 1);
    cv::Mat fullTransformation = translation * transformation;

    // perform the image perspective warp
    cv::cuda::GpuMat gpu_img (small_img);
    cv::cuda::GpuMat gpu_img_wp;
    cv::cuda::GpuMat gpu_ft (fullTransformation);
    //std::cout << "fullTransformation = " << fullTransformation << std::endl;
    cv::Size gpu_size{ range2d.getWidth(), range2d.getHeight() };
    //std::cout << "gpu_size = " << gpu_size << std::endl;
    cv::cuda::warpPerspective(gpu_img, gpu_img_wp, fullTransformation, gpu_size);

    // download the final result matrix
    cv::Mat result (gpu_img_wp.size(), gpu_img_wp.type());
    gpu_img_wp.download (result);
    return result;
}

cv::Mat combinePair (cv::Mat& img1, cv::Mat& img2) {
    std::cout << "combinePair: start" << std::endl;

    cv::cuda::GpuMat img1_gpu (img1), img2_gpu (img2);
    cv::cuda::GpuMat img1_gray_gpu, img2_gray_gpu;

    cv::cuda::cvtColor (img1_gpu, img1_gray_gpu, cv::COLOR_BGR2GRAY);
    cv::cuda::cvtColor (img2_gpu, img2_gray_gpu, cv::COLOR_BGR2GRAY);

    cv::cuda::GpuMat mask1;
    cv::cuda::GpuMat mask2;

    cv::cuda::threshold (img1_gray_gpu, mask1, 1, 255, cv::THRESH_BINARY);
    cv::cuda::threshold (img2_gray_gpu, mask2, 1, 255, cv::THRESH_BINARY);

    std::cout << "detect keypoints" << std::endl;
    cv::cuda::SURF_CUDA detector;

    cv::cuda::GpuMat keypoints1_gpu, descriptors1_gpu;
    detector (img1_gray_gpu, mask1, keypoints1_gpu, descriptors1_gpu);
    
    std::vector<cv::KeyPoint> keypoints1;
    detector.downloadKeypoints (keypoints1_gpu, keypoints1);

    cv::cuda::GpuMat keypoints2_gpu, descriptors2_gpu;
    detector(img2_gray_gpu, mask2, keypoints2_gpu, descriptors2_gpu);
    
    std::vector<cv::KeyPoint> keypoints2;
    detector.downloadKeypoints (keypoints2_gpu, keypoints2);

    std::cout << "do matching" << std::endl;
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher ();

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch (descriptors2_gpu, descriptors1_gpu, knn_matches, 2);

    std::vector<cv::DMatch> matches;
    std::vector<std::vector<cv::DMatch>>::const_iterator it;
    for (it = knn_matches.begin(); it != knn_matches.end(); ++it) {
        if(it->size() > 1 && (*it)[0].distance/(*it)[1].distance < 0.55) {
            matches.push_back((*it)[0]);
        }
    }

    std::vector<cv::Point2f> src_pts;
    std::vector<cv::Point2f> dst_pts;
    for (auto m : matches) {
        src_pts.push_back (keypoints2[m.queryIdx].pt);
        dst_pts.push_back (keypoints1[m.trainIdx].pt);
    }

    std::cout << "estimate rigid transform" << std::endl;
    cv::Mat A = cv::estimateRigidTransform(src_pts, dst_pts, false);
    float height1 = static_cast<float>(img1.rows), width1 = static_cast<float>(img1.cols);
    float height2 = static_cast<float>(img2.rows), width2 = static_cast<float>(img2.cols);

    std::cout << "compute corners" << std::endl;
    std::vector<std::vector<float>> corners1 {{0,0},{0,height1},{width1,height1},{width1,0}};
    std::vector<std::vector<float>> corners2 {{0,0},{0,height2},{width2,height2},{width2,0}};

    std::vector<std::vector<float>> warpedCorners2 (4, std::vector<float>(2));
    std::vector<std::vector<float>> allCorners = corners1;

    for (int i = 0; i < 4; i++) {
        float cornerX = corners2[i][0];
        float cornerY = corners2[i][1];
        warpedCorners2[i][0] = A.at<double> (0,0) * cornerX +
                            A.at<double> (0,1) * cornerY + A.at<double> (0,2);
        warpedCorners2[i][1] = A.at<double> (1,0) * cornerX +
                            A.at<double> (1,1) * cornerY + A.at<double> (1,2);
        allCorners.push_back (warpedCorners2[i]);
    }

    float max_x = 1e9, min_x = -1e9;
    float max_y = 1e9, min_y = -1e9;
    for (int i = 0; i < 7; i++) {
        max_x = (max_x > allCorners[i][0])? allCorners[i][0] : max_x;
        min_x = (min_x < allCorners[i][0])? allCorners[i][0] : min_x;
        max_y = (max_y > allCorners[i][1])? allCorners[i][1] : max_y;
        min_y = (min_y < allCorners[i][1])? allCorners[i][1] : min_y;
    }
    int max_x_ = (max_x - 0.5);
    int min_x_ = (min_x + 0.5);
    int max_y_ = (max_y - 0.5);
    int min_y_ = (min_y + 0.5);

    cv::Mat translation = (cv::Mat_<double>(3,3) << 1, 0, -max_x_, 0, 1, -max_y_, 0, 0, 1);

    cv::cuda::GpuMat warpedResImg;
    cv::cuda::warpPerspective (img1_gpu, warpedResImg, translation,
                               cv::Size (min_x_-max_x_, min_y_-max_y_));

    cv::cuda::GpuMat warpedImageTemp;
    cv::cuda::warpPerspective (img2_gpu, warpedImageTemp, translation,
                                cv::Size (min_x_ - max_x_, min_y_ - max_y_));
    cv::cuda::GpuMat warpedImage2;
    cv::cuda::warpAffine (warpedImageTemp, warpedImage2, A,
                          cv::Size (min_x_ - max_x_, min_y_ - max_y_));

    cv::cuda::GpuMat mask;
    cv::cuda::threshold (warpedImage2, mask, 1, 255, cv::THRESH_BINARY);
    int type = warpedResImg.type();

    warpedResImg.convertTo (warpedResImg, CV_32FC3);
    warpedImage2.convertTo (warpedImage2, CV_32FC3);
    mask.convertTo (mask, CV_32FC3, 1.0/255);
    cv::Mat mask_;
    mask.download (mask_);

    cv::cuda::GpuMat dst (warpedImage2.size(), warpedImage2.type());
    cv::cuda::multiply (mask, warpedImage2, warpedImage2);

    cv::Mat diff_ = cv::Scalar::all (1.0) - mask_;
    cv::cuda::GpuMat diff (diff_);
    cv::cuda::multiply(diff, warpedResImg, warpedResImg);
    cv::cuda::add (warpedResImg, warpedImage2, dst);
    dst.convertTo (dst, type);

    cv::Mat ret;
    dst.download (ret);
    std::cout << "combinePair: end" << std::endl;
    return ret;
}

cv::Mat combine (std::vector<cv::Mat>& imageList) {
    cv::Mat result = imageList[0];
    for (int i = 1; i < imageList.size(); i++) {
        cv::Mat image = imageList[i];
        cout << i << endl;
        auto start = high_resolution_clock::now();
        result = combinePair (result, image);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds> (end-start);
        cout << "time taken by the functions: " << duration.count() << endl;
        float h = result.rows;
        float w = result.cols;
        if (h > 4000 || w > 4000) {
            if (h > 4000) {
                float hx = 4000.0/h;
                h = h * hx;
                w = w * hx;
            }
            else if (w > 4000) {
                float wx = 4000.0/w;
                w = w * wx;
                h = h * wx;
            }
        }
        cout << h << " " << w << endl;
        cv::resize (result, result, cv::Size (w, h));
    }
    return result;
}

void readData (std::string& filename,
               std::vector<imageData>& dataMatrix) {
    std::ifstream file;
    file.open (filename);
    if (file.is_open()) {
        std::string line;
        while (getline (file, line)) {
            std::stringstream ss (line);
            std::string word;
            imageData id;
            int i = 0;
            while (getline (ss, word, ',')) {
                if (i == 0)	{ id.imageName = word; }
                else if (i == 1) { id.latitude = stof(word); }
                else if (i == 2) { id.longitude = stof(word); }
                else if (i == 3) {
                    id.altitudeFeet = stof (word);
                    id.altitudeMeter = id.altitudeFeet * 0.3048;
                }
                else if (i == 4) { id.yaw = stof(word); }
                else if (i == 5) { id.pitch = stof(word); }
                else if (i == 6) { id.roll = stof(word); }
                i++;
            }
            dataMatrix.push_back (id);
        }
    }
}

void getImageList (std::vector<cv::Mat>& imageList,
                   std::vector<imageData>& dataMatrix,
                   std::string base_path) {
    for (auto data : dataMatrix) {
        std::string img_path = base_path + data.imageName;
        cv::Mat img = cv::imread (img_path, 1);
        // cout << img.empty () << endl;
        // cout << img_path << endl;
        imageList.push_back (img);
    }
}

void changePerspective(std::vector<cv::Mat>& imageList,
    std::vector<imageData>& dataMatrix) {
    std::cout << "Warping Images Now" << std::endl;
    int n = imageList.size();
    for (int i = 0; i < n; i++) {
        // get a transformation to the origin plane
        // to unwind any rotations done to the camera before the image was taken
        cv::Mat M;
        getTransformationToOriginPlane(dataMatrix[i], M);
        cv::Mat correctedImage = warpPerspectiveWithPadding(imageList[i], M);
        cv::imwrite("../../output/temp/" + dataMatrix[i].imageName, correctedImage);
    }
    std::cout << "Image Warping Done" << std::endl;
}

int main() {
    std::string filename = "../../input/image_catalog.txt";
    std::vector<imageData> dataMatrix;
    readData(filename, dataMatrix);
    std::vector<cv::Mat> imageList;
    std::string base_path = "../../input/";
    getImageList(imageList, dataMatrix, base_path);
    changePerspective(imageList, dataMatrix);
    imageList.clear();
    base_path = "../../output/temp/";
    getImageList(imageList, dataMatrix, base_path);
    cv::Mat result = combine(imageList);
    cv::imwrite("../../output/result.png", result);
    return 0;
}
