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

using namespace Eigen;
using namespace std;
using namespace chrono;

const float PI = 3.14285714286;

const string DATASETS_DIR = "../../datasets";
const string CURRENT_SET_DIR = "1";
const string INPUT_DIR = "input";
const string PRE_PROCESS_DIR = "pre-process";
const string INTERMEDIATE_EDUCATIONAL_DIR = "intermediate-educational";
const string OUTPUT_DIR = "output";

struct imageData {
    string imageName = "";
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
        //cout << "width = " << w << endl;
        return w;
    }

    int getHeight() {
        int h = max_y - min_y;
        //cout << "height = " << h << endl;
        return h;
    }
};

inline ostream& operator<<(ostream& oss, const imageRange2d& other)
{
    oss << "min_x: " << other.min_x << ", max_x: " << other.max_x << ", min_y: " << other.min_y << ", max_y: " << other.max_y;
    return oss;
}

void getMatFromGpuMat(cv::Mat & mat, cv::cuda::GpuMat & mat_gpu, int original_type = -1) {
    // convert from gpu matrix
    if (original_type != -1)
    {
        // if we are passed an original type we must convert the pixel data type of the gpu matrix
        cv::cuda::GpuMat mat_gpu_original;
        mat_gpu.convertTo (mat_gpu_original, original_type);
        mat_gpu_original.download (mat);
    }
    else
    {
        mat_gpu.download (mat);
    }
}

void printMat(cv::Mat& mat) {
    int rows = mat.rows;
    int cols = mat.cols;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << mat.at<double>(i, j) << " ";
        }
        cout << endl;
    }
}

void printGpuMatReport(cv::cuda::GpuMat & mat_gpu, const char * name, int original_type = -1)
{
    cout << ">>> matrix = " << name << endl;
    
    // convert from gpu matrix
    cv::Mat mat;
    getMatFromGpuMat(mat, mat_gpu, original_type);

    // change to grayscale
    //cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
    //int type = mat.type();
    //int cn = CV_MAT_CN(type);
    //cout << "    type = " << type << ", cn = " << cn << endl;

    // dimensions
    cout << "    width = " << mat.cols << ", height = " << mat.rows << endl;

    // number of non-zero elements
    //cout << "    num non-zero elements = " << cv::countNonZero(mat)  << endl;
}

string getDatasetPath() {
    string path = DATASETS_DIR + "/" + CURRENT_SET_DIR + "/";
    return path;
}

string getInputPath() {
    string path = getDatasetPath() + INPUT_DIR + "/";
    return path;
}

string getPreProcessPath() {
    string path = getDatasetPath() + PRE_PROCESS_DIR + "/";
    return path;
}

string getIntermediateEducationalPath() {
    string path = getDatasetPath() + INTERMEDIATE_EDUCATIONAL_DIR + "/";
    return path;
}

string getOutputPath() {
    string path = getDatasetPath() + OUTPUT_DIR + "/";
    return path;
}

void writeIntermediateImage(cv::cuda::GpuMat& mat_gpu, const char* name, int iteration, int step, int original_type = -1) {
    // convert from gpu matrix
    cv::Mat mat;
    getMatFromGpuMat(mat, mat_gpu, original_type);

    // construct filename
    string intermediate_dir = getIntermediateEducationalPath();
    string intermediate_filename = intermediate_dir + to_string(iteration) + "_" + to_string(step) + "_" + name  +  + ".png";

    // write image
    cv::imwrite(intermediate_filename, mat);
}

void writeIntermediateText(cv::cuda::GpuMat& mat_gpu, const char* name, int iteration, int step, int original_type = -1) {
    // convert from gpu matrix
    cv::Mat mat;
    getMatFromGpuMat(mat, mat_gpu, original_type);

    // construct filename
    string intermediate_dir = getIntermediateEducationalPath();
    string intermediate_filename = intermediate_dir + to_string(iteration) + "_" + to_string(step) + "_" + name  +  + ".txt";

    // write text file
    ofstream myfile;
    myfile.open (intermediate_filename);
    myfile << mat;
    myfile.close();
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

void getRangeFromRectangleCorners(vector <vector<cv::Point2f>>& rectangles, imageRange2d& range2d) {
    // create the range to just include all the corners of all the rectangles in the collection

    // mmin & max values start out in opposite postions
    // and are iteratively pushed in the other direction
    float min_x = 1e9, max_x = -1e9;
    float min_y = 1e9, max_y = -1e9;

    // do the loop
    for (auto & rect : rectangles) {
        //cout << "rect = " << rect << endl;
        for (auto & corner : rect) {
            min_x = (min_x > corner.x)? corner.x : min_x;
            max_x = (max_x < corner.x)? corner.x : max_x;
            min_y = (min_y > corner.y)? corner.y : min_y;
            max_y = (max_y < corner.y)? corner.y : max_y;
        }
    }

    // round floats to appropriate integer value
    range2d.min_x = static_cast<int>(min_x - 0.5);
    range2d.max_x = static_cast<int>(max_x + 0.5);
    range2d.min_y = static_cast<int>(min_y - 0.5);
    range2d.max_y = static_cast<int>(max_y + 0.5);
    //cout << "getRangeFromRectangleCorners: range2d = " << range2d << endl;
}

void getScaledPaddedTransformation(cv::Mat& transformation, imageSize2d& size2d, cv::Mat& padded_transformation, imageRange2d& range2d) {
    // translate corners according to the scale factor
    vector<cv::Point2f> corners { 
            cv::Point2f(0.,0.), 
            cv::Point2f(0.,size2d.height/ size2d.scale),
            cv::Point2f(size2d.width/ size2d.scale,size2d.height/ size2d.scale),
            cv::Point2f(size2d.width/ size2d.scale,0.)
    };
    //cout << "corners = " << corners << endl;

    // apply the transformation to get the warped corners
    vector<cv::Point2f> warpedCorners;
    cv::perspectiveTransform(corners, warpedCorners, transformation);
    //cout << "warpedCorners = " << warpedCorners << endl;
    //cout << "transformation = " << transformation << endl;

    //get x & y range to encompass all the warped corners
    vector <vector<cv::Point2f>> rectangles;
    rectangles.push_back(warpedCorners);
    getRangeFromRectangleCorners(rectangles, range2d);
}

cv::Mat warpPerspectiveWithPadding (const cv::Mat& image, cv::Mat& transformation) {
    // reduce the individual image size so we have enough room to create the final stitched tapestry 
    imageSize2d size2d{ static_cast<float>(image.cols), static_cast<float>(image.rows), 2.0 };
    //cout << "size2d = " << size2d << endl;
    cv::Mat small_img;
    cv::resize (image, small_img, cv::Size (size2d.getNewWidth(), size2d.getNewHeight()));

    // get a padded transformation scaled to to our reduced image size
    cv::Mat padded_transformation;
    imageRange2d range2d;
    getScaledPaddedTransformation(transformation, size2d, padded_transformation, range2d);
    cv::Mat translation = (cv::Mat_<double>(3,3) << 1, 0, -range2d.min_x, 0, 1, -range2d.min_y, 0, 0, 1);
    cv::Mat fullTransformation = translation * transformation;

    // perform the image perspective warp
    cv::cuda::GpuMat gpu_img (small_img);
    cv::cuda::GpuMat gpu_img_wp;
    cv::cuda::GpuMat gpu_ft (fullTransformation);
    //cout << "fullTransformation = " << fullTransformation << endl;
    cv::Size gpu_size{ range2d.getWidth(), range2d.getHeight() };
    //cout << "gpu_size = " << gpu_size << endl;
    cv::cuda::warpPerspective(gpu_img, gpu_img_wp, fullTransformation, gpu_size);

    // download the final result matrix
    cv::Mat result (gpu_img_wp.size(), gpu_img_wp.type());
    gpu_img_wp.download (result);
    return result;
}

void getKeypoints(  cv::cuda::SURF_CUDA & detector,
                    cv::cuda::GpuMat & img_gpu, 
                    cv::cuda::GpuMat & descriptors_gpu,
                    vector<cv::KeyPoint> & keypoints)
{
    cv::cuda::GpuMat img_gray_gpu;
    cv::cuda::cvtColor (img_gpu, img_gray_gpu, cv::COLOR_BGR2GRAY);
    cv::cuda::GpuMat mask_gpu;
    cv::cuda::threshold (img_gray_gpu, mask_gpu, 1, 255, cv::THRESH_BINARY);

    cv::cuda::GpuMat keypoints1_gpu;
    detector (img_gray_gpu, mask_gpu, keypoints1_gpu, descriptors_gpu);
    detector.downloadKeypoints (keypoints1_gpu, keypoints);
}

void matchKeypointDescriptors(  cv::cuda::GpuMat & descriptors1_gpu, cv::cuda::GpuMat & descriptors2_gpu, 
                                vector<cv::KeyPoint> & keypoints1, vector<cv::KeyPoint> & keypoints2,
                                vector<cv::Point2f> & src_pts, vector<cv::Point2f> & dst_pts)
{
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher ();
    vector<vector<cv::DMatch>> knn_matches;
    matcher->knnMatch (descriptors2_gpu, descriptors1_gpu, knn_matches, 2);

    vector<cv::DMatch> matches;
    vector<vector<cv::DMatch>>::const_iterator it;
    for (it = knn_matches.begin(); it != knn_matches.end(); ++it) {
        if(it->size() > 1 && (*it)[0].distance/(*it)[1].distance < 0.55) {
            matches.push_back((*it)[0]);
        }
    }

    for (auto m : matches) {
        src_pts.push_back (keypoints2[m.queryIdx].pt);
        dst_pts.push_back (keypoints1[m.trainIdx].pt);
    }
}

void getAffineTransformation(  float height1, float width1, float height2, float width2, 
                               vector<cv::Point2f> & src_pts, vector<cv::Point2f> & dst_pts, 
                               cv::Mat & A, imageRange2d& range2d)
{
    A = cv::estimateRigidTransform(src_pts, dst_pts, false);

    vector<cv::Point2f> corners1{
            cv::Point2f(0.,0.),
            cv::Point2f(0.,height1),
            cv::Point2f(width1,height1),
            cv::Point2f(width1,0.)
    };

    vector<cv::Point2f> corners2{
            cv::Point2f(0.,0.),
            cv::Point2f(0.,height2),
            cv::Point2f(width2,height2),
            cv::Point2f(width2,0.)
    };

    vector<cv::Point2f> warpedCorners2{
            cv::Point2f(0.,0.),
            cv::Point2f(0.,0.),
            cv::Point2f(0.,0.),
            cv::Point2f(0.,0.)
    };

    vector<vector<cv::Point2f>> allCorners;
    allCorners.push_back(corners1);

    for (int i = 0; i < 4; i++) {
        float cornerX = corners2[i].x;
        float cornerY = corners2[i].y;
        warpedCorners2[i].x = A.at<double> (0,0) * cornerX + A.at<double> (0,1) * cornerY + A.at<double> (0,2);
        warpedCorners2[i].y = A.at<double> (1,0) * cornerX + A.at<double> (1,1) * cornerY + A.at<double> (1,2);
    }
    allCorners.push_back(warpedCorners2);

    //get x & y range to encompass all the warped corners
    getRangeFromRectangleCorners(allCorners, range2d);
}

cv::Mat combinePair (cv::Mat& img1, cv::Mat& img2) {
    static int call_count = 0;
    call_count++;
    int step = 0;

    // convert to a form our gpu can handle
    cv::cuda::GpuMat img1_gpu (img1), img2_gpu (img2);

    // save original image type (type is the data type used to store each pixel)
    // we will later convert image types to a floating point for math calculations
    // and then when done we will convert back to the original unsigned 8bit type
    // which is necessary for image display or writing file to disk 
    int original_type = img1_gpu.type();

    // get keypoint descriptors for the two images
    cv::cuda::SURF_CUDA detector;
    cv::cuda::GpuMat descriptors1_gpu, descriptors2_gpu;
    vector<cv::KeyPoint> keypoints1, keypoints2;
    getKeypoints(detector, img1_gpu, descriptors1_gpu, keypoints1);
    getKeypoints(detector, img2_gpu, descriptors2_gpu, keypoints2);

    // match the keypoint descriptors
    vector<cv::Point2f> src_pts, dst_pts;
    matchKeypointDescriptors(descriptors1_gpu, descriptors2_gpu, keypoints1, keypoints2, src_pts, dst_pts);

    // get affine transformation from our src pts (image 2) to our dst pts (image 1)
    float height1 = static_cast<float>(img1.rows), width1 = static_cast<float>(img1.cols);
    float height2 = static_cast<float>(img2.rows), width2 = static_cast<float>(img2.cols);
    cv::Mat A;
    imageRange2d range2d;
    getAffineTransformation(height1, width1, height2, width2, src_pts, dst_pts, A, range2d);
    cout << "range2d(" << range2d.getWidth() << "," << range2d.getHeight() << ")" << endl;
    cv::Mat translation = (cv::Mat_<double>(3,3) << 1, 0, -range2d.min_x, 0, 1, -range2d.min_y, 0, 0, 1);

    // homographically warp each image so that it is sized correctly 
    // relative to the size of the bounding rectangle
    // who was sized to exactly accomodate the 4 corners of both of the images
    cv::cuda::GpuMat warpedPerspectiveImg1_gpu;
    cv::cuda::warpPerspective (img1_gpu, warpedPerspectiveImg1_gpu, translation, cv::Size (range2d.getWidth(), range2d.getHeight()));
    printGpuMatReport(img1_gpu, "img1_gpu");
    printGpuMatReport(warpedPerspectiveImg1_gpu, "warpedPerspectiveImg1_gpu");
    writeIntermediateImage(warpedPerspectiveImg1_gpu, "warpedPerspectiveImg1_gpu", call_count, ++step);
    //
    cv::cuda::GpuMat warpedPerspectiveImg2_gpu;
    cv::cuda::warpPerspective (img2_gpu, warpedPerspectiveImg2_gpu, translation, cv::Size (range2d.getWidth(), range2d.getHeight()));
    printGpuMatReport(img2_gpu, "img2_gpu");
    printGpuMatReport(warpedPerspectiveImg2_gpu, "warpedPerspectiveImg2_gpu");
    writeIntermediateImage(warpedPerspectiveImg2_gpu, "warpedPerspectiveImg2_gpu", call_count, ++step);

    // affinely warp image 2 so that it is positioned correctly
    // within the bounding rectangle relative to the position of image 1
    // (remember the affine transformation was created by matching keypoints of the two images)
    cv::cuda::GpuMat warpedPerspectiveAffineImg2_gpu;
    cv::cuda::warpAffine (warpedPerspectiveImg2_gpu, warpedPerspectiveAffineImg2_gpu, A, cv::Size (range2d.getWidth(), range2d.getHeight()));
    printGpuMatReport(warpedPerspectiveAffineImg2_gpu, "warpedPerspectiveAffineImg2_gpu");
    writeIntermediateImage(warpedPerspectiveAffineImg2_gpu, "warpedPerspectiveAffineImg2_gpu", call_count, ++step);

    // apply a threshhold to our affinely warped image 2 and save as a mask
    // this mask will indicate the portion of our combined images reserved for our image 2
    cv::cuda::GpuMat mask_gpu;
    cv::cuda::threshold (warpedPerspectiveAffineImg2_gpu, mask_gpu, 1, 255, cv::THRESH_BINARY);
    writeIntermediateImage(mask_gpu, "warpedPerspectiveAffineImg2_mask_gpu", call_count, ++step);
    writeIntermediateText(mask_gpu, "warpedPerspectiveAffineImg2_mask_gpu", call_count, step);

    // convert our 2 images and our mask to 32bit floating point type for our math calculations
    warpedPerspectiveImg1_gpu.convertTo (warpedPerspectiveImg1_gpu, CV_32FC3);
    warpedPerspectiveAffineImg2_gpu.convertTo (warpedPerspectiveAffineImg2_gpu, CV_32FC3);
    mask_gpu.convertTo (mask_gpu, CV_32FC3, 1.0/255);

    // create an empty combined image of the correct size type
    cv::cuda::GpuMat combined_gpu (warpedPerspectiveAffineImg2_gpu.size(), warpedPerspectiveAffineImg2_gpu.type());

    // multiply our affinely warped image 2 by our mask to get only the overlay image data
    cv::cuda::multiply (mask_gpu, warpedPerspectiveAffineImg2_gpu, warpedPerspectiveAffineImg2_gpu);
    writeIntermediateImage(warpedPerspectiveAffineImg2_gpu, "warpedPerspectiveAffineImg2_gpu_multiply_by_mask_gpu", call_count, ++step, original_type);

    // create our difference mask which will be used to carve out a place in our image 1 to receive our image 2
    cv::Mat mask;
    mask_gpu.download (mask);
    cv::Mat difference_mask = cv::Scalar::all (1.0) - mask;
    cv::cuda::GpuMat difference_mask_gpu (difference_mask);
    writeIntermediateImage(difference_mask_gpu, "warpedPerspectiveAffineImg2_difference_mask_gpu", call_count, ++step, original_type);
    writeIntermediateText(difference_mask_gpu, "warpedPerspectiveAffineImg2_difference_mask_gpu", call_count, step);

    // multiply our perspective warped image 1 by our difference mask to get the portion of the image data we will retain
    cv::cuda::multiply(difference_mask_gpu, warpedPerspectiveImg1_gpu, warpedPerspectiveImg1_gpu);
    writeIntermediateImage(warpedPerspectiveImg1_gpu, "warpedPerspectiveImg1_gpu_multiply_by_difference_mask_gpu", call_count, ++step, original_type);

    // add our two correctly masked image 1 and image 2 to get the combined image
    cv::cuda::add (warpedPerspectiveImg1_gpu, warpedPerspectiveAffineImg2_gpu, combined_gpu);
    combined_gpu.convertTo (combined_gpu, original_type);
    writeIntermediateImage(combined_gpu, "combined_gpu", call_count, ++step);

    // get a cpu version of the combined image
    cv::Mat combined;
    combined_gpu.download (combined);
    return combined;
}

cv::Mat combine (vector<cv::Mat>& imageList) {
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
        cout << "result(" << w << "," << h << ")" << endl;
        cv::resize (result, result, cv::Size (w, h));
        cout << "________________________________________" << endl;
    }
    return result;
}

void readData (string& filename,
               vector<imageData>& dataMatrix) {
    ifstream file;
    file.open (filename);
    if (file.is_open()) {
        string line;
        while (getline (file, line)) {
            stringstream ss (line);
            string word;
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

void getImageList (vector<cv::Mat>& imageList,
                   vector<imageData>& dataMatrix,
                   string base_path) {
    for (auto data : dataMatrix) {
        string img_path = base_path + data.imageName;
        cv::Mat img = cv::imread (img_path, 1);
        // cout << img.empty () << endl;
        // cout << img_path << endl;
        imageList.push_back (img);
    }
}

void changePerspective(vector<cv::Mat>& imageList,
    vector<imageData>& dataMatrix) {
    cout << "Warping Images Now" << endl;
    int n = imageList.size();
    for (int i = 0; i < n; i++) {
        // get a transformation to the origin plane
        // to unwind any rotations done to the camera before the image was taken
        cv::Mat M;
        getTransformationToOriginPlane(dataMatrix[i], M);
        cv::Mat correctedImage = warpPerspectiveWithPadding(imageList[i], M);
        cv::imwrite(getPreProcessPath() + dataMatrix[i].imageName, correctedImage);
    }
    cout << "Image Warping Done" << endl;
}

int main() {
    string filename = getInputPath() + "image_catalog.txt";
    vector<imageData> dataMatrix;
    readData(filename, dataMatrix);
    vector<cv::Mat> imageList;
    string base_path = getInputPath();
    getImageList(imageList, dataMatrix, base_path);
    changePerspective(imageList, dataMatrix);
    imageList.clear();
    base_path = getPreProcessPath();
    getImageList(imageList, dataMatrix, base_path);
    cv::Mat result = combine(imageList);
    cv::imwrite(getOutputPath() + "result.png", result);
    return 0;
}
