/**
 * Name: Junyao (Martin) Han & Xinyue (Sia) Xuan
 * Date: 2026-02-08
 * Purpose:
 *   Project 2 Task 7: Custom CBIR design for yellow object detection
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

// 移除 #include "feature.h"

/**
 * Summary:
 *   Computes the ratio of yellow pixels.
 */
float computeYellowRatio(const cv::Mat &img) {
    if (img.empty() || img.channels() != 3) return 0.0f;
    
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    
    int yellowCount = 0;
    int totalPixels = img.rows * img.cols;
    
    for (int y = 0; y < hsv.rows; y++) {
        for (int x = 0; x < hsv.cols; x++) {
            cv::Vec3b pixel = hsv.at<cv::Vec3b>(y, x);
            uchar H = pixel[0];
            uchar S = pixel[1];
            uchar V = pixel[2];
            
            if (H >= 20 && H <= 30 && S > 100 && V > 100) {
                yellowCount++;
            }
        }
    }
    
    return (float)yellowCount / (float)totalPixels;
}

/**
 * Summary:
 *   Computes average Sobel gradient magnitude as texture feature.
 */
float computeTextureMagnitude(const cv::Mat &img) {
    if (img.empty()) return 0.0f;
    
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img;
    }
    
    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    
    cv::Mat magnitude;
    cv::magnitude(gx, gy, magnitude);
    
    cv::Scalar meanMag = cv::mean(magnitude);
    return (float)meanMag[0];
}

/**
 * Summary:
 *   Reads DNN features from CSV.
 */
int readDNNFeatures(const char *csvPath,
                    std::vector<std::string> &filenames,
                    std::vector<std::vector<float>> &features) {
    FILE *fp = fopen(csvPath, "r");
    if (!fp) {
        std::cerr << "Error: Cannot open CSV file " << csvPath << std::endl;
        return -1;
    }
    
    filenames.clear();
    features.clear();
    
    char line[65536];
    while (fgets(line, sizeof(line), fp)) {
        char imgname[512];
        std::vector<float> feat;
        
        char *token = strtok(line, ",");
        if (!token) continue;
        
        strcpy(imgname, token);
        
        while ((token = strtok(NULL, ",")) != NULL) {
            feat.push_back(atof(token));
        }
        
        if (feat.size() != 512) continue;
        
        filenames.push_back(std::string(imgname));
        features.push_back(feat);
    }
    
    fclose(fp);
    std::cout << "Loaded " << filenames.size() << " DNN features\n";
    return 0;
}

/**
 * Summary:
 *   Computes cosine distance.
 */
double computeCosineDistance(const std::vector<float> &a, 
                             const std::vector<float> &b) {
    if (a.size() != b.size() || a.empty()) return 2.0;
    
    double dot = 0.0, normA = 0.0, normB = 0.0;
    
    for (size_t i = 0; i < a.size(); i++) {
        dot += (double)a[i] * (double)b[i];
        normA += (double)a[i] * (double)a[i];
        normB += (double)b[i] * (double)b[i];
    }
    
    normA = sqrt(normA);
    normB = sqrt(normB);
    
    if (normA < 1e-10 || normB < 1e-10) return 2.0;
    
    double cosSim = dot / (normA * normB);
    if (cosSim > 1.0) cosSim = 1.0;
    if (cosSim < -1.0) cosSim = -1.0;
    
    return 1.0 - cosSim;
}

/**
 * Summary:
 *   Checks if filename is an image.
 */
static bool isImageName(const char *name) {
    return (strstr(name, ".jpg") || strstr(name, ".JPG") ||
            strstr(name, ".jpeg") || strstr(name, ".JPEG") ||
            strstr(name, ".png") || strstr(name, ".PNG"));
}

/**
 * Summary:
 *   Main entry point.
 */
int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] 
                  << " <target_image> <image_dir> <dnn_csv> <topN>\n";
        return -1;
    }
    
    const char *targetPath = argv[1];
    const char *imageDir = argv[2];
    const char *csvPath = argv[3];
    int topN = atoi(argv[4]);
    
    if (topN <= 0) {
        std::cerr << "Error: topN must be positive\n";
        return -1;
    }
    
    // Read target image
    cv::Mat targetImg = cv::imread(targetPath, cv::IMREAD_COLOR);
    if (targetImg.empty()) {
        std::cerr << "Error: Cannot read target image\n";
        return -1;
    }
    
    // Compute target features
    float targetYellow = computeYellowRatio(targetImg);
    float targetTexture = computeTextureMagnitude(targetImg);
    
    std::cout << "Target yellow ratio: " << targetYellow << "\n";
    std::cout << "Target texture magnitude: " << targetTexture << "\n";
    
    // Load DNN features
    std::vector<std::string> dnnFilenames;
    std::vector<std::vector<float>> dnnFeatures;
    
    if (readDNNFeatures(csvPath, dnnFilenames, dnnFeatures) != 0) {
        return -1;
    }
    
    // Extract target filename
    std::string targetFilename = std::string(targetPath);
    size_t lastSlash = targetFilename.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        targetFilename = targetFilename.substr(lastSlash + 1);
    }
    
    // Find target in DNN database
    int targetDNNIdx = -1;
    for (size_t i = 0; i < dnnFilenames.size(); i++) {
        if (dnnFilenames[i] == targetFilename) {
            targetDNNIdx = i;
            break;
        }
    }
    
    if (targetDNNIdx == -1) {
        std::cerr << "Error: Target not found in DNN database\n";
        return -1;
    }
    
    // Scan directory
    DIR *dirp = opendir(imageDir);
    if (!dirp) {
        std::cerr << "Error: Cannot open directory\n";
        return -1;
    }
    
    std::vector<std::pair<std::string, double>> results;
    struct dirent *dp;
    char imgPath[1024];
    
    while ((dp = readdir(dirp)) != NULL) {
        if (dp->d_name[0] == '.' || !isImageName(dp->d_name)) continue;
        
        strcpy(imgPath, imageDir);
        strcat(imgPath, "/");
        strcat(imgPath, dp->d_name);
        
        cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);
        if (img.empty()) continue;
        
        // Compute features
        float imgYellow = computeYellowRatio(img);
        float imgTexture = computeTextureMagnitude(img);
        
        // Compute distances
        double yellowDist = fabs(targetYellow - imgYellow);
        double textureDist = fabs(targetTexture - imgTexture) / 50.0;
        if (textureDist > 1.0) textureDist = 1.0;
        
        // Find DNN distance
        double dnnDist = 1.0;
        for (size_t i = 0; i < dnnFilenames.size(); i++) {
            if (dnnFilenames[i] == std::string(dp->d_name)) {
                dnnDist = computeCosineDistance(
                    dnnFeatures[targetDNNIdx], 
                    dnnFeatures[i]
                );
                break;
            }
        }
        
        // Combined distance: 0.45 yellow + 0.1 texture + 0.45 DNN
        double combinedDist = 0.45 * yellowDist + 0.1 * textureDist + 0.45 * dnnDist;
        
        results.emplace_back(imgPath, combinedDist);
    }
    
    closedir(dirp);
    
    // Sort and display
    std::sort(results.begin(), results.end(),
              [](const auto &a, const auto &b) {
                  return a.second < b.second;
              });
    
    int n = std::min(topN, (int)results.size());
    std::cout << "\nTop " << n << " matches:\n";
    for (int i = 0; i < n; i++) {
        std::cout << (i+1) << ": " << results[i].first 
                  << "  (distance: " << results[i].second << ")\n";
    }
    
    // Bottom 5
    std::cout << "\nBottom 5 matches:\n";
    if (results.size() >= 5) {
        for (size_t i = results.size() - 5; i < results.size(); i++) {
            std::cout << (i - (results.size() - 5) + 1) << ": " 
                      << results[i].first
                      << "  (distance: " << results[i].second << ")\n";
        }
    }
    
    return 0;
}