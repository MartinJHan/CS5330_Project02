/**
 * Name: Junyao (Martin) Han & Xinyue (Sia) Xuan
 * Date: 2026-02-08
 * Purpose:
 *   Project 2 Task 7: Custom CBIR design for yellow object detection
 *   Combines yellow color detection with texture (Sobel) and DNN features
 *
 * Usage:
 *   ./cbir_custom <target_image> <image_dir> <dnn_csv> <topN>
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

#include "feature.h" 

/**
 * Summary:
 *   Computes the ratio of yellow pixels in an image.
 *   Uses HSV color space: Hue 20-30, Saturation > 100, Value > 100
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
 *   Computes texture distance using histogram intersection.
 *   Reuses sobelMagHist1D and histIntersection from feature.h
 */
double computeTextureDistance(const cv::Mat &img1, const cv::Mat &img2, int bins) {
    std::vector<float> hist1, hist2;
    
    if (sobelMagHist1D(img1, bins, hist1) != 0) return 1.0;
    if (sobelMagHist1D(img2, bins, hist2) != 0) return 1.0;
    
    double intersection = histIntersection(hist1, hist2);
    return 1.0 - intersection;  // Convert similarity to distance
}

/**
 * Summary:
 *   Reads DNN features from CSV file.
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
 *   Computes cosine distance between two feature vectors.
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
 *   Computes custom distance for yellow object detection.
 *   Balanced weights: color + texture + DNN
 *   
 *   Distance = 0.3 × |ΔYellow| + 0.3 × TextureDist + 0.4 × DNN_distance
 *
 * Weight rationale:
 *   - Yellow color (30%): Identifies yellow objects
 *   - Texture (30%): Distinguishes smooth banana surface from other objects
 *   - DNN (40%): Provides semantic context
 */
double computeCustomDistance(float yellowT, double textureDist, double dnnDist) {
    double yellowDist = (double)yellowT;
    
    // Balanced weights: color + texture = 60%, DNN = 40%
    return 0.3 * yellowDist + 0.3 * textureDist + 0.4 * dnnDist;
}

/**
 * Summary:
 *   Checks if filename is an image file.
 */
static bool isImageName(const char *name) {
    return (strstr(name, ".jpg") || strstr(name, ".JPG") ||
            strstr(name, ".jpeg") || strstr(name, ".JPEG") ||
            strstr(name, ".png") || strstr(name, ".PNG"));
}

/**
 * Summary:
 *   Main entry point for custom banana detector.
 */
int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] 
                  << " <target_image> <image_dir> <dnn_csv> <topN>\n";
        std::cout << "Example: ./cbir_custom pic.0343.jpg ./data/olympus "
                  << "./data/ResNet18_olym.csv 10\n";
        return -1;
    }
    
    const char *targetPath = argv[1];
    const char *imageDir = argv[2];
    const char *csvPath = argv[3];
    int topN = atoi(argv[4]);
    
    const int textureBins = 16;  // Number of bins for Sobel histogram
    
    if (topN <= 0) {
        std::cerr << "Error: topN must be positive\n";
        return -1;
    }
    
    // Read target image
    cv::Mat targetImg = cv::imread(targetPath, cv::IMREAD_COLOR);
    if (targetImg.empty()) {
        std::cerr << "Error: Cannot read target image " << targetPath << "\n";
        return -1;
    }
    
    // Compute target features
    float targetYellow = computeYellowRatio(targetImg);
    
    std::cout << "Target yellow ratio: " << targetYellow << "\n";
    
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
    
    std::cout << "Looking for target filename: " << targetFilename << "\n";
    
    // Find target in DNN database
    int targetDNNIdx = -1;
    for (size_t i = 0; i < dnnFilenames.size(); i++) {
        if (dnnFilenames[i] == targetFilename) {
            targetDNNIdx = i;
            break;
        }
    }
    
    if (targetDNNIdx == -1) {
        std::cerr << "Warning: Target not found in DNN database\n";
        return -1;
    }
    
    // Scan directory and compute distances
    DIR *dirp = opendir(imageDir);
    if (!dirp) {
        std::cerr << "Error: Cannot open directory " << imageDir << "\n";
        return -1;
    }
    
    std::vector<std::pair<std::string, double>> results;
    struct dirent *dp;
    char imgPath[1024];
    
    int processedCount = 0;
    
    while ((dp = readdir(dirp)) != NULL) {
        if (dp->d_name[0] == '.' || !isImageName(dp->d_name)) continue;
        
        strcpy(imgPath, imageDir);
        strcat(imgPath, "/");
        strcat(imgPath, dp->d_name);
        
        cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);
        if (img.empty()) continue;
        
        // 1. Compute yellow ratio
        float imgYellow = computeYellowRatio(img);
        double yellowDist = fabs(targetYellow - imgYellow);
        
        // 2. Compute texture distance (using existing Sobel function)
        double textureDist = computeTextureDistance(targetImg, img, textureBins);
        
        // 3. Find DNN feature and compute distance
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
        
        // 4. Compute combined distance
        double combinedDist = 0.3 * yellowDist + 0.3 * textureDist + 0.4 * dnnDist;
        
        results.emplace_back(imgPath, combinedDist);
        processedCount++;
    }
    
    closedir(dirp);
    
    std::cout << "Processed " << processedCount << " images\n";
    
    // Sort and display results
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
    
    // Print bottom 5
    std::cout << "\nBottom 5 matches (least similar):\n";
    if (results.size() >= 5) {
        for (size_t i = results.size() - 5; i < results.size(); i++) {
            std::cout << (i - (results.size() - 5) + 1) << ": " 
                      << results[i].first
                      << "  (distance: " << results[i].second << ")\n";
        }
    }
    
    return 0;
}