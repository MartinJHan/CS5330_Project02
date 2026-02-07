/**
 * Name: Junyao Han
 * Date: 2026-02-06
 * Purpose:
 *   Project 2 CBIR program supporting Tasks 1-4:
 *   - Task 1: baseline 7x7 center patch feature + SSD (method=baseline)
 *   - Task 2: whole-image 2D rg chromaticity histogram + intersection (method=hist2d)
 *   - Task 3: two 3D RGB histograms (top/bottom halves) + weighted intersection (method=multihist)
 *   - Task 4: whole-image color histogram + texture histogram (Sobel magnitude) (method=colortexture)
 *
 * Usage:
 *   ./cbir_baseline <target_image> <image_dir> <topN> <method>
 *   method: baseline | hist2d | multihist | colortexture
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "feature.h"

/**
 * Summary:
 *   Converts an input image to an 8-bit, 3-channel BGR image.
 *
 * Arguments:
 *   img - Input image.
 *
 * Return value:
 *   CV_8UC3 image on success, empty Mat on failure.
 */
static cv::Mat toBGR8U(const cv::Mat &img) {
    if (img.empty()) return cv::Mat();
    if (img.type() == CV_8UC3) return img;

    cv::Mat out;
    if (img.type() == CV_8UC1) {
        cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
    } else if (img.type() == CV_8UC4) {
        cv::cvtColor(img, out, cv::COLOR_BGRA2BGR);
    } else {
        cv::Mat tmp;
        img.convertTo(tmp, CV_8U);
        if (tmp.channels() == 1) cv::cvtColor(tmp, out, cv::COLOR_GRAY2BGR);
        else if (tmp.channels() == 3) out = tmp;
        else if (tmp.channels() == 4) cv::cvtColor(tmp, out, cv::COLOR_BGRA2BGR);
        else return cv::Mat();
    }
    return out;
}

/**
 * Summary:
 *   Extracts Task 1 baseline feature from the center 7x7 patch (BGR flattened).
 *
 * Arguments:
 *   imgInput - Input image.
 *   feat     - Output feature vector.
 *
 * Return value:
 *   0 on success, negative on failure.
 */
int baselineFeature(const cv::Mat &imgInput, std::vector<float> &feat) {
    feat.clear();

    cv::Mat img = toBGR8U(imgInput);
    if (img.empty()) return -1;
    if (img.rows < 7 || img.cols < 7) return -2;

    int cy = img.rows / 2;
    int cx = img.cols / 2;
    int y0 = cy - 3;
    int x0 = cx - 3;

    if (y0 < 0 || x0 < 0 || y0 + 6 >= img.rows || x0 + 6 >= img.cols) return -3;

    feat.reserve(7 * 7 * 3);

    for (int y = y0; y <= y0 + 6; y++) {
        for (int x = x0; x <= x0 + 6; x++) {
            const cv::Vec3b &px = img.at<cv::Vec3b>(y, x);
            feat.push_back((float)px[0]);
            feat.push_back((float)px[1]);
            feat.push_back((float)px[2]);
        }
    }
    return 0;
}

/**
 * Summary:
 *   Computes SSD distance between two feature vectors (Task 1).
 *
 * Arguments:
 *   a - First feature vector.
 *   b - Second feature vector.
 *
 * Return value:
 *   SSD distance, or a large value if incompatible.
 */
double ssdDistance(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size() || a.empty()) return 1e18;

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return sum;
}

/**
 * Summary:
 *   Computes a normalized 2D rg chromaticity histogram for the whole image (Task 2).
 *
 * Arguments:
 *   imgInput - Input image.
 *   bins     - Number of bins for r and g.
 *   hist     - Output histogram (bins*bins), normalized.
 *
 * Return value:
 *   0 on success, negative on failure.
 */
int rgChromHist2D(const cv::Mat &imgInput, int bins, std::vector<float> &hist) {
    hist.assign(bins * bins, 0.0f);

    cv::Mat img = toBGR8U(imgInput);
    if (img.empty()) return -1;
    if (bins <= 1) return -2;

    int rows = img.rows;
    int cols = img.cols;
    int validCount = 0;

    // Build histogram by iterating pixels
    for (int y = 0; y < rows; y++) {
        const cv::Vec3b *rowPtr = img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < cols; x++) {
            unsigned char B = rowPtr[x][0];
            unsigned char G = rowPtr[x][1];
            unsigned char R = rowPtr[x][2];

            int sum = (int)R + (int)G + (int)B;
            if (sum == 0) continue;

            float r = (float)R / (float)sum;
            float g = (float)G / (float)sum;

            int rbin = (int)(r * bins);
            int gbin = (int)(g * bins);
            if (rbin >= bins) rbin = bins - 1;
            if (gbin >= bins) gbin = bins - 1;

            hist[gbin * bins + rbin] += 1.0f;
            validCount++;
        }
    }

    if (validCount == 0) return -3;

    // Normalize so sum(hist) = 1
    float inv = 1.0f / (float)validCount;
    for (float &v : hist) v *= inv;

    return 0;
}

/**
 * Summary:
 *   Computes histogram intersection between two normalized histograms.
 *
 * Arguments:
 *   h1 - First histogram (normalized).
 *   h2 - Second histogram (normalized).
 *
 * Return value:
 *   Intersection score in [0,1] (larger means more similar).
 */
double histIntersection(const std::vector<float> &h1, const std::vector<float> &h2) {
    if (h1.size() != h2.size() || h1.empty()) return 0.0;

    double s = 0.0;
    for (size_t i = 0; i < h1.size(); i++) {
        s += (h1[i] < h2[i]) ? h1[i] : h2[i];
    }
    return s;
}

/**
 * Summary:
 *   Computes a normalized 3D RGB histogram for rows [yStart, yEnd) (Task 3).
 *
 * Arguments:
 *   imgInput - Input image.
 *   bins     - Number of bins per channel.
 *   yStart   - Start row (inclusive).
 *   yEnd     - End row (exclusive).
 *   hist     - Output histogram (bins^3), normalized.
 *
 * Return value:
 *   0 on success, negative on failure.
 */
int rgbHist3DRegion(const cv::Mat &imgInput, int bins, int yStart, int yEnd, std::vector<float> &hist) {
    hist.assign(bins * bins * bins, 0.0f);

    cv::Mat img = toBGR8U(imgInput);
    if (img.empty()) return -1;
    if (bins <= 1) return -2;

    int rows = img.rows;
    int cols = img.cols;

    if (yStart < 0) yStart = 0;
    if (yEnd > rows) yEnd = rows;
    if (yStart >= yEnd) return -3;

    int count = 0;

    // Build histogram over the specified region
    for (int y = yStart; y < yEnd; y++) {
        const cv::Vec3b *rowPtr = img.ptr<cv::Vec3b>(y);
        for (int x = 0; x < cols; x++) {
            unsigned char B = rowPtr[x][0];
            unsigned char G = rowPtr[x][1];
            unsigned char R = rowPtr[x][2];

            int rbin = (int)((R / 256.0) * bins);
            int gbin = (int)((G / 256.0) * bins);
            int bbin = (int)((B / 256.0) * bins);

            if (rbin >= bins) rbin = bins - 1;
            if (gbin >= bins) gbin = bins - 1;
            if (bbin >= bins) bbin = bins - 1;

            int idx = rbin * (bins * bins) + gbin * bins + bbin;
            hist[idx] += 1.0f;
            count++;
        }
    }

    if (count == 0) return -4;

    // Normalize
    float inv = 1.0f / (float)count;
    for (float &v : hist) v *= inv;

    return 0;
}

/**
 * Summary:
 *   Combines top and bottom histogram intersections using equal weights (Task 3).
 *
 * Arguments:
 *   hTopT - Target top histogram.
 *   hBotT - Target bottom histogram.
 *   hTopI - Image top histogram.
 *   hBotI - Image bottom histogram.
 *
 * Return value:
 *   Distance in [0,1], smaller means more similar.
 */
double multiHistDistance(const std::vector<float> &hTopT,
                         const std::vector<float> &hBotT,
                         const std::vector<float> &hTopI,
                         const std::vector<float> &hBotI) {
    double interTop = histIntersection(hTopT, hTopI);
    double interBot = histIntersection(hBotT, hBotI);

    double dTop = 1.0 - interTop;
    double dBot = 1.0 - interBot;

    return 0.5 * dTop + 0.5 * dBot;
}

/**
 * Summary:
 *   Computes a normalized 1D histogram of Sobel gradient magnitudes (Task 4 texture).
 *
 * Arguments:
 *   imgInput - Input image.
 *   bins     - Number of magnitude bins.
 *   hist     - Output histogram (bins), normalized.
 *
 * Return value:
 *   0 on success, negative on failure.
 */
int sobelMagHist1D(const cv::Mat &imgInput, int bins, std::vector<float> &hist) {
    hist.assign(bins, 0.0f);

    cv::Mat img = toBGR8U(imgInput);
    if (img.empty()) return -1;
    if (bins <= 1) return -2;

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Sobel gradients as float
    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);

    // Magnitude
    cv::Mat mag;
    cv::magnitude(gx, gy, mag);

    int rows = mag.rows;
    int cols = mag.cols;
    int count = rows * cols;
    if (count == 0) return -3;

    // Histogram over magnitudes (clamp to [0,255] for stable binning)
    for (int y = 0; y < rows; y++) {
        const float *rowPtr = mag.ptr<float>(y);
        for (int x = 0; x < cols; x++) {
            float m = rowPtr[x];
            if (m < 0.0f) m = 0.0f;
            if (m > 255.0f) m = 255.0f;

            int bin = (int)((m / 256.0f) * bins);
            if (bin >= bins) bin = bins - 1;

            hist[bin] += 1.0f;
        }
    }

    // Normalize
    float inv = 1.0f / (float)count;
    for (float &v : hist) v *= inv;

    return 0;
}

/**
 * Summary:
 *   Combines color and texture distances using equal weights (Task 4).
 *
 * Arguments:
 *   colorT - Target color histogram.
 *   texT   - Target texture histogram.
 *   colorI - Image color histogram.
 *   texI   - Image texture histogram.
 *
 * Return value:
 *   Distance in [0,1], smaller means more similar.
 */
double colorTextureDistance(const std::vector<float> &colorT,
                            const std::vector<float> &texT,
                            const std::vector<float> &colorI,
                            const std::vector<float> &texI) {
    double interColor = histIntersection(colorT, colorI);
    double interTex = histIntersection(texT, texI);

    double dColor = 1.0 - interColor;
    double dTex = 1.0 - interTex;

    return 0.5 * dColor + 0.5 * dTex;
}

/**
 * Summary:
 *   Checks if a filename looks like an image file.
 *
 * Arguments:
 *   name - Filename to check.
 *
 * Return value:
 *   true if supported image extension, false otherwise.
 */
static bool isImageName(const char *name) {
    return (strstr(name, ".jpg") || strstr(name, ".JPG") ||
            strstr(name, ".jpeg") || strstr(name, ".JPEG") ||
            strstr(name, ".png") || strstr(name, ".PNG") ||
            strstr(name, ".ppm") || strstr(name, ".PPM") ||
            strstr(name, ".tif") || strstr(name, ".TIF") ||
            strstr(name, ".tiff") || strstr(name, ".TIFF"));
}

/**
 * Summary:
 *   Main entry point. Runs CBIR with the selected method.
 *
 * Arguments:
 *   argv[1] - target image path
 *   argv[2] - database directory
 *   argv[3] - top N
 *   argv[4] - method name
 *
 * Return value:
 *   0 on success, -1 on failure.
 */
int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cout << "usage: " << argv[0]
                  << " <target_image> <image_dir> <topN> <method>\n";
        std::cout << "method: baseline | hist2d | multihist | colortexture\n";
        return -1;
    }

    char targetPath[512];
    char dirname[512];
    int topN = atoi(argv[3]);
    std::string method = argv[4];

    strcpy(targetPath, argv[1]);
    strcpy(dirname, argv[2]);

    if (topN <= 0) {
        std::cout << "topN must be > 0\n";
        return -1;
    }

    cv::Mat targetImg = cv::imread(targetPath, cv::IMREAD_UNCHANGED);
    if (targetImg.empty()) {
        std::cout << "failed to read target image: " << targetPath << "\n";
        return -1;
    }

    // Compute target feature(s)
    std::vector<float> targetFeat;

    std::vector<float> targetTop, targetBot;      // Task3
    std::vector<float> targetColor, targetTex;    // Task4

    const int bins2d = 16;     // Task2 color
    const int binsRGB = 8;     // Task3 RGB bins
    const int binsTex = 16;    // Task4 texture bins

    if (method == "baseline") {
        if (baselineFeature(targetImg, targetFeat) != 0) return -1;

    } else if (method == "hist2d") {
        if (rgChromHist2D(targetImg, bins2d, targetFeat) != 0) return -1;

    } else if (method == "multihist") {
        int mid = targetImg.rows / 2;
        if (rgbHist3DRegion(targetImg, binsRGB, 0, mid, targetTop) != 0) return -1;
        if (rgbHist3DRegion(targetImg, binsRGB, mid, targetImg.rows, targetBot) != 0) return -1;

    } else if (method == "colortexture") {
        if (rgChromHist2D(targetImg, bins2d, targetColor) != 0) return -1;
        if (sobelMagHist1D(targetImg, binsTex, targetTex) != 0) return -1;

    } else {
        std::cout << "unknown method: " << method << "\n";
        return -1;
    }

    DIR *dirp = opendir(dirname);
    if (dirp == NULL) {
        std::cout << "cannot open directory: " << dirname << "\n";
        return -1;
    }

    std::vector<std::pair<std::string, double>> results;
    struct dirent *dp;
    char buffer[1024];

    // Compare target to each database image
    while ((dp = readdir(dirp)) != NULL) {
        if (dp->d_name[0] == '.') continue;
        if (!isImageName(dp->d_name)) continue;

        strcpy(buffer, dirname);
        strcat(buffer, "/");
        strcat(buffer, dp->d_name);

        cv::Mat img = cv::imread(buffer, cv::IMREAD_UNCHANGED);
        if (img.empty()) continue;

        double dist = 1e18;

        if (method == "baseline") {
            std::vector<float> feat;
            if (baselineFeature(img, feat) != 0) continue;
            dist = ssdDistance(targetFeat, feat);

        } else if (method == "hist2d") {
            std::vector<float> feat;
            if (rgChromHist2D(img, bins2d, feat) != 0) continue;
            dist = 1.0 - histIntersection(targetFeat, feat);

        } else if (method == "multihist") {
            std::vector<float> topH, botH;
            int mid = img.rows / 2;

            if (rgbHist3DRegion(img, binsRGB, 0, mid, topH) != 0) continue;
            if (rgbHist3DRegion(img, binsRGB, mid, img.rows, botH) != 0) continue;

            dist = multiHistDistance(targetTop, targetBot, topH, botH);

        } else if (method == "colortexture") {
            std::vector<float> colorH, texH;

            if (rgChromHist2D(img, bins2d, colorH) != 0) continue;
            if (sobelMagHist1D(img, binsTex, texH) != 0) continue;

            dist = colorTextureDistance(targetColor, targetTex, colorH, texH);
        }

        results.emplace_back(buffer, dist);
    }

    closedir(dirp);

    // Sort by distance (ascending), tie-break by filename
    std::sort(results.begin(), results.end(),
              [](const auto &a, const auto &b) {
                  if (a.second < b.second) return true;
                  if (a.second > b.second) return false;
                  return a.first < b.first;
              });

    int n = std::min(topN, (int)results.size());
    std::cout << "Target: " << targetPath << "\n";
    for (int i = 0; i < n; i++) {
        std::cout << (i + 1) << ": " << results[i].first
                  << "  " << results[i].second << "\n";
    }

    return 0;
}