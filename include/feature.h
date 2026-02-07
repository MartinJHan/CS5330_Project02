/**
 * Name: Junyao Han
 * Date: 2026-02-06
 * Purpose:
 *   Declares feature extraction and distance functions for Project 2 CBIR.
 *   These functions are used by the CBIR pipeline to compute image features
 *   and compare them using different similarity metrics.
 *
 *   Implemented tasks:
 *   - Task 1: Baseline feature (center 7x7 patch) and SSD distance
 *   - Task 2: Whole-image 2D color histogram and histogram intersection
 *   - Task 3: Multi-histogram matching (top/bottom RGB histograms)
 *   - Task 4: Color + texture matching (color histogram + Sobel magnitude histogram)
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * Summary:
 *   Computes the baseline feature using the center 7x7 pixel patch (BGR flattened).
 *
 * Arguments:
 *   img  - Input image.
 *   feat - Output feature vector.
 *
 * Return value:
 *   0 on success, negative on failure.
 */
int baselineFeature(const cv::Mat &img, std::vector<float> &feat);

/**
 * Summary:
 *   Computes SSD distance between two feature vectors.
 *
 * Arguments:
 *   a - First feature vector.
 *   b - Second feature vector.
 *
 * Return value:
 *   SSD distance, or a large value if incompatible.
 */
double ssdDistance(const std::vector<float> &a, const std::vector<float> &b);

/**
 * Summary:
 *   Computes a 2D rg chromaticity histogram for the whole image.
 *   The histogram is normalized so all bins sum to 1.
 *
 * Arguments:
 *   img   - Input image.
 *   bins  - Number of bins for r and g (e.g., 16).
 *   hist  - Output histogram flattened to bins*bins entries.
 *
 * Return value:
 *   0 on success, negative on failure.
 */
int rgChromHist2D(const cv::Mat &img, int bins, std::vector<float> &hist);

/**
 * Summary:
 *   Computes histogram intersection between two normalized histograms.
 *
 * Arguments:
 *   h1 - First histogram (normalized).
 *   h2 - Second histogram (normalized).
 *
 * Return value:
 *   Intersection score in [0, 1] (larger means more similar).
 */
double histIntersection(const std::vector<float> &h1, const std::vector<float> &h2);

/**
 * Summary:
 *   Computes a normalized 3D RGB histogram over a specified vertical region
 *   of the image [yStart, yEnd).
 *
 * Arguments:
 *   img    - Input image.
 *   bins   - Number of bins per channel (e.g., 8).
 *   yStart - Start row (inclusive).
 *   yEnd   - End row (exclusive).
 *   hist   - Output histogram flattened to bins*bins*bins entries.
 *
 * Return value:
 *   0 on success, negative on failure.
 */
int rgbHist3DRegion(const cv::Mat &img, int bins, int yStart, int yEnd, std::vector<float> &hist);

/**
 * Summary:
 *   Computes a multi-histogram distance using two RGB histograms:
 *   top half and bottom half. Each region is compared using histogram
 *   intersection and combined with equal weights.
 *
 * Arguments:
 *   hTopT - Target top-half histogram (normalized).
 *   hBotT - Target bottom-half histogram (normalized).
 *   hTopI - Image top-half histogram (normalized).
 *   hBotI - Image bottom-half histogram (normalized).
 *
 * Return value:
 *   Distance in [0, 1] (smaller means more similar).
 */
double multiHistDistance(const std::vector<float> &hTopT,
                         const std::vector<float> &hBotT,
                         const std::vector<float> &hTopI,
                         const std::vector<float> &hBotI);

/**
 * Summary:
 *   Computes a normalized 1D histogram of Sobel gradient magnitudes for the whole image.
 *
 * Arguments:
 *   img   - Input image.
 *   bins  - Number of magnitude bins (e.g., 16).
 *   hist  - Output histogram (size=bins), normalized to sum to 1.
 *
 * Return value:
 *   0 on success, negative on failure.
 */
int sobelMagHist1D(const cv::Mat &img, int bins, std::vector<float> &hist);

/**
 * Summary:
 *   Computes an equal-weight combined distance using color and texture histograms.
 *
 * Arguments:
 *   colorT - Target color histogram (normalized).
 *   texT   - Target texture histogram (normalized).
 *   colorI - Image color histogram (normalized).
 *   texI   - Image texture histogram (normalized).
 *
 * Return value:
 *   Distance in [0, 1] (smaller means more similar).
 */
double colorTextureDistance(const std::vector<float> &colorT,
                            const std::vector<float> &texT,
                            const std::vector<float> &colorI,
                            const std::vector<float> &texI);