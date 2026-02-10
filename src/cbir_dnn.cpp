/**
 * Name: Xinyue Xuan
 * Date: 2026-02-07
 * Purpose:
 *   Project 2 Task 5: Deep Network Embedding matching using pre-computed ResNet18 features
 *
 * Usage:
 *   ./cbir_dnn <target_filename> <feature_csv> <topN> <distance_metric>
 *   distance_metric: ssd | cosine
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

/**
 * Summary:
 *   Reads CSV file containing image features.
 *   Format: filename,f1,f2,...,f512
 *
 * Arguments:
 *   filename - CSV file path
 *   filenames - Output vector of image filenames
 *   features - Output 2D vector of feature vectors (each row is 512 floats)
 *
 * Return value:
 *   0 on success, negative on failure
 */
int read_feature_csv(const char *filename, 
                     std::vector<std::string> &filenames,
                     std::vector<std::vector<float>> &features) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        std::cerr << "Error: Cannot open feature file " << filename << std::endl;
        return -1;
    }

    filenames.clear();
    features.clear();

    char line[65536];  // Large buffer for long lines
    while (fgets(line, sizeof(line), fp)) {
        // Parse line: first token is filename, rest are feature values
        char imgname[512];
        std::vector<float> feat;

        char *token = strtok(line, ",");
        if (!token) continue;
        
        strcpy(imgname, token);

        // Read all feature values
        while ((token = strtok(NULL, ",")) != NULL) {
            feat.push_back(atof(token));
        }

        // Validate feature dimension (should be 512 for ResNet18)
        if (feat.size() != 512) {
            std::cerr << "Warning: " << imgname << " has " << feat.size() 
                      << " features (expected 512), skipping" << std::endl;
            continue;
        }

        filenames.push_back(std::string(imgname));
        features.push_back(feat);
    }

    fclose(fp);

    std::cout << "Loaded " << filenames.size() << " feature vectors from " 
              << filename << std::endl;

    return 0;
}

/**
 * Summary:
 *   Computes Sum of Squared Differences (SSD) between two feature vectors.
 *
 * Arguments:
 *   a - First feature vector
 *   b - Second feature vector
 *
 * Return value:
 *   SSD distance, or -1.0 on error
 */
double compute_ssd(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size() || a.empty()) {
        return -1.0;
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double diff = (double)a[i] - (double)b[i];
        sum += diff * diff;
    }

    return sum;
}

/**
 * Summary:
 *   Computes cosine distance between two feature vectors.
 *   Cosine distance = 1 - cos(theta) = 1 - (a·b)/(||a||*||b||)
 *
 * Arguments:
 *   a - First feature vector
 *   b - Second feature vector
 *
 * Return value:
 *   Cosine distance in [0, 2], or -1.0 on error
 */
double compute_cosine_distance(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size() || a.empty()) {
        return -1.0;
    }

    // Compute dot product and norms
    double dot_product = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        dot_product += (double)a[i] * (double)b[i];
        norm_a += (double)a[i] * (double)a[i];
        norm_b += (double)b[i] * (double)b[i];
    }

    norm_a = sqrt(norm_a);
    norm_b = sqrt(norm_b);

    // Avoid division by zero
    if (norm_a < 1e-10 || norm_b < 1e-10) {
        return 2.0;  // Maximum distance
    }

    // Cosine similarity = dot_product / (norm_a * norm_b)
    double cos_sim = dot_product / (norm_a * norm_b);

    // Clamp to [-1, 1] to avoid numerical errors
    if (cos_sim > 1.0) cos_sim = 1.0;
    if (cos_sim < -1.0) cos_sim = -1.0;

    // Cosine distance = 1 - cosine_similarity
    return 1.0 - cos_sim;
}

/**
 * Summary:
 *   Main entry point for DNN embedding-based image retrieval.
 *
 * Arguments:
 *   argv[1] - target image filename (just the filename, e.g., "pic.0893.jpg")
 *   argv[2] - CSV file containing pre-computed features
 *   argv[3] - number of top matches to return
 *   argv[4] - distance metric: "ssd" or "cosine"
 *
 * Return value:
 *   0 on success, -1 on failure
 */
int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] 
                  << " <target_filename> <feature_csv> <topN> <distance_metric>\n";
        std::cout << "  distance_metric: ssd | cosine\n";
        std::cout << "  Example: ./cbir_dnn pic.0893.jpg features.csv 5 cosine\n";
        return -1;
    }

    const char *target_filename = argv[1];
    const char *csv_path = argv[2];
    int topN = atoi(argv[3]);
    std::string distance_metric = argv[4];

    if (topN <= 0) {
        std::cerr << "Error: topN must be positive" << std::endl;
        return -1;
    }

    if (distance_metric != "ssd" && distance_metric != "cosine") {
        std::cerr << "Error: distance_metric must be 'ssd' or 'cosine'" << std::endl;
        return -1;
    }

    // Read feature database
    std::vector<std::string> filenames;
    std::vector<std::vector<float>> features;

    if (read_feature_csv(csv_path, filenames, features) != 0) {
        return -1;
    }

    // Find target image in database
    int target_idx = -1;
    for (size_t i = 0; i < filenames.size(); i++) {
        if (filenames[i] == target_filename) {
            target_idx = i;
            break;
        }
    }

    if (target_idx == -1) {
        std::cerr << "Error: Target image " << target_filename 
                  << " not found in feature database" << std::endl;
        return -1;
    }

    const std::vector<float> &target_feat = features[target_idx];

    // Compute distances to all images
    std::vector<std::pair<std::string, double>> results;

    for (size_t i = 0; i < filenames.size(); i++) {
        double dist;

        if (distance_metric == "ssd") {
            dist = compute_ssd(target_feat, features[i]);
        } else {  // cosine
            dist = compute_cosine_distance(target_feat, features[i]);
        }

        if (dist < 0.0) {
            std::cerr << "Warning: Distance computation failed for " 
                      << filenames[i] << std::endl;
            continue;
        }

        results.emplace_back(filenames[i], dist);
    }

    // Sort by distance (ascending)
    std::sort(results.begin(), results.end(),
              [](const auto &a, const auto &b) {
                  if (a.second < b.second) return true;
                  if (a.second > b.second) return false;
                  return a.first < b.first;
              });

    // Print results
    int n = std::min(topN, (int)results.size());
    std::cout << "\nTarget: " << target_filename << std::endl;
    std::cout << "Distance metric: " << distance_metric << std::endl;
    std::cout << "\nTop " << n << " matches:\n";
    
    for (int i = 0; i < n; i++) {
        std::cout << (i + 1) << ": " << results[i].first
                  << "  (distance: " << results[i].second << ")\n";
    }

    return 0;
}