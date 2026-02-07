# CS5330 Project 2 – Content-Based Image Retrieval (CBIR)

Name: Junyao Martin Han, Xinyue Sia Xuan   
Course: CS5330 – Pattern Recognition and Computer Vision  
Project: Project 2 – Image Matching and Retrieval

------------------------------------------------------------

## Overview

This project implements a content-based image retrieval (CBIR) system in C++ using OpenCV.
Given a target image and a database directory, the program computes image features and returns
the top N most similar images based on the selected matching method.

Implemented tasks:
- Task 1: Baseline matching (center 7x7 patch + SSD)
- Task 2: Whole-image 2D color histogram matching (rg chromaticity + intersection)
- Task 3: Multi-histogram matching (top/bottom RGB histograms + equal weighting)
- Task 4: Color + texture matching (color histogram + Sobel magnitude histogram)

------------------------------------------------------------

## Project Structure

    CS5330_Project02/
    ├── CMakeLists.txt
    ├── include/
    │   ├── feature.h
    │   └── csv_util.h
    ├── src/
    │   ├── cbir_baseline.cpp
    │   ├── csv_util.cpp
    │   └── readfiles.cpp
    ├── data/
    │   └── olympus/    (image dataset, not tracked in git)
    └── README.md

------------------------------------------------------------

## Build Instructions

This project uses CMake and OpenCV.

    mkdir build
    cd build
    cmake ..
    make

The executable will be created in the build directory (depending on your setup).

------------------------------------------------------------

## Run Instructions

General usage:

    ./cbir_baseline <target_image> <image_dir> <topN> <method>

Arguments:
- target_image: path to the query image (e.g., data/olympus/pic.0164.jpg)
- image_dir: directory containing the dataset images (e.g., data/olympus)
- topN: number of results to print
- method: matching method name (baseline, hist2d, multihist, colortexture)

Important:
- Make sure the working directory is set to the project root if using CLion,
  so relative paths like data/olympus/... resolve correctly.

------------------------------------------------------------

## Task 1 – Baseline Matching

Feature:
- Center 7x7 pixel patch (flattened BGR values)

Distance:
- Sum of Squared Differences (SSD)

Run:

    ./cbir_baseline data/olympus/pic.1016.jpg data/olympus 5 baseline

Expected behavior:
- The target image should appear as the top match with distance 0.

------------------------------------------------------------

## Task 2 – Histogram Matching (2D Color Histogram)

Feature:
- Whole-image 2D rg chromaticity histogram (16x16 bins), normalized

Distance:
- Histogram intersection, with distance defined as (1 - intersection)

Run (required query example):

    ./cbir_baseline data/olympus/pic.0164.jpg data/olympus 5 hist2d

------------------------------------------------------------

## Task 3 – Multi-Histogram Matching

Feature:
- Two 3D RGB histograms (top half and bottom half), 8 bins per channel, normalized

Distance:
- Histogram intersection for each region
- Final distance is the equal-weighted average of the two regional distances

Run (required query example):

    ./cbir_baseline data/olympus/pic.0274.jpg data/olympus 5 multihist

------------------------------------------------------------

## Task 4 – Texture and Color

Color feature:
- Whole-image 2D rg chromaticity histogram (normalized)

Texture feature:
- Whole-image Sobel gradient magnitude histogram (16 bins, normalized)

Distance:
- Equal-weighted combination of color distance and texture distance

Run (required query example):

    ./cbir_baseline data/olympus/pic.0535.jpg data/olympus 5 colortexture

------------------------------------------------------------

## Notes

- The dataset directory (data/olympus) is not included in this repository due to size.
- Feature extraction and matching functions are implemented in C++ as required.
- OpenCV is used for reading images and computing Sobel gradients (Task 4).

------------------------------------------------------------

## Acknowledgements

Starter code and project description provided by the course instructor.
OpenCV library is used for image processing utilities.