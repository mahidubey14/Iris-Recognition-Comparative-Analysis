# Comparative Analysis of Classical and Deep Learning-Based Iris Recognition Techniques: LBPH, Daugman, and VGG16 Approaches

## Overview
This project presents a comparative study of iris recognition methods using three distinct approaches:
- LBPH (Local Binary Pattern Histogram)
- Daugman’s Traditional Iris Recognition Algorithm
- VGG16 (Deep Learning-based CNN model)

The goal is to analyze and compare their accuracy, feature extraction efficiency, and computational performance using a dataset of 20 individuals.

## Objectives
- Implement three iris recognition methods: LBPH, Daugman, and VGG16.
- Compare classical and deep learning-based techniques.
- Evaluate accuracy, processing time, and feature quality.
- Identify the most effective method for small and large-scale iris datasets.

## Methodology

### Dataset
- Dataset contains iris images of 40+ individuals.
- Each folder represents a person with multiple labeled iris images.
- The dataset is preprocessed (grayscale conversion, normalization, ROI extraction).

### Techniques Used

| Method | Type | Description |
|--------|------|-------------|
| LBPH | Classical | Extracts texture features using local binary patterns. |
| Daugman | Classical | Uses Gabor filters and rubber sheet model for iris encoding. |
| VGG16 | Deep Learning | A CNN architecture fine-tuned for iris recognition. |

## Implementation

### LBPH
- Feature extraction using local binary pattern histograms.
- Classification based on histogram comparison.

### Daugman
- Segmentation using integro-differential operator.
- Normalization and encoding using Gabor wavelets.

### VGG16
- Transfer learning using pretrained VGG16 model.
- Feature extraction and classification using Softmax layer.

