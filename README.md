# Pantanal Acoustic AI 🌿🎧

Deep learning project focused on wildlife species identification using passive acoustic monitoring data collected from the Pantanal wetlands. This project explores bioacoustic machine learning pipelines for understanding and analyzing complex environmental soundscapes.

## Dataset

BirdCLEF 2026 Competition Dataset:
https://www.kaggle.com/competitions/birdclef-2026/data

## Project Overview

This repository documents my learning journey from traditional machine learning into acoustic AI and bioacoustics. The project begins with training a baseline EfficientNet based acoustic classification model using mel spectrogram representations and will later expand into foundation audio models such as Perch and ensemble learning systems.

The goal is not only to build accurate wildlife sound classification models, but also to strengthen my understanding of:

* Audio signal processing
* Spectrogram based learning
* Deep learning for sound classification
* Foundation audio models
* Ensemble learning
* Acoustic inference pipelines
* Probability calibration and soundscape understanding

## Current Workflow

Raw Audio (.ogg)
→ Audio Chunking
→ Mel Spectrogram Generation
→ EfficientNet CNN
→ Species Probability Prediction

## Current Features

* Audio preprocessing using Librosa
* Mel spectrogram generation
* Wildlife species classification
* EfficientNet based CNN training
* GPU accelerated PyTorch training
* Fold based validation pipeline
* Acoustic inference experimentation

## Planned Future Work

### 1. Perch Foundation Audio Model

I plan to train and experiment with Google's Perch bioacoustic foundation model to better understand pretrained acoustic embeddings and transfer learning for environmental sound classification.

### 2. Ensemble Learning

Predictions from EfficientNet and Perch based models will later be combined using ensemble techniques to compare model behavior, improve robustness, and explore how different architectures interpret ecological soundscapes.

### 3. Advanced Audio Learning Concepts

Future exploration includes:

* Multi label audio classification
* Probability calibration
* Isotonic calibration
* Delta and delta-delta acoustic features
* Adaptive smoothing
* Attention based audio models
* Acoustic representation learning

## Technologies Used

* Python
* PyTorch
* Librosa
* NumPy
* Pandas
* Scikit-learn
* EfficientNet
* Google Colab
* Audio Signal Processing

## Learning Goals

This project is designed as both:

1. A wildlife acoustic AI system
2. A deep learning research and learning framework for understanding modern audio machine learning systems

The repository will continue evolving as I explore more advanced bioacoustic AI architectures and acoustic learning techniques.
