# Veritas AI: Synthetic Image Forensics System

![Project Status](https://img.shields.io/badge/status-live-success)
![Tech Stack](https://img.shields.io/badge/stack-PyTorch_Streamlit_Python-blue)
![Accuracy](https://img.shields.io/badge/accuracy-98.2%25-green)

## Overview
Veritas AI is a deep learning-based forensic tool designed to detect Generative Adversarial Network (GAN) and Diffusion-based synthetic imagery. Unlike standard classifiers, Veritas combines **Convolutional Neural Networks (CNNs)** with traditional **Digital Forensics** techniques (Error Level Analysis & Metadata Scrapers) to provide an explainable verdict on image authenticity.

The system was trained on the **CIFAKE dataset** (60,000 images) and achieves **98.2% accuracy** in distinguishing real photographs from AI-generated artifacts (Stable Diffusion/Midjourney).

## Key Features

### 1. Deep Learning Classification
* **Architecture:** Custom ResNet-Lite CNN optimized for artifact detection in 32x32 pixel space.
* **Performance:** ~20ms inference time on CPU.
* **Robustness:** Trained to identify noise pattern inconsistencies rather than semantic content.

### 2. Forensic Workbench
* **Error Level Analysis (ELA):** visualizes compression artifacts. Real images have uniform compression; AI images often show irregular "ghosting" due to pixel generation differences.
* **Metadata extraction:** Automatically scans for EXIF data (ISO, Aperture, Date). AI generators typically strip this data, acting as a secondary heuristic for falsification.

## Tech Stack
* **Core Logic:** Python 3.10+
* **Machine Learning:** PyTorch (TorchVision)
* **Interface:** Streamlit (Custom CSS styled)
* **Image Processing:** PIL (Python Imaging Library), NumPy

## Running Locally

1. **Clone the repository**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/veritas-ai.git](https://github.com/YOUR_USERNAME/veritas-ai.git)
   cd veritas-ai