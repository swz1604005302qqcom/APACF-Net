# Multimodal Face Anti-Spoofing with Adaptive Transformer and Collaborative Fusion

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg">
  <img src="https://img.shields.io/badge/PyTorch-1.12+-red.svg">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-green.svg">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
</p>

---

## Introduction

This project implements **Multimodal Face Anti-Spoofing with Adaptive Transformer and Collaborative Fusion**.  
It leverages **RGB, Depth, and IR modalities** to improve robustness against various spoofing attacks in complex real-world scenarios.

The framework consists of:

- **Adaptive Multimodal Transformer Extractor (AMTE)**
- **Expert Augmented Projection (EAP)**
- **Collaborative Fusion Module (CFM)**

to effectively capture both modality-specific and cross-modal discriminative features.

The project supports training and evaluation on public multi-modal face anti-spoofing datasets such as CASIA-SURF and CASIA-SURF CeFA.

---

## Features

- Supports RGB / Depth / IR multi-modal inputs  
- Adaptive Multimodal Transformer Extractor (AMTE) for modality-specific feature extraction  
- Expert Augmented Projection (EAP) for enhanced cross-modal representation  
- Collaborative Fusion Module (CFM) for effective multi-modal feature fusion  
- Flexible single-modal and multi-modal training  
- PyTorch implementation with multi-GPU support  
- Easy-to-use training and testing pipeline  

---

## Requirements

- Python >= 3.8  
- PyTorch >= 1.12  
- CUDA >= 11.3  

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
torch
torchvision
numpy
opencv-python
matplotlib
tqdm
einops
```

---

## Dataset Preparation

The project supports the following datasets:

- CASIA-SURF  
- CASIA-SURF CeFA  

Organize the dataset as follows:

```text
dataset/
├── CASIA_SURF/
│   ├── RGB/
│   ├── Depth/
│   └── IR/
```

Please modify the dataset path in the configuration file before training.

---

## Training

The main training script is:

```text
train-fusion.py
```

Run training with:

```bash
python train-fusion.py
```

For multi-GPU training:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train-fusion.py
```

---

## Testing

Run evaluation using pretrained weights:

```bash
python train-fusion.py --mode test --checkpoint path/to/checkpoint.pth
```

The output includes:

- Live / spoof classification result  
- Evaluation metrics  
- Intermediate feature outputs  

---

## Model Architecture

The core model implementation is located at:

```text
model/multivit.py
```

Main components include:

| Module | Description |
|---|---|
| AMTE | Adaptive Multimodal Transformer Extractor for modality-specific feature extraction |
| EAP | Expert Augmented Projection for enhanced cross-modal representation |
| CFM | Collaborative Fusion Module for final multi-modal feature fusion |
| Classifier | Final spoof classification head |

Input size:

```text
48 × 48
```

---

## Project Structure

```text
APACF-Net/
├── model/
│   ├── multivit.py
│   └── __init__.py
├── train-fusion.py
├── dataset/
├── utils
├── README.md
└── docs/
```

---



## License

This project is released under the MIT License.

---

## Star History

If this project helps your research, please give it a ⭐ !
