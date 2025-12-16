# Comparison of Optimization Techniques for Deep Learning in Medical Image Classification

## Overview

This project evaluates the impact of several **model- and system-level optimization techniques** on deep learning performance in a real-world computer vision task: **COVID-19 detection from chest X-ray images**.

The primary objective is not to improve model accuracy, but to **quantify performance gains** (training and inference throughput) under different optimization strategies, with particular attention to **CPU-constrained and mixed CPU/GPU environments**.

---

## Problem Setting

Medical imaging models are often deployed in environments with:
- Limited compute resources
- High latency sensitivity
- Strict reliability requirements

Understanding how optimization techniques affect **data loading, forward pass execution, and end-to-end throughput** is therefore critical for practical deployment.

This study focuses on:
- Measuring performance tradeoffs
- Isolating the contribution of individual optimizations
- Evaluating their suitability for constrained runtime environments

---

## Dataset

The dataset consists of approximately **35,900 chest X-ray images** of patients with and without COVID-19 diagnoses.

Each sample includes:
- A chest X-ray image
- Associated metadata (patient demographics, diagnosis annotation, outcomes)

For the purposes of this study, **only the diagnosis annotation** was used as the target label.

---

## Preprocessing Pipeline

Before training or inference, all images undergo the following preprocessing steps:

- Image smoothing
- Contrast normalization
- Spatial resizing

These steps were held constant across all experiments to ensure fair comparison between optimization strategies.

---

## Optimization Techniques Evaluated

The following optimizations were evaluated independently and in combination:

- **DataLoader Optimization**
  - Memory pinning and optimized batching
- **TorchScript Tracing**
  - Conversion of models to traced representations for faster execution
- **cuDNN Autotuner**
  - Automatic selection of optimal convolution algorithms on supported GPUs

Each optimization was profiled to measure its effect on:
- Data loading time
- Forward pass execution
- Backward pass execution (training only)

---

## Experimental Framework

The task evaluated is binary classification of chest X-ray images to detect the presence of COVID-19.

Performance was measured separately for:
- **Training**
- **Inference**

For both settings, timing was collected across:
- CPU data loading
- GPU data loading
- CPU forward pass
- GPU forward pass

Each optimization was enabled independently and then combined to assess cumulative impact.

---

## Results

### Training Performance Profile

| Model Configuration | CPU Dataloading | CPU Forward | GPU Dataloading | GPU Forward |
|--------------------|-----------------|-------------|-----------------|-------------|
| Baseline           | —               | —           | —               | —           |
| Optimized          | ↑ 11.23%        | —           | ↑ 18.9%         | ↑ 5.7%     |

### Inference Performance Profile

| Model Configuration | CPU Dataloading | CPU Forward | GPU Dataloading | GPU Forward |
|--------------------|-----------------|-------------|-----------------|-------------|
| Baseline           | —               | —           | —               | —           |
| Optimized          | ↑ 52.0%         | —           | ↑ 35.0%         | ↑ 27.5%    |

---

## Key Observations

- DataLoader optimizations provided the **largest gains** in CPU-bound scenarios.
- cuDNN autotuning contributed consistent improvements in GPU forward execution.
- TorchScript tracing improved inference throughput more significantly than training throughput.
- Performance gains were **non-uniform across pipeline stages**, reinforcing the importance of fine-grained profiling.

---

## Challenges and Limitations

- **Training Time:** Large model sizes and dataset scale resulted in long experiment runtimes.
- **Optimization Tradeoffs:** Selecting appropriate optimization strategies required careful evaluation of stability, reproducibility, and hardware constraints.
- **Scope:** This work focuses on performance optimization and does not explore model accuracy tradeoffs in depth.

---

## Conclusion

This project demonstrates that **system-level optimizations can yield substantial performance improvements** in medical image classification workflows, particularly for inference-time deployment in constrained environments.

While no single optimization dominated across all settings, **combinations of lightweight optimizations** produced meaningful cumulative gains without altering model architecture or training objectives.

---

## Future Work

Potential extensions include:
- Evaluating optimization impact across additional medical imaging modalities
- Exploring CPU-only inference scenarios in greater depth
- Studying interactions between optimization strategies and model accuracy









