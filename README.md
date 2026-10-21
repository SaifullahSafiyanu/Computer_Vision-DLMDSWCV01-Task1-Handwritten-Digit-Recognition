# Handwritten Digit Recognition: Historical vs. Modern CNN Architectures

## Overview
This repository contains the implementation, training, and evaluation of two Convolutional Neural Network (CNN) architectures for handwritten digit recognition using the MNIST dataset. The project was completed as part of the **DLMDSWCV01 - Project: Computer Vision** course at IU International University of Applied Sciences.

The primary objective of this project is to explore the historical foundations of computer vision by approximating the pioneering network architecture introduced by **Le Cun et al. (1989)** using modern Keras, and comparing its parameter efficiency and predictive performance against a modern baseline established by **Chollet (2021)**.

## Project Structure
- `Task1_LeCun_Approximation_Notebook.ipynb`: The fully executed Jupyter Notebook containing the data preparation, model implementations, training logs, and visualizations.
- `lecun_training_performance.png`: The exported visualization comparing the validation accuracy of both models over 15 epochs.

## Methodology
Two models were implemented and trained on the MNIST dataset (60k train / 10k test) for 15 epochs using a batch size of 128, Categorical Cross-Entropy, and the Adam optimizer.

1. **Modern Baseline (Chollet, 2021):**
   - Utilizes `Conv2D` layers with `ReLU` activations.
   - Spatial downsampling via `MaxPooling2D`.
   - Regularization via `Dropout (0.5)` before a dense softmax output.
   - **Total Parameters:** 34,826

2. **Historical Approximation (Le Cun et al., 1989):**
   - Utilizes `Conv2D` layers (5x5 filters) with `tanh` activations to mirror historical squashing functions.
   - Spatial downsampling via `AveragePooling2D`.
   - Substituting historically manually defined sparse connections (H2-H3) with a modern `Dropout (0.25)` layer for regularization.
   - **Total Parameters:** 3,246

## Results & Efficiency Trade-offs
| Metric | Chollet (2021) Baseline | Le Cun (1989) Approximation |
| :--- | :--- | :--- |
| **Total Parameters** | 34,826 | 3,246 |
| **Test Accuracy** | 99.14% | 98.06% |
| **Test Loss** | 0.0243 | 0.0598 |
| **First Epoch Time** | ~18 seconds | ~7 seconds |

The comparative analysis demonstrates that the historical approximation retains roughly 98.9% of the baseline's accuracy while utilizing approximately 90.7% fewer parameters, confirming the remarkable parameter efficiency of early structural priors in neural design.

## References
- Chollet, F. (2021). *Simple MNIST convnet*. Keras. https://keras.io/examples/vision/mnist_convnet/
- LeCun, Y., Boser, B., Denker, J., Henderson, D., Howard, R., Hubbard, W., & Jackel, L. (1989). Handwritten digit recognition with a back-propagation network. *Advances in neural information processing systems*, 2.

## Author
**Saifullah Safiyanu**  
IU International University of Applied Sciences
