# NTUA ECE Image and Video Processing Course

This repository showcases the laboratory assignments, algorithmic implementations, and deep learning models I completed during the Image and Video Processing course at the National Technical University of Athens (NTUA), School of Electrical and Computer Engineering. Through this coursework, I built an applied foundation across classic digital image processing and modern vision paradigms, implementing multiresolution pyramid coding and entropy-constrained quantization from scratch, as well as designing, training, and benchmarking deep convolutional neural networks (LeNet, AlexNet, VGG) on CIFAR-100 using PyTorch.

---

## Laboratory Exercises & Assignments

### Assignment 1: Multiresolution Pyramid Coding & Image Quantization
*Implementation of the Burt & Adelson multiscale representation, entropy analysis, and lossy uniform quantization.*

* **Key Tasks:**
  * **Theory:** Analyzed "The Laplacian Pyramid as a Compact Image Code" (Burt & Adelson), investigating the influence of generating kernel parameter $a$ on Gaussian filtering, theoretical maximum entropy for 8-bit grayscale images ($8\text{ bits/pixel}$), quantization bin size tradeoffs, and multi-tier quantization across pyramid depths.
  * **Pyramid Algorithm Implementation:** Programmed custom 5-tap generating kernels (`GKernel`), downsampling reduction filters (`GREDUCE`), iterative Gaussian pyramids (`GPyramid`), upsampling expansion filters (`GEXPAND`), Laplacian bandpass difference pyramids (`LPyramid`), exact reconstruction decoders (`L_Pyramid_Decode`), and Laplacian quantization stages (`L_Quantization`).
  * **Experimental Evaluation:** Validated reconstruction fidelity on standard benchmarks (`Lena`, `camera`) across RGB and grayscale formats. Conducted parameter sweeps over $a \in [0.3, 0.7]$ and decomposition depths $(3–6)$, identified optimal $a$ yielding minimal first-order entropy at each octave, and evaluated rate-distortion effects under variable quantization bin widths.
* **Tech Stack:** Python, NumPy, SciPy, Matplotlib, scikit-image, Jupyter Notebook.

### Assignment 2: Deep Convolutional Architectures & CIFAR-100 Classification
*Comparative study of CNN paradigms and end-to-end deep learning pipeline design in PyTorch.*

* **Key Tasks:**
  * **Architectural Study:** Conducted an in-depth comparative architectural analysis across seminal convolutional networks: LeNet-5, AlexNet, and VGG-16/19. Tabulated layer topologies, receptive field filter sizes, activation nonlinearities (Sigmoid/Tanh vs. ReLU), parameter counts, pooling operations, and regularization strategies (Dropout, Data Augmentation).
  * **Data Pipeline & Partitioning:** Seeded reproducible CIFAR-100 data extraction (`team_seed = 229`), constructing tailored training, validation, and test splits with data preprocessing and augmentation transforms.
  * **Model Engineering & Training:** Built, trained, and evaluated deep CNN models in PyTorch. Explored convolutional feature extraction depth, batch normalization, learning rate scheduling, cross-entropy optimization, and regularization strategies to mitigate overfitting on fine-grained image classes.
  * **Performance Diagnostics:** Evaluated accuracy curves, loss convergence, confusion matrices, and per-class precision/recall metrics.
* **Tech Stack:** Python, PyTorch, Torchvision, NumPy, Matplotlib, Jupyter Notebook / Google Colab.

---

## Technologies Used

* **Languages:** Python (3.10+)
* **Frameworks & Libraries:** PyTorch, Torchvision, NumPy, SciPy, scikit-image, Matplotlib, Pandas
* **Environments:** Jupyter Notebook, Google Colab
* **Core Concepts:** Multiresolution Image Decomposition, Gaussian & Laplacian Pyramids, Image Entropy & Compression, Lossy Quantization, Convolutional Neural Networks (CNNs), Deep Learning, Image Classification
