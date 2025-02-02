# Variational Autoencoder (VAE) with Convolutional Layers

## Dillon McCarthy

This repository contains the implementation of a **Variational Autoencoder (VAE)** that incorporates **convolutional layers** in both the encoder and decoder networks. The VAE is trained on the **CIFAR-10** dataset and extended to include interpolation in the latent space to visualize the generative capabilities of the model. Additionally, the model is applied to the **CelebA** dataset to evaluate its generalization to real-world images.

## Project Overview

In this assignment, the following tasks are completed:

1. **Convolutional VAE Architecture**  
   The VAE is extended to include **convolutional layers** in both the encoder and decoder networks. Convolutional layers are well-suited for image data as they are capable of capturing spatial hierarchies, leading to better feature extraction and more efficient learning of latent representations.

2. **Latent Space Interpolation**  
   Interpolation is performed between two points in the latent space, corresponding to two different images from the CIFAR-10 dataset. By generating intermediate points, the smoothness of transitions between two images is visualized, providing insights into how the VAE has structured the latent space.

3. **Generative Application to CelebA Dataset**  
   The VAE model is applied to the **CelebA** dataset, which contains images of celebrity faces. This task demonstrates the ability of the model to generalize to more complex and real-world datasets. The generated images are visualized to assess the model's performance and latent space structure for face images.

*Tasks were completed with limited availbility to high-compute platforms.  For real-world use cases, models would require much more training and hyperparameter fine-tuning.*

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib

You can install the necessary dependencies by running:
```bash
pip install -r requirements.txt
```

## Datasets
CIFAR-10
The CIFAR-10 dataset is used for training the VAE in its initial form. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

CelebA
The CelebA dataset is a large-scale dataset of celebrity face images used to test the VAE's generalization to real-world data.  It consists of 162,770 178x218 color images; however, for illustrating the concepts of VAEs we will resize these to 64x64 square images.

You can download the CIFAR-10 dataset via torchvision, and CelebA can be downloaded directly through its official site or through the torchvision.datasets.CelebA interface.

## Key Conceptual Questions
**Q1: Why is the KL Divergence term important in the VAE loss function?**

KL divergence measures the closeness of the approximation between two distributions. This is imperative for a loss function when creating a model that can generate realistic data because the model does so by attempting to recreate the true distribution of the data. By evaluating the differences in the true and predicted distributions, the model can learn to minimize the KL divergence over time.

**Q2: How does the reparameterization trick enable backpropagation through the stochastic layers of a VAE?**

When the target of a model is to learn the probabilistic latent distribution such as 𝒩~(μ, σ^2), a model will attempt to replicate this distribution and backpropogate using the gradients from the loss function. If the target is a distribution though, the function is non-differentiable because a distribution has "randomness" to it and is dependent on μ and σ, so the gradient can not be defined. By introducing some noise ϵ sampled from a normal distribution, the gradients of μ and σ can be computed in terms of ϵ making them differentiable. This allows the model to backpropogate and minimize the loss as it models the true distribution.

**Q3: Why does a VAE use a probabilistic latent space instead of a fixed latent space?**

If a fixed latent space were used, then the VAE would be a one-to-one model, meaning for each input there is a single output. For instance, if the VAE were trained to generate images from text, the input "dog" would generate the same photo of a dog every time. Instead, by training the VAE to map inputs to the probabilistic latent space, it gives the ability to then use this synthetic distribution and sample from it, creating a slightly different output every time. Therefore, with this example, the input "dog" would then produce a unique dog image every time from the learned distribution.

**Q4: What role does KL Divergence play in ensuring a smooth latent space?**

KL divergence ensures a smooth latent space by normalizing the distribution across the known prior. When introducing noise to the output through the reparameterization trick, a known distribution, often the normal distribution, is sampled. Then the normal distribution can be used to calculate a close-form solution for KL divergence and confines the output into a known, smooth latent distribution. This helps prevent discontinuity and fragmentation in the latent space and controls the predictablility of the training process.
