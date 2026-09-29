# Generative Adversarial Networks (GANs)

## 📌 Introduction

**Generative Adversarial Network (GAN)** is a type of **Deep Learning architecture** used to generate new data that looks similar to real training data.

GANs were introduced by **Ian Goodfellow and his colleagues in 2014**.

A GAN consists of two neural networks:

* **Generator (G)** – Generates fake/synthetic data.
* **Discriminator (D)** – Determines whether the input data is real or generated.

The two networks compete with each other during training.

---

## 🧠 Basic Architecture

```text
                  Random Noise (z)
                         │
                         ▼
                  ┌─────────────┐
                  │  Generator  │
                  │     (G)     │
                  └──────┬──────┘
                         │
                    Fake Data
                         │
                         ▼
                  ┌─────────────┐
Real Data ───────►│Discriminator│
                  │     (D)     │
                  └──────┬──────┘
                         │
                         ▼
                   Real / Fake
```

### Generator

The Generator takes a random vector called **latent noise** and converts it into synthetic data.

```text
Random Noise → Generator → Fake Image
```

### Discriminator

The Discriminator receives both real and fake samples and predicts whether the sample is real or fake.

```text
Real Image ──────┐
                 ├──→ Discriminator → Real / Fake
Fake Image ──────┘
```

---

# 🔄 How GAN Works

GAN training can be understood as a competition:

### Step 1 — Generator creates fake data

```text
Noise → Generator → Fake Image
```

### Step 2 — Discriminator sees real data

```text
Real Image → Discriminator → Real
```

### Step 3 — Discriminator sees fake data

```text
Fake Image → Discriminator → Fake
```

### Step 4 — Both networks improve

The Generator tries to create more realistic samples, while the Discriminator becomes better at detecting fake samples.

Eventually:

```text
Generator → Very Realistic Data
Discriminator → Difficult to distinguish Real vs Fake
```

---

# 📐 GAN Objective

The original GAN is based on a **minimax game** between Generator and Discriminator.

The objective is commonly written as:

```text
min_G max_D V(D,G)
```

where:

```text
V(D,G) =
E[log D(x)] + E[log(1 - D(G(z)))]
```

Where:

* `x` = real data
* `z` = random noise
* `G(z)` = generated data
* `D(x)` = probability that real data is real
* `D(G(z))` = probability that generated data is real

---

# 🏷️ Types of GANs

There are many GAN architectures designed for different applications.

## 1. Vanilla GAN

The **Vanilla GAN** is the original GAN architecture introduced in 2014.

```text
Noise
  ↓
Generator
  ↓
Fake Data
  ↓
Discriminator ← Real Data
  ↓
Real / Fake
```

### Applications

* Basic image generation
* Understanding GAN fundamentals
* Educational experiments

---

## 2. DCGAN

**DCGAN = Deep Convolutional GAN**

DCGAN replaces fully connected layers with **Convolutional Neural Networks (CNNs)**.

```text
Noise
  ↓
Transposed Convolution
  ↓
Conv Layers
  ↓
Generated Image
```

### Applications

* Image generation
* Face generation
* Object generation

DCGAN is one of the most commonly used GAN architectures for learning practical GAN implementation.

---

## 3. Conditional GAN (cGAN)

A **Conditional GAN** generates data according to a given condition or label.

For example:

```text
Noise + Label "7"
        ↓
    Generator
        ↓
   Image of 7
```

Instead of simply generating any digit, we can tell the GAN which digit we want.

### Applications

* Controlled image generation
* Digit generation
* Class-specific image generation

---

## 4. Pix2Pix

**Pix2Pix** performs **image-to-image translation**.

It learns to convert one type of image into another.

```text
Input Image
     ↓
 Generator
     ↓
Output Image
```

Examples:

```text
Sketch → Realistic Image
Edges → Photograph
Day → Night
```

Pix2Pix generally uses **paired training data**.

---

## 5. CycleGAN

**CycleGAN** performs image-to-image translation **without requiring paired images**.

```text
Domain A                  Domain B

Horse ──→ Generator ──→ Zebra
Horse ←── Generator ←── Zebra
```

The important idea is **cycle consistency**.

For example:

```text
Horse → Zebra → Horse
```

The final image should resemble the original horse image.

### Applications

* Horse ↔ Zebra
* Summer ↔ Winter
* Painting ↔ Photograph
* Day ↔ Night

---

## 6. WGAN

**WGAN = Wasserstein GAN**

WGAN was introduced to improve the **training stability** of GANs.

Traditional GANs use a discriminator, while WGAN uses a **critic**.

```text
Traditional GAN:

Generator → Discriminator → Probability


WGAN:

Generator → Critic → Wasserstein Distance
```

### Advantages

* More stable training
* Better learning behavior
* Reduced problems such as mode collapse

---

## 7. WGAN-GP

**WGAN-GP = Wasserstein GAN with Gradient Penalty**

WGAN-GP improves WGAN by adding a **gradient penalty**.

```text
WGAN
  +
Gradient Penalty
  ↓
WGAN-GP
```

It is widely used when stable GAN training is required.

---

## 8. Progressive GAN

**Progressive GAN** gradually increases the resolution of generated images during training.

For example:

```text
4 × 4
  ↓
8 × 8
  ↓
16 × 16
  ↓
32 × 32
  ↓
64 × 64
  ↓
128 × 128
  ↓
256 × 256
```

This makes it easier to train GANs for high-resolution image generation.

---

## 9. StyleGAN

**StyleGAN** was developed for high-quality image generation and introduced the concept of controlling image generation through different latent/style representations.

```text
Latent Vector
     ↓
Mapping Network
     ↓
Style Information
     ↓
Generator
     ↓
High-Quality Image
```

StyleGAN is particularly famous for generating realistic human faces.

### Applications

* Face generation
* Image synthesis
* Style manipulation
* Image editing

---

## 10. StyleGAN2

**StyleGAN2** improves upon the original StyleGAN.

It addresses several image-quality artifacts and provides improved image synthesis.

### Applications

* High-quality face generation
* Image synthesis
* Style manipulation

---

## 11. StyleGAN3

**StyleGAN3** further improves the architecture and focuses on better handling of image transformations and aliasing.

### Applications

* High-quality image generation
* Animation
* Image manipulation

---

## 12. BigGAN

**BigGAN** is designed for generating **high-quality and diverse images at large scale**.

It uses large-scale training and class conditioning.

```text
Noise + Class Label
        ↓
      BigGAN
        ↓
 High-Quality Image
```

### Applications

* Large-scale image generation
* Class-conditional image synthesis

---

## 13. InfoGAN

**InfoGAN = Information Maximizing GAN**

InfoGAN attempts to learn meaningful and interpretable representations from the generated data.

For example, different latent variables may learn to control different properties of an image.

```text
Latent Code
    ↓
Generator
    ↓
Generated Image

Latent Code → Interpretable Features
```

### Applications

* Representation learning
* Unsupervised feature discovery
* Controllable generation

---

## 14. SRGAN

**SRGAN = Super-Resolution GAN**

SRGAN is designed to increase the resolution of an image.

```text
Low Resolution Image
        ↓
      SRGAN
        ↓
High Resolution Image
```

Example:

```text
256 × 256 → 1024 × 1024
```

### Applications

* Image super-resolution
* Improving image details
* Enhancing low-resolution images

---

## 15. ESRGAN

**ESRGAN = Enhanced Super-Resolution GAN**

ESRGAN improves the image super-resolution approach used by SRGAN.

```text
Low Resolution
      ↓
   ESRGAN
      ↓
High Resolution
```

It focuses on producing sharper and more realistic details.

---

# 📊 GAN Types at a Glance

| GAN             | Main Purpose                               |
| --------------- | ------------------------------------------ |
| Vanilla GAN     | Basic data generation                      |
| DCGAN           | Image generation using CNNs                |
| cGAN            | Conditional/controlled generation          |
| Pix2Pix         | Paired image-to-image translation          |
| CycleGAN        | Unpaired image-to-image translation        |
| WGAN            | Stable GAN training                        |
| WGAN-GP         | Improved WGAN stability                    |
| Progressive GAN | High-resolution image generation           |
| StyleGAN        | High-quality controllable image generation |
| StyleGAN2       | Improved StyleGAN                          |
| StyleGAN3       | Improved image transformation handling     |
| BigGAN          | Large-scale high-quality image generation  |
| InfoGAN         | Interpretable representation learning      |
| SRGAN           | Image super-resolution                     |
| ESRGAN          | Enhanced image super-resolution            |

---

# 🗂️ GAN Family

A simple way to organize GANs is:

```text
GAN
│
├── Basic GAN
│   └── Vanilla GAN
│
├── Image Generation
│   ├── DCGAN
│   ├── Progressive GAN
│   ├── StyleGAN
│   ├── StyleGAN2
│   ├── StyleGAN3
│   └── BigGAN
│
├── Conditional Generation
│   ├── cGAN
│   └── InfoGAN
│
├── Image Translation
│   ├── Pix2Pix
│   └── CycleGAN
│
├── Stable Training
│   ├── WGAN
│   └── WGAN-GP
│
└── Super Resolution
    ├── SRGAN
    └── ESRGAN
```

---

# ⚠️ Common GAN Problems

### 1. Mode Collapse

The Generator produces very similar samples instead of diverse samples.

```text
Noise 1 → Face A
Noise 2 → Face A
Noise 3 → Face A
Noise 4 → Face A
```

The Generator has effectively learned only a small part of the data distribution.

---

### 2. Training Instability

GANs can be difficult to train because two networks are learning simultaneously.

```text
Generator ↔ Discriminator
```

If one becomes much stronger than the other, training can become difficult.

---

### 3. Vanishing Gradients

If the Discriminator becomes too good, the Generator may receive very weak gradients and stop learning effectively.

---

# 🛠️ Technologies Commonly Used

GANs can be implemented using:

* Python
* TensorFlow
* Keras
* PyTorch
* NumPy
* Matplotlib
* OpenCV

---

# 🚀 Learning Path

A good order for learning GANs is:

```text
1. GAN Fundamentals
       ↓
2. Vanilla GAN
       ↓
3. DCGAN
       ↓
4. Conditional GAN
       ↓
5. WGAN
       ↓
6. WGAN-GP
       ↓
7. Pix2Pix
       ↓
8. CycleGAN
       ↓
9. Progressive GAN
       ↓
10. StyleGAN
       ↓
11. StyleGAN2 / StyleGAN3
       ↓
12. BigGAN
       ↓
13. SRGAN / ESRGAN
```

---

# 💡 GAN Applications

GANs can be used for:

* 🖼️ Image generation
* 👤 Face generation
* 🎨 Image-to-image translation
* 🔍 Super-resolution
* 🏞️ Synthetic image generation
* 🧑‍🎨 Image editing
* 🎮 Game asset generation
* 🧪 Synthetic data generation
* 🖌️ Artistic image generation

---

# 📚 Key Terms

| Term          | Meaning                                               |
| ------------- | ----------------------------------------------------- |
| Generator     | Creates synthetic data                                |
| Discriminator | Distinguishes real and fake data                      |
| Latent Vector | Random representation used as Generator input         |
| Noise         | Random input given to Generator                       |
| Real Data     | Data from the training dataset                        |
| Fake Data     | Data generated by Generator                           |
| Mode Collapse | Generator produces limited varieties                  |
| Epoch         | One complete pass through training data               |
| Latent Space  | Representation space from which samples are generated |

---

# 🎯 Summary

A **GAN consists of two competing neural networks**:

```text
Generator
    ↓
Creates Fake Data
    ↓
Discriminator
    ↓
Real or Fake?
```

The Generator tries to **fool the Discriminator**, while the Discriminator tries to **detect generated samples**.

Through this adversarial training process, the Generator learns to create increasingly realistic data.

> **GAN = Generator + Discriminator + Adversarial Training**

