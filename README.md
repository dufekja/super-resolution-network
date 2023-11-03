
<!-- main header -->
<div align="center">
    Super Resolution Network
</div>

<!-- contents -->
<details>
    <summary>Table of Contents</summary>
    <li><a href="#about-the-project">About the project</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#refereces">References</a></li>
</details>

## About The Project

Aim of this project is to create and train custom SRESNN for image upscaling.

## Roadmap

- [] download script
    - [] DIV2K single use download
    - [] custom url dataset selection
- [] readme
    - [] getting started
    - [] usage
    - [] implementation
- [x] custom dataset loader
- [] utils
    - [x] image format converter function
    - [x] image training and validation transformer
- [x] SRESNN
    - [x] training script
    - [x] residual convolution blocks
    - [x] subpixel blocks
- [] GAN
    - transfer learning
    - generator model
    - discriminator model
- [] evaluation


## Getting Started

In this section is complete guide with environment and prerequisities setup.

### Prerequisities

todo

### Installation

todo

## Usage

todo

## Implementation details

### Residual convolution blocks
todo

### Subpixel blocks
todo

### Transfer learning
todo

### GAN
todo

## License

Distributed under the MIT license. Visit `LICENSE` for more information.

## References

### Data
- [DIV2K image dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

### Tutorials
- [Super resolution tutorial](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution#overview)
- [GAN in super resolution](https://jonathan-hui.medium.com/gan-super-resolution-gan-srgan-b471da7270ec)

### Science papers
- [SRESNN](https://arxiv.org/pdf/1501.00092.pdf)
- [Image quality assessment using SSIM](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf)
- [Loss functions in SRESNN](https://arxiv.org/pdf/1511.08861.pdf)

