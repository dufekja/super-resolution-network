
<!-- main header -->
<div align="center">
    <h1>Super Resolution Network</h1>
</div>

<!-- contents -->
<details>
    <summary>Table of Contents</summary>
    <ul>
    <li><a href="#about-the-project">About the project</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#getting-started">Getting started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#implementation-details">Implementation</a>
        <ul>
            <li><a href="#download-script">Download</a></li>
            <li><a href="#dataset-class">Dataset</a></li>
            <li><a href="#utils">Utils</a></li>
            <li><a href="#residual-convolution-block">Residual convolution block</a></li>
            <li><a href="#subpixel-block">Subpixel block</a></li>
            <li><a href="#sresnet">Sresnet</a></li>
        </ul>
    </li>
    <li><a href="#evaluation">Evaluation</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#refereces">References</a></li>
    </ul>
</details>

## About The Project

This project aims to create and train a specialized `Super Resolution Neural Network` designed for image upscaling. The process involves several stages, such as downloading image data, building the essential components of the neural network, conducting model training, and evaluating performance.

## Roadmap

The project primarily involves the following key tasks:
- DIV2K download script
- README
- utility functions
- models
- evaluation

<details>
    <summary>Detailed roadmap</summary>

- [x] download script
    - [x] DIV2K single use download
- [x] readme
    - [x] getting started
    - [x] usage
    - [x] implementation
        - [x] download script
        - [x] dataset class
        - [x] utils
        - [x] residual block
        - [x] subpixel block
        - [x] sresnet
- [x] custom dataset loader
- [x] utils
    - [x] image format converter function
    - [x] image training and validation transformer
- [x] SRESNN
    - [x] training script
    - [x] residual convolution blocks
    - [x] subpixel blocks
- [ ] evaluation
- [ ] GAN
    - [ ] transfer learning
    - [ ] generator model
    - [ ] discriminator model
</details>

## Getting Started

This section provides a comprehensive guide for setting up the environment and prerequisites.

### Prerequisities

#### Ubuntu
 
To run the project code and train or use the sresnet model, you need to install `python3` and `pip`. Use the following commands:

Install python and pip using:
```
sudo apt update
sudo apt install python3
sudo apt install python3-pip
```

Verify your installation:
```
python3 --version
pip3 --version
```

The last prerequisite is to have the `venv` Python package installed:
```
python3 -m pip install venv
```

### Installation

Now, let's create a Python virtual environment (venv) and install all packages from `requirements.txt`.

Create and activate the venv:
```
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:
```
pip install --upgrade pip
pip install -r requirements.txt
```

To deactivate virtual environment, run:
```
deactivate
```

## Usage

This section explains how to use pretrained models for upscaling your own images and how to train sresnet on custom data with a custom configuration.

### Upscaling

You can find an example of image upscaling in superres.ipynb. Alternatively, you can load a trained model file using the following code:
```
state = torch.load('sresnet.pt')
model = state['model'].to(DEVICE)
```

Upscale your low-resolution image (default PIL format) using the `upscale_img` utility function:

```
sr = upscale_img(image, model)
```
Note: The low-resolution image size cannot exceed 1300 in sum due to computational limitations.


### Training your own model

#### Data

Start by downloading data. You can use any high-resolution image data or use the download script for DIV2K images. Run the following command with the virtual environment activated:
```
python3 download.py
```

#### Training

Use the `train_sresnet.py` script for training. Before running the script, specify important parameters:

- `SCALE`: Set your preferred upscale factor.
- `TRAIN_EPOCHS`: Number of epochs.
- `DATA_DIR`: Folder with image data.
- `PT_SAVED`: Trained model file (default is sresnet.pt).

Feel free to tweak other parameters, then run:
```
python3 train_sresnet.py
```

This trains the super resolution model for the specified number of epochs and saves its parameters into the `PT_SAVED` file.

## Implementation details

### Download script

The download script serves as a user-friendly tool for effortlessly obtaining `DIV2K` image data and transforming it into a format compatible with network data loaders.

The script is divided into two key components. Firstly, there's the script itself, which handles the downloading, unzipping, and organization of both training and validation data into the DIV2K folder. The second component is the `Downloader` class, designed to display download progress during its usage.

### Dataset class

This class is inherited from PyTorch's Dataset and is designed for custom data loading from a specified folder. It includes an optional parameter, `ImgTransformer`, which is a class for image conversion specifically created for use with the sresnet model.

### Utils

"The `utils.py` module houses functions and classes dedicated to manipulating image formats. The primary class within this module is `ImgTransformer`, which plays a key role in transforming images within the data loader. It offers two modes: train and validation."

#### ImgTransformer

This function takes an image as input and provides both a cropped high-resolution image and its low-resolution counterpart. The mode of operation determines whether it returns a crop with the maximum scale divisible size or a crop with a size specified in the training script's configuration file.

The low-resolution image is generated by downscaling the high-resolution image using the `BICUBIC` method.

#### convert_image

Convert image function supports 4 converisons.
- `pil`: Default RGB pil image format.
- `[0, 255]`: RGB image tensor.
- `[0, 1]`: Scaled tensor.
- `[-1, 1]`: Scaled tensor suitable for the tanh activation function.

### Residual convolution block

#### Explanation

Deeper neural networks are challenging to train due to the increased number of layers and issues such as vanishing/exploding gradients. This is where residual blocks with residual learning come into play.

In a standard layer, there is an input $x$ and our layer $F(x)$. Typically, we aim to find the function to obtain $y$. Finding this function is equivalent to determining weights and biases for the vector $x$. This is the basic operation of a standard layer.

Now, let's consider the residual layer. This time, our desired output is $F(x) + x$. (we simply add the input vector to our layer output). What this truly implies is that our function doesn't learn a direct mapping from $x$ to $y$, but rather it learns what to subtract or add from the given input vector.

According to the `Deep Residual Learning` paper, it is easier to find this residual mapping compared to finding a direct input transformation. Consequently, deeper networks can be trained more effectively.

![residual layer](images/residual-layer.png)

#### Code

Residual convolution blocks are implemented as class inherited from `nn.Module`. It consists of two convolutional layers with `prelu` as activation function for the first one.

Output of this layer is exaclty $F(x) + x$.

### Subpixel block

#### Explanation

Suppose we aim to design a layer capable of upscaling a given vector by a scale $s$. To achieve this upscaling, precisely $s^2$ channels are required, which can later be consolidated into a single channel to enhance the image resolution (as illustrated in the image).

Our subpixel block is composed of a convolution block that generates $s^2$ output channels. Subsequently, these channels are passed through a pixel shuffle operation, which rearranges them to enhance image resolution..

![subpixel block](images/subpixel-block.png)

#### Code

Subpixel block is implemented as class inherited from `nn.Module`. It utilizes `nn.Conv2d` -> `nn.PixelShuffle` -> `nn.PReLU`. 

### SResNet

The SResNet model consists of three main convolutional layers, residual blocks and subpixel blocks. Due to this structure, it can learn important relationships between low-resolution and high-resolution images.

This model is explained more in [this](https://arxiv.org/pdf/1501.00092.pdf) science paper.

### Transfer learning

Transfer learning involves reusing a pre-trained model that is then further trained for a specific purpose. In this instance, our primary model is `sresnet`, which aims to minimize `Mean Squared Error (MSE) loss`. The weights of this trained model can be utilized and fine-tuned, for instance, through the use of `Generative Adversarial Networks` (GANs). This approach significantly reduces the time required for training a new super-resolution model from scratch.

### GAN
todo

## Evaluation
todo

## License

Distributed under the MIT license. Visit `LICENSE` for more information.

## References

### Data
- [DIV2K image dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

### Tutorials and documentation
- [Pytorch documentation](https://pytorch.org/docs/stable/index.html)
- [Super resolution tutorial](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution)
- [GAN in super resolution](https://jonathan-hui.medium.com/gan-super-resolution-gan-srgan-b471da7270ec)

### Science papers
- [Deep residual learning](https://arxiv.org/pdf/1512.03385.pdf)
- [SRESNN](https://arxiv.org/pdf/1501.00092.pdf)
- [Image quality assessment using SSIM](https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf)
- [Loss functions in SRESNN](https://arxiv.org/pdf/1511.08861.pdf)

