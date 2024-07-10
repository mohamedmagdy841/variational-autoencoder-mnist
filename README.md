# MNIST Digit Image Generation with Variational Autoencoders

## Overview

This project demonstrates the use of Variational Autoencoders (VAEs) to generate images of digits (0-9) from the MNIST dataset. The VAE is trained to learn the distribution of the digit images, allowing it to generate new, realistic images by sampling from this learned distribution.

## Features

- Trains a Variational Autoencoder on the MNIST dataset.
- Extracts latent space representations (`mu` and `sigma`) for each digit.
- Generates new images by sampling from the latent space.
- Allows generation of a specified number of images for a particular digit.


