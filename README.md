# steganography
Audio Video Steganography integrated Machine Learning

## Introduction
This project combines the use of classical steganography techniques for concealing data with machine learning (ML) models to detect hidden information within audio and video files. The implemented steganography techniques include Least Significant Bit, Phase Coding, and Spread Spectrum, and the project allows for embedding various data types, including text, images, and audio.

Recently, numerous studies in this field have found that ML models can easily detect data that is intricately concealed using steganography techniques. Therefore, this project also provides a method by which a machine learning model can detect concealed data in audio by utilizing Convolutional Neural Networks (CNNs).

## Dataset
We use the original dataset, Acted Emotional Speech Dynamic Database [AESDD](https://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/), and process it with the `prepare_data.py` script to initialize the training and evaluation data for the CNN model.

## Enviroment
We conduct experiments with the ML model in the [Kaggle Notebook environment](https://www.kaggle.com/), utilizing a Tesla P100 GPU with 16GB, 15GB of CPU, and 20GB of memory.

## Usage
##### Preparing data for the ML model:
