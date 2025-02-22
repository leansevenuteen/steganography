# steganography
Audio Video Steganography integrated Machine Learning

## Introduction
This project combines the use of classical steganography techniques for concealing data with machine learning (ML) models to detect hidden information within audio and video files. The implemented steganography techniques include Least Significant Bit, Phase Coding, and Spread Spectrum, and the project allows for embedding various data types, including text, images, and audio.

Recently, numerous studies in this field have found that ML models can easily detect data that is intricately concealed using steganography techniques. Therefore, this project also provides a method by which a machine learning model can detect concealed data in audio by utilizing Convolutional Neural Networks (CNNs).

## Dataset
We use the original dataset, Acted Emotional Speech Dynamic Database [(data link)](https://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/), and process it with the `prepare_data.py` script to initialize the training and evaluation data for the CNN model.

## Enviroment
We conduct experiments with the ML model in the [Kaggle Notebook](https://www.kaggle.com/) environment, utilizing a Tesla P100 GPU with 16GB, 15GB of CPU, and 20GB of memory.

## Usage
### Using classical steganography techniques:
If you want to learn how to embed and retrieve various data types from audio files, run the `Algorithm_Embed.py` and `Algorithm_Retrieve.py` scripts. Please note that the project currently only works with cover data in .wav file format.

### Using the CNN model for steganalysis in audio:
If you want to learn how the ML model detects audio steganalysis, you can refer to the `steganography.ipynb` script. However, before you run the `steganography.ipynb` script, make sure you have prepared the data for training the model using the `prepare_data.py` script.
