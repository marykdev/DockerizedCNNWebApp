# Dockerized CNN Web App

## Overview
This project is a Dockerized web application that deploys two Convolutional Neural Network (CNN) models. One model is trained on the MNIST dataset to recognize handwritten digits,
while the other model analyzes Chest X-ray images to determine if they indicate a healthy or unhealthy condition.

## Features
- **MNIST Digit Recognition**: Users can upload images of handwritten digits, and the model will classify them.
- **Chest X-ray Analysis**: Users can upload Chest X-ray images, and the model will predict whether the image shows signs of health or illness.
- **Dockerized Deployment**: The entire application is packaged in a Docker container for easy deployment and scalability.

## Technologies Used
- Python
- Flask
- TensorFlow/Keras
- Docker
- HTML/CSS/JavaScript

## Getting Started

### Prerequisites
- Docker installed on your machine
- Docker Compose (if applicable)

### Clone the Repository
```bash
git clone https://github.com/marykdev/DockerizedCNNWebApp.git
cd DockerizedCNNWebApp
