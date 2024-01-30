# Example implementation of the K-NN algorithm

A simple program implementing the K-NN algorithm (Supervised Machine Learning) is used to predict clothing sizes based on weight, age, and height. The program consists of five size classes, namely S, M, L, XL, and XXL.

## Requirements
Install python (I'm using python 3.10)

## How to run
Clone the repository

    https://github.com/dzakyy04/tugas-gdsc.git

Switch to the repo folder

    cd tugas-git-gdsc/1-git-github

Install all required packages

    pip install -r requirements.txt

Run the generate_model.py file

    python generate_model.py

Perform prediction from terminal

    python predict.py

Perform prediction from API

    python app.py

The server will be running on port 5000, now you can try sending a request

Example of JSON format for sending a request

    {
        "weight": 87,
        "age": 19,
        "height": 180.8
    }

Example of prediction result response

    {
        "prediction": "XL"
    }