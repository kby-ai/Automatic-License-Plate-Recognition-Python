<p align="center">
  <a href="https://play.google.com/store/apps/dev?id=7086930298279250852" target="_blank">
    <img alt="" src="https://github-production-user-asset-6210df.s3.amazonaws.com/125717930/246971879-8ce757c3-90dc-438d-807f-3f3d29ddc064.png" width=500/>
  </a>  
</p>

#### 📚 Product & Resources - [Here](https://github.com/kby-ai/Product)
#### 🛟 Help Center - [Here](https://docs.kby-ai.com)
#### 💼 KYC Verification Demo - [Here](https://github.com/kby-ai/KYC-Verification-Demo-Android)
#### 🙋‍♀️ Docker Hub - [Here](https://hub.docker.com/u/kbyai)

# Automatic-License-Plate-Recognition

## Overview

We implemented `ANPR/ALPR(Automatic Number/License Plate Recognition)` engine with unmatched accuracy and precision by applying `SOTA(State-of-the-art)` deep learning techniques in this repository. 
This repository demonstrates `ANPR/ALPR` model inference in `Linux` server.

KBY-AI's `LPR` solutions utilizes artificial intelligence and machine learning to greatly surpass legacy solutions. Now, in real-time, users can receive a vehicle's plate number.

The `ALPR` system consists of the following steps:
- Vehicle image capture
- Preprocessing
- Vehicle detection
- Number plate extraction
- Charater segmentation
- Optical Character Recognition(OCR) </br>

The `ALPR` system works in these strides, the initial step is the location of the vehicle and capturing a vehicle image of front or back perspective of the vehicle, the second step is the localization of Number Plate and then extraction of vehicle Number Plate is an image. The final stride uses image segmentation strategy, for the segmentation a few techniques neural network, mathematical morphology, color analysis and histogram analysis. Segmentation is for individual character recognition. Optical Character Recognition (OCR) is one of the strategies to perceive the every character with the assistance of database stored for separate alphanumeric character.

## Online Test Demo
To try KBY-AI ALPR Online Test, please visit [here](https://web.kby-ai.com/)

## Model Weights

To run this repository, model weights are needed.

- To request model weights, please contact us:</br>
🧙`Email:` contact@kby-ai.com</br>
🧙`Telegram:` [@kbyai](https://t.me/kbyai)</br>
🧙`WhatsApp:` [+19092802609](https://wa.me/+19092802609)</br>
🧙`Skype:` [live:.cid.66e2522354b1049b](https://join.skype.com/invite/OffY2r1NUFev)</br>
🧙`Facebook:` https://www.facebook.com/KBYAI</br>

## About Repository

### 1. Set up
1. Clone this repository to local or server machine.

2. Install python 3.9 or later version

3. Install dependencies using 'pip' command
```bash
pip install tensorflow
```
4. Run inference
```bash
python main.py
```
### 2. Performance Video

You can visit our YouTube video for ANPR/ALPR model's performance [here](https://www.youtube.com/watch?v=sLBYxgMdXlA) to see how well our demo app works.</br></br>
[![ANPR/ALPR Demo](https://img.youtube.com/vi/sLBYxgMdXlA/0.jpg)](https://www.youtube.com/watch?v=sLBYxgMdXlA)</br>

## Application of ALPR
`Automatic license-plate recognition (ALPR)` is a technology that uses `OCR(optical character recognition)` on images to read vehicle registration plates. It can use existing closed-circuit television, road-rule enforcement cameras, or cameras specifically designed for the task. ALPR can be used by police forces around the world for law enforcement purposes, including to check if a vehicle is registered or licensed. It is also used for electronic toll collection on pay-per-use roads and as a method of cataloguing the movements of traffic, for example by highways agencies.</br>
`ALPR` has many uses including:
- Recovering stolen cars
- Identifying drivers with an open warrant for arrest
- Catching speeders by comparing the average time it takes to get from stationary camera A to stationary camera B
- Determining what cars do and do not belong in a parking garage
- Expediting parking by eliminating the need for human confirmation of parking passes

