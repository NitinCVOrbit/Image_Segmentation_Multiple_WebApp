# 🧠 Image Segmentation App using Pretrained Models

This repository contains a **Streamlit-based web app** for performing **semantic segmentation** using multiple pretrained deep learning models. It supports various domains such as **medical imaging, vehicles, roads, agriculture, and human segmentation**.

---

## 🚀 Features

- ✅ Interactive interface with **Streamlit**
- ✅ Multiple segmentation models:
  - `FCN-ResNet50`
  - `DeepLabV3-ResNet50`
  - `U-Net with ResNet34`
- ✅ Supports these categories:
  - Blood Cells
  - Cars (with parts)
  - Road Cracks
  - Leaf Disease
  - People
  - Potholes
- ✅ Color-coded segmentation mask overlay
- ✅ Beautiful background styling for UI

---

## 🧠 Model & Class Summary

| Model         | Classes                                | Architecture   |
|---------------|-----------------------------------------|----------------|
| Blood Cell    | bg, Blood Cell                          | FCN-ResNet50   |
| Car           | bg, Car, Wheel, Lights, Window          | FCN-ResNet50   |
| Cracks        | bg, Cracks                              | DeepLabV3      |
| Leaf Disease  | bg, Leaf Disease                        | U-Net          |
| Person        | bg, Person                              | DeepLabV3      |
| Pothole       | bg, Pothole                             | U-Net          |

---

## 📥 Download Pretrained Weights

Download all the pretrained `.pth` files and place them in the `weights/` folder.  [Download](https://drive.google.com/drive/folders/1WzQKkPYrQSfiyuT0s7hogyfiE4Z2Up3A?usp=sharing)


