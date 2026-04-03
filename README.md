<h1 align="center">😴 DrowsyNet</h1>
<h3 align="center">Real-Time Driver Drowsiness Detection using Deep Learning</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow"/>
  <img src="https://img.shields.io/badge/Flask-Web%20App-green?style=for-the-badge&logo=flask"/>
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=for-the-badge&logo=opencv"/>
</p>

---

## 🚗 About The Project

Driver drowsiness is one of the leading causes of road accidents worldwide. **DrowsyNet** is a real-time drowsiness detection system that monitors a driver's eye state through a webcam and instantly triggers an alert when drowsiness is detected.

---

## 🧠 Models Used

| Model | Type | Role |
|-------|------|------|
| 🔷 VGG16 | Pretrained CNN | Transfer Learning |
| 🔶 ResNet50V2 | Pretrained CNN | Transfer Learning |
| 🟢 MobileNetV2 | Pretrained CNN | Transfer Learning |
| ⭐ DrowsyNet | Custom Built CNN | New Proposed Model |

---

## 📂 Dataset

- **CEW Dataset** — Closed Eyes in the Wild
- Binary Classification → `DROWSY (0)` and `NATURAL (1)`

---

## 📥 Model Downloads

> Model files exceed GitHub's size limit. Download them below and place inside the `models/` folder.

| Model | Download |
|-------|----------|
| DrowsyNet Best | [⬇️ Download](https://drive.google.com/file/d/1lTHpGmQQU4Iy_Cc7O5N9y9NBBXWCCZGv/view?usp=drive_link) |
| DrowsyNet Final | [⬇️ Download](https://drive.google.com/file/d/1pnelolXuKlQQftlX8LtGDL1QH39PtrNs/view?usp=drive_link) |
| MobileNetV2 Final | [⬇️ Download](https://drive.google.com/file/d/1eBLKSJCYbO3aqF23nf21O2rhTy3-kOAV/view?usp=drive_link) |
| MobileNetV2 Best | [⬇️ Download](https://drive.google.com/file/d/1KhCjVVZWvyi161Nn7PXI2pR2FUPKg9fp/view?usp=drive_link) |
| ResNet50V2 Final | [⬇️ Download](https://drive.google.com/file/d/1fByi8snxUoelBp5EnmFR1xIcIZkzL3xE/view?usp=drive_link) |
| ResNet50V2 Best | [⬇️ Download](https://drive.google.com/file/d/1j-clBSfwJy3WrBITHqVKV0lNCO4QQih8/view?usp=drive_link) |
| VGG16 Best | [⬇️ Download](https://drive.google.com/file/d/15DxTkReF70jc-ayIVXCQc35GL3D9mvoF/view?usp=drive_link) |
| VGG16 Final | [⬇️ Download](https://drive.google.com/file/d/1-tHamYoiAUB3BVVxDM7US9jzooTa3CXQ/view?usp=drive_link) |

---

## 🚀 How To Run
```bash
# Install dependencies
pip install flask opencv-python tensorflow numpy

# Run the app
python app.py
```

Then open your browser at **http://localhost:5000**

---

## 📁 Project Structure
```
DrowsinessV2/
├── app.py                  # Main Flask application
├── train.py                # Model training script
├── class_info.json         # Class label mapping
├── templates/
│   └── index.html          # Frontend UI
└── models/                 # Place downloaded models here
```

 
