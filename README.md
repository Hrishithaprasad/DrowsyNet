# DrowsyNet — Real-Time Driver Drowsiness Detection

A real-time drowsiness detection system built using deep learning and computer vision. Monitors a driver's eye state via webcam and triggers an alert when drowsiness is detected.

---

## Models Used
| Model | Type |
|-------|------|
| VGG16 | Pretrained CNN (Transfer Learning) |
| ResNet50V2 | Pretrained CNN (Transfer Learning) |
| MobileNetV2 | Pretrained CNN (Transfer Learning) |
| DrowsyNet | Custom Built CNN |

---

## Dataset
- **CEW Dataset** — Closed Eyes in the Wild
- Classes: `DROWSY (0)` and `NATURAL (1)`

---

## Model Downloads
Model files are too large for GitHub. Download and place them in the `models/` folder.

| Model | Download |
|-------|----------|
| DrowsyNet Best | [Download](https://drive.google.com/file/d/1lTHpGmQQU4Iy_Cc7O5N9y9NBBXWCCZGv/view?usp=drive_link) |
| DrowsyNet Final | [Download](https://drive.google.com/file/d/1pnelolXuKlQQftlX8LtGDL1QH39PtrNs/view?usp=drive_link) |
| MobileNetV2 Final | [Download](https://drive.google.com/file/d/1eBLKSJCYbO3aqF23nf21O2rhTy3-kOAV/view?usp=drive_link) |
| MobileNetV2 Best | [Download](https://drive.google.com/file/d/1KhCjVVZWvyi161Nn7PXI2pR2FUPKg9fp/view?usp=drive_link) |
| ResNet50V2 Final | [Download](https://drive.google.com/file/d/1fByi8snxUoelBp5EnmFR1xIcIZkzL3xE/view?usp=drive_link) |
| ResNet50V2 Best | [Download](https://drive.google.com/file/d/1j-clBSfwJy3WrBITHqVKV0lNCO4QQih8/view?usp=drive_link) |
| VGG16 Best | [Download](https://drive.google.com/file/d/15DxTkReF70jc-ayIVXCQc35GL3D9mvoF/view?usp=drive_link) |
| VGG16 Final | [Download](https://drive.google.com/file/d/1-tHamYoiAUB3BVVxDM7US9jzooTa3CXQ/view?usp=drive_link) |

---

## How To Run
```bash
pip install flask opencv-python tensorflow numpy
python app.py
```
Open browser at `http://localhost:5000`

---
