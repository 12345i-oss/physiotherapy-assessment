# 🏥 Physiotherapy Assessment with VideoMAE & TimeSformer

This repository contains a **deep learning framework for automated physiotherapy exercise assessment**.  
We leverage two transformer-based video models:

- **VideoMAE** → Multi-label assessment of **exercise correctness** (correct vs incorrect).  
- **TimeSformer** → Exercise **classification** (exercise type + side).  

---

## 🚀 Features
- **Data Augmentation** → Simulates real-world conditions (low light, jitter, shadows, motion blur, etc.).  
- **VideoMAE Assessment** → Binary prediction (correct/incorrect) per exercise type.  
- **TimeSformer Classification** → Recognizes exercise category and laterality (e.g., `Shoulder_Flexion_LEFT`).  
- **Modular Source Code** → Separated into augmentation, assessment, and classification scripts.  

---

