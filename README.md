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

## 📂 Project Structure
physiotherapy-assessment/
├── data/
├── docs/
├── notebooks/
├── results/ # plots & evaluation results
│ ├── timesformer_confusion_matrices_classification.png
│ ├── timesformer_training_history_classification.png
│ ├── videomae_multilabel_assessment_confusion_matrix.png
│ └── videomae_multilabel_assessment_training_history.png
├── src/
│ ├── assessment.py # VideoMAE multi-label exercise correctness
│ ├── classification.py # TimeSformer exercise classifier
│ ├── augmentation.py # dataset augmentation
│ └── models/ # trained weights (.pth) [ignored by Git]
├── tests/
├── README.md
├── requirements.txt
└── .gitignore