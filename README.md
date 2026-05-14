# Sickle Cell Disease Classification

Deep learning pipeline for classifying sickle cell disease from microscopy images, comparing object detection (YOLOv8, YOLO12) and image classification (Swin Transformer) approaches.

## Dataset

- **Source**: Microscopy images from Eastern Uganda ([Kaggle](https://www.kaggle.com/datasets/florencetushabe/sickle-cell-disease-dataset))
- **Classes**: Positive (sickle)=422 images | Negative (normal)=147 images
- **Split**: 60 / 20 / 20 (train / val / test), stratified

## Models

| Model | Task | Parameters | Accuracy | F1-Score | AUC-ROC |
|-------|------|-----------|----------|----------|---------|
| YOLOv8n | Detection | 3.0 M | 0.9231 | 0.9434 | 0.9615 |
| YOLO12n | Detection | 2.6 M | 0.7807 | 0.8252 | 0.9385 |
| SW-ViT (Swin-Base) | Classification | 86.7 M | 0.9825 | 0.9881 | 0.9980 |

## Google Colab Tutorials

Each tutorial is a self-contained notebook designed to run on Google Colab with a T4 GPU but for full training Tutorial 6 use A100 GPU.

| # | Tutorial | Description |
|---|----------|-------------|
| 1 | [Dataset](Tutorial_1_Dataset.ipynb) | Dataset setup, directory structure, data visualization, I/O specs, train/val/test splits |
| 2 | [Model Description](Tutorial_2_Model_Description.ipynb) | Model loading/saving, parameter counting, input/output layers, architecture details |
| 3 | [Model Optimization](Tutorial_3_Model_Optimization.ipynb) | Loss functions, optimizer selection, learning rate scheduling, train vs val loss analysis |
| 4 | [Basic Testing](Tutorial_4_Basic_Testing.ipynb) | Download pre-trained models, run inference on a single test example |
| 5 | [Basic Fine-tuning](Tutorial_5_Basic_Finetuning.ipynb) | Download pre-trained model, optimization loop, demonstrate loss reduction |
| 6 | [Full Training](Tutorial_6_Full_Training.ipynb) | Complete training pipeline, convergence analysis, data augmentation, 5-fold cross-validation |

## Project Structure

```
Project/
├── Tutorial_1_Dataset.ipynb
├── Tutorial_2_Model_Description.ipynb
├── Tutorial_3_Model_Optimization.ipynb
├── Tutorial_4_Basic_Testing.ipynb
├── Tutorial_5_Basic_Finetuning.ipynb
├── Tutorial_6_Full_Training.ipynb
├── sickle_classification_output.ipynb   # Full combined notebook
├── data/
│   ├── processed/labels/                # Extracted YOLO labels
│   └── yolo_dataset/                    # Train/val/test splits
├── outputs/
│   ├── results_report.md
│   ├── eda_class_distribution.png
│   ├── all_models_confusion_matrices.png
│   ├── comparative_results_table.png
│   ├── swvit_training_curves.png
│   ├── swvit_gradcam.png
│   ├── yolov8_training_curves.png
│   ├── yolo12_training_curves.png
│   └── ...
└── Sickle_cell_data_Set/                # Raw dataset
```

## Requirements

All dependencies are installed automatically in each Colab notebook. Key libraries:

- `ultralytics` (YOLOv8, YOLO12)
- `timm` (Swin Transformer)
- `torch`, `torchvision`
- `albumentations`
- `scikit-learn`
- `matplotlib`, `seaborn`
