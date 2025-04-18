# Chest CT Scan Classification Project

This project implements deep learning models for classifying chest CT scan images into four categories:

- Adenocarcinoma
- Large cell carcinoma
- Squamous cell carcinoma
- Normal

## Models Implemented

1. **Basic CNN Model**
   - Custom CNN architecture with batch normalization and dropout
   - Data augmentation for improved generalization

2. **VGG16 Transfer Learning Model**
   - Pre-trained VGG16 base with custom classification layers
   - Two-phase training: frozen layers and fine-tuning
   - Uses ImageNet weights for transfer learning

## Setup

Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Models

The main script provides several options for training:

To train both models with default parameters:

   ```bash
   python main.py
   ```


### Additional Parameters

- `--batch-size`: Set batch size (default: 32)
- `--epochs`: Number of epochs for CNN training (default: 30)
- `--initial-epochs`: Initial epochs for VGG16 training (default: 15)
- `--fine-tune-epochs`: Fine-tuning epochs for VGG16 (default: 15)

Example with custom parameters:

```bash
python main.py --model both --batch-size 64 --epochs 50 --initial-epochs 20 --fine-tune-epochs 20
```

## Model Outputs

- Training history plots (`.png` files)
- Classification reports and accuracy metrics

## Dataset

The project uses the "Chest CT-Scan Images" dataset from Kaggle, which will be automatically downloaded using `kagglehub` when running the models.
