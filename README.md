# Face Detection and Classification Project

## Overview
This project implements a Graph Attention Network (GAT) based model for face image detection and classification using facial landmarks and SuperPoint features.

## Project Structure
```
├── adjacency.py            # Adjacency matrix calculation utilities
├── data_processing.py      # Dataset processing and loading
├── environment.yml         # Conda environment config
├── extract_descriptor.py   # Feature descriptor extraction
├── model.py               # Neural network model architecture
├── similarities.py        # Similarity metrics implementation
├── train.py              # Main training script 
├── dataset/              # Dataset directory
│   └── ADM_dataset/      # Contains train/valid/test splits
├── models/               # Saved model checkpoints
└── test/                # Test implementations
```

## Features
- Graph Attention Network (GAT) for face detection
- SuperPoint feature extraction
- Face landmark detection with dlib
- Adaptive feature filtering
- Multiple similarity metrics (SCM, cosine)
- ROC curve evaluation

## Dependencies
- TensorFlow 2.x
- PyTorch
- dlib
- OpenCV
- scikit-learn
- NumPy
- Pandas
- Matplotlib

## Installation
1. Clone the repository
2. Create conda environment:
```bash
conda env create -f environment.yml
```

## Dataset Structure
```
dataset/
└── ADM_dataset/
    ├── train/
    ├── valid/
    └── test/
```

## Training
To train the model:
```bash
python train.py
```

Key parameters:
- Learning rate: 1e-4 
- Batch size: 32
- Early stopping patience: 10
- Learning rate reduction factor: 0.5

## Model Architecture
- Input layer
- GAT layers with multi-head attention
- Batch normalization
- Global average pooling
- Dense layers
- Binary classification output

## Outputs
- submission.csv: Prediction results
- roc_curve.png: ROC curve visualization
- Model checkpoints in models directory

## Performance Metrics
- Binary Cross Entropy Loss
- Accuracy
- Area Under ROC Curve (AUC)

## Usage
1. Prepare dataset in required structure
2. Run training:
```bash
python train.py
```
3. Results and model checkpoints will be saved

## Requirements
- CUDA-capable GPU (recommended)
- shape_predictor_68_face_landmarks.dat file
- Sufficient disk space for model checkpoints



## License
MIT License

## References
- SuperPoint: [magic-leap-community/superpoint](https://huggingface.co/magic-leap-community/superpoint)
- dlib face landmarks predictor

## Contact
For questions and support, please open an issue on the project's GitHub repository.
