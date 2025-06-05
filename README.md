# MNISTFASHION - Deep Learning Classification of Fashion Items
We will apply a convolutional neural network CNN to the MNIST FASHION dataset.


## Project Structure
MNISTFASHION/
├── DATA/ # Raw CSV files (train/test) 
├── MODELS/ # Saved trained models (.h5)
├── NOTEBOOKS/ # All Jupyter notebooks
│ ├── EDA.ipynb
│ ├── preprocessing.ipynb
│ └── modelFitting.ipynb
├── requirements.txt
└── README.md

Here, we could not upload the dataset because of the size, though mnist dataset is widely available on internet. Kindly download and save in this DATA directory, make sure to have 2 separate files for train and test named fashion-mnist_train.csv and fashion-mnist_test.csv


## Objective
To build a high-performance image classifier using deep learning that:
- Accurately predicts fashion categories
- Avoids overfitting through smart training strategies


## Dataset Overview
- **Dataset**: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
- **Images**: 60,000 training + 10,000 test samples
- **Image Size**: 28 × 28 pixels, grayscale
- **Classes**:
  0 - T-shirt/top  
  1 - Trouser  
  2 - Pullover  
  3 - Dress  
  4 - Coat  
  5 - Sandal  
  6 - Shirt  
  7 - Sneaker  
  8 - Bag  
  9 - Ankle boot


## Model Architecture
The CNN is designed to be deep yet regularized:

- **Input**: 28×28×1 grayscale image
- **Layers**:
  - `Conv2D(64) → BatchNormalization → Conv2D(64) → MaxPooling → Dropout(0.25)`
  - `Conv2D(128) → BatchNormalization → Conv2D(128) → MaxPooling → Dropout(0.35)`
  - `Flatten → Dense(256) → BatchNormalization → Dropout(0.5)`
  - `Dense(10, activation='softmax')`

This architecture balances **depth and generalization**.
We have included techniques like Dropout, BatchNormalization, earlyStopping, Reduce Learning rate on Plateau and ModelCheckpoint.


## ✅ Results
- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~93–94%
- **Test Accuracy**: ~93%
- **Model Generalization**: Excellent — no significant overfitting


## 👨‍💻 Author
**Ananay Singh**  
GitHub: [@ananays-ingh](https://github.com/ananays-ingh)
