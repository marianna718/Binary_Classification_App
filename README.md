# 🍄 Mushroom Classification Web App

This is a **Streamlit web app** that performs binary classification to predict whether a mushroom is **edible** or **poisonous** using different machine learning algorithms.

## 🚀 Features

- Choose from:
  - **Support Vector Machine (SVM)**
  - **Logistic Regression**
  - **Random Forest Classifier**
- Interactive **hyperparameter tuning** from the sidebar
- Display performance metrics:
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve
- View raw data used in training

## 🧠 Dataset

The app uses the `mushrooms.csv` dataset. Make sure the file is in the same directory as your Python script.

## 📦 Installation

1. Clone this repository or download the files.

2. Install dependencies:

```bash
pip install -r requirements.txt
```
##  Runing 
```bash
streamlit run app.py
```

##  Structure
``` bash
project/
├── app.py                 # Main Streamlit application
├── mushrooms.csv          # Mushroom dataset (input)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

```


