# Breast Cancer Detection

This project aims to predict whether a tumor is **malignant** or **benign** based on features such as the size and texture of cell nuclei. The analysis leverages **binary classification** techniques using **Logistic Regression** and **Random Forest Classifier** models.

---

## Table of Contents
- [Skills Gained](#skills-gained)
- [Tools and Libraries](#tools-and-libraries)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
  - [Step 1: Load and Explore the Dataset](#step-1-load-and-explore-the-dataset)
  - [Step 2: Data Preprocessing](#step-2-data-preprocessing)
  - [Step 3: Train and Evaluate Models](#step-3-train-and-evaluate-models)
  - [Step 4: Model Comparison and Interpretation](#step-4-model-comparison-and-interpretation)
- [Results](#results)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)

---

## Skills Gained
- Binary classification
- Model evaluation (accuracy, precision, recall)
- Visualization using confusion matrices

## Tools and Libraries
- **Scikit-learn**: Model building and evaluation
- **Pandas**: Data manipulation
- **Seaborn & Matplotlib**: Visualization tools

---

## Dataset

The dataset used is the **Breast Cancer Wisconsin Diagnostic dataset**, which is readily available in the `sklearn` library.

- **Features**: 30 numerical attributes related to cell nuclei
- **Target**:  
  - `0`: Malignant  
  - `1`: Benign  

---

## Project Workflow

### Step 1: Load and Explore the Dataset

- Import the **Breast Cancer dataset** from `sklearn.datasets`.
- Convert the data to a Pandas DataFrame for analysis.
- Explore the data using functions like `head()`, `describe()`, and `value_counts()`.

---

### Step 2: Data Preprocessing

- Split the data into **features (X)** and **target (y)**.
- Use `train_test_split` to divide the dataset into **training** and **testing** sets.

---

### Step 3: Train and Evaluate Models

#### 1. Logistic Regression
- Train the **Logistic Regression** model on the training set.
- Evaluate the performance using:
  - **Accuracy**
  - **Classification Report** (Precision, Recall, F1-Score)
  - **Confusion Matrix**

#### 2. Random Forest Classifier
- Train the **Random Forest** model.
- Evaluate the model using the same metrics as Logistic Regression.

---

### Step 4: Model Comparison and Interpretation

- Compare the **accuracy** of both models.
- Visualize the **Confusion Matrix** for both Logistic Regression and Random Forest to understand the predictions.

---

## Results

| Model                | Accuracy |
|----------------------|----------|
| Logistic Regression  | 95%      |
| Random Forest        | 97%      |

The Random Forest model achieved slightly higher accuracy than Logistic Regression.

---

## Conclusion

In this project:
- We used **Logistic Regression** and **Random Forest** to predict tumor malignancy.
- **Accuracy** was used as the main evaluation metric.
- The **Confusion Matrix** helped us identify false positives and false negatives.
- Random Forest outperformed Logistic Regression in this case.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/breast-cancer-detection.git
   cd breast-cancer-detection
