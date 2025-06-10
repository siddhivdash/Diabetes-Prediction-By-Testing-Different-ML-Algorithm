# Diabetes Prediction Using Machine Learning

This project aims to predict the likelihood of a person having diabetes using different machine learning algorithms. The model is trained on the **Pima Indians Diabetes Dataset**, and the performance of various classifiers is evaluated—after applying **hyperparameter tuning**—to determine the most accurate prediction model.

## 📌 Project Overview

Diabetes is a chronic medical condition that affects millions worldwide. Early detection can help in timely treatment and lifestyle changes. This project tests the effectiveness of multiple classification algorithms and compares their performance to choose the best-performing model.

## 🧠 Algorithms Tested & Tuned

The following machine learning algorithms were trained, tuned, and evaluated:

- **Decision Tree Classifier**
- **Support Vector Classifier (SVC)**
- **Bernoulli Naive Bayes**

### ✅ Hyperparameter Tuning

Each model was fine-tuned using `GridSearchCV` and manual testing to identify the best parameters for optimal accuracy and generalization.  
Example tuning parameters:
- **SVC:** `C`, `kernel`, `gamma`
- **Decision Tree:** `max_depth`, `min_samples_split`, `criterion`
- **BernoulliNB:** `alpha`, `binarize`

After evaluation, **SVC** showed the highest prediction accuracy and was selected as the final model.

## 📊 Dataset Used

- **Name:** Pima Indians Diabetes Dataset  
- **Source:** [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Features:**
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (Target variable: 1 = diabetic, 0 = non-diabetic)

## 🔧 Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn (sklearn)

## 🚀 Project Workflow

1. **Data Loading:** Load the CSV dataset into a Pandas DataFrame.
2. **Data Preprocessing:**
   - Handle missing or zero values.
   - Normalize or scale features as required.
   - Split the dataset into training and test sets.
3. **Model Training & Tuning:**
   - Train Decision Tree, SVC, and Bernoulli Naive Bayes models.
   - Apply hyperparameter tuning to each algorithm.
4. **Model Evaluation:**
   - Compare accuracy scores and confusion matrices.
5. **Model Selection:**
   - Choose the model with the highest accuracy (SVC).
6. **Prediction:**
   - Use the final model to predict diabetes for new inputs.
