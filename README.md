# 🏡 End-to-End Housing Price Prediction (California Housing Dataset)

This project is a full pipeline machine learning workflow based on Chapter 2 of the book **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** by Aurélien Géron. The goal is to predict median housing prices in California districts using various features.

## 📚 Project Overview

This project walks through a complete machine learning process, from data loading to model evaluation and tuning. It demonstrates best practices in data science and machine learning engineering, including:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing Pipelines
- Model Training and Evaluation
- Hyperparameter Tuning
- Model Persistence

## 📂 Dataset

We use the **California Housing dataset** provided by `sklearn.datasets.fetch_california_housing()` or a similar version from StatLib. The dataset includes features like:

- `longitude`, `latitude`
- `housing_median_age`
- `total_rooms`, `total_bedrooms`
- `population`, `households`
- `median_income`
- `median_house_value` (target)

## ⚙️ Workflow Steps

### 1. **Data Loading**
- Load dataset using `fetch_california_housing()`
- Convert to Pandas DataFrame
- Preview and inspect basic structure and stats

### 2. **Data Visualization & EDA**
- Create geographical scatter plots
- Histograms of features
- Correlation matrix and scatter matrix plots

### 3. **Data Splitting**
- Split into training/test sets using `train_test_split` and a stable hash-based method
- Stratified sampling on `median_income` bins for representative splits

### 4. **Data Cleaning and Transformation**
- Handle missing values using `SimpleImputer`
- Encode categorical attributes using `OneHotEncoder`
- Create custom transformers with `FunctionTransformer` and `ColumnTransformer`
- Construct full preprocessing pipeline

### 5. **Feature Engineering**
- Add new features: `rooms_per_household`, `bedrooms_per_room`, `population_per_household`
- Evaluate new features’ impact using correlation and model performance

### 6. **Model Training**
- Train various models:
  - Linear Regression
  - Decision Tree
  - Random Forest
- Evaluate using cross-validation and `mean_squared_error`

### 7. **Model Fine-Tuning**
- Use `GridSearchCV` and `RandomizedSearchCV`
- Analyze best hyperparameters and feature importances

### 8. **Final Evaluation**
- Evaluate best model on test set
- Measure root mean square error (RMSE) and confidence intervals

### 9. **Saving the Model**
- Serialize final model with `joblib` for deployment

## 🛠 Tools & Libraries

- Python 3.x
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-Learn
- Jupyter Notebook / Google Colab

## ✅ Results

- Final RMSE around ~50,000–60,000 depending on model tuning
- Random Forest typically yields the best results
- Pipeline is clean and production-ready

## 📈 Future Improvements

- Try Gradient Boosting or XGBoost
- Include more advanced feature selection
- Experiment with deep learning models using TensorFlow

---




