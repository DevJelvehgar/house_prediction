# 🏡 California Housing Price Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning project that predicts California housing prices using advanced data science techniques and provides an interactive web application for real-time predictions.

## 🌟 Features

- **📊 Interactive Web App**: Beautiful Streamlit interface with real-time predictions
- **🔬 Complete ML Pipeline**: End-to-end data science workflow from EDA to deployment
- **🎯 Advanced Feature Engineering**: Custom transformers and feature combinations
- **📈 Model Optimization**: Hyperparameter tuning with RandomizedSearchCV
- **🗺️ Geographic Analysis**: Location-based clustering and visualization
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DevJelvehgar/house_prediction.git
   cd house_prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 📱 Web Application

The Streamlit app provides an intuitive interface where users can:

- **Adjust Parameters**: Use interactive sliders for all housing features
- **Get Predictions**: Real-time house price predictions
- **View Insights**: Feature importance and dataset statistics
- **Explore Data**: Interactive visualizations and correlations

### App Screenshots

| Main Interface | Prediction Results | Feature Analysis |
|----------------|-------------------|------------------|
| Interactive sidebar with sliders | Real-time predictions with metrics | Feature importance charts |

## 📊 Dataset

The **California Housing Dataset** contains information about housing districts in California:

| Feature | Description | Type |
|---------|-------------|------|
| `longitude` | Geographic longitude | Float |
| `latitude` | Geographic latitude | Float |
| `housing_median_age` | Median age of houses | Float |
| `total_rooms` | Total number of rooms | Float |
| `total_bedrooms` | Total number of bedrooms | Float |
| `population` | Total population | Float |
| `households` | Total number of households | Float |
| `median_income` | Median income (tens of thousands) | Float |
| `ocean_proximity` | Distance from ocean | Categorical |
| `median_house_value` | Target variable (USD) | Float |

## 🔬 Methodology

### 1. **Exploratory Data Analysis (EDA)**
- Geographic visualization with California map overlay
- Statistical analysis and correlation matrices
- Distribution analysis and outlier detection

### 2. **Data Preprocessing**
- **Missing Value Handling**: Median imputation for numerical features
- **Categorical Encoding**: One-hot encoding for ocean proximity
- **Feature Scaling**: StandardScaler for numerical features
- **Custom Transformations**: Log scaling and ratio calculations

### 3. **Feature Engineering**
- **Geographic Clustering**: K-means clustering on coordinates
- **Ratio Features**: 
  - `rooms_per_household`
  - `bedrooms_ratio` (bedrooms/total_rooms)
  - `population_per_household`
- **Log Transformations**: Applied to skewed features

### 4. **Model Training & Evaluation**
- **Algorithm**: Random Forest Regressor
- **Cross-Validation**: 3-fold stratified cross-validation
- **Hyperparameter Tuning**: RandomizedSearchCV
- **Evaluation Metrics**: RMSE with confidence intervals

### 5. **Pipeline Architecture**
```python
Pipeline([
    ('preprocessing', ColumnTransformer([
        ('bedrooms', ratio_pipeline(), ['total_bedrooms', 'total_rooms']),
        ('rooms_per_house', ratio_pipeline(), ['total_rooms', 'households']),
        ('people_per_house', ratio_pipeline(), ['population', 'households']),
        ('log', log_pipeline, ['total_bedrooms', 'total_rooms', 'population', 'households', 'median_income']),
        ('geo', cluster_simil, ['latitude', 'longitude']),
        ('cat', cat_pipeline, make_column_selector(dtype_include=object)),
    ], remainder=default_num_pipeline)),
    ('random_forest', RandomForestRegressor(random_state=42))
])
```

## 📈 Results

### Model Performance
- **RMSE**: ~42,000 USD
- **Confidence Interval**: 95% CI [40,121 - 44,284]
- **Best Algorithm**: Random Forest with optimized hyperparameters

### Feature Importance (Top 5)
1. **Median Income** - Most predictive feature
2. **Geographic Location** - Latitude/longitude clustering
3. **Rooms per Household** - Housing density
4. **Housing Median Age** - Property age
5. **Population per Household** - Demographics

## 🛠️ Technical Stack

### Core Libraries
- **Python 3.8+**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn**: Data visualization

### Web Framework
- **Streamlit**: Interactive web application
- **Plotly**: Interactive charts and graphs

### Development Tools
- **Jupyter Notebook**: Data exploration and analysis
- **Git**: Version control
- **Joblib**: Model serialization

## 📁 Project Structure

```
house_prediction/
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── california_districts_prediction.ipynb  # Jupyter notebook analysis
├── .gitignore                         # Git ignore rules
├── README.md                          # Project documentation
└── datasets/                          # Data directory (auto-created)
    └── housing/                       # California housing dataset
        └── housing.csv
```

## 🔧 Configuration

### Environment Variables
No environment variables required. The app automatically downloads the dataset on first run.

### Model Persistence
- Models are automatically saved as `california_districts_model.pkl`
- If no saved model exists, the app will train a new one
- Model retraining occurs on app restart if needed

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
The app is ready for deployment on:
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Add Procfile and requirements.txt
- **Docker**: Use provided Dockerfile
- **AWS/GCP/Azure**: Container-based deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 📊 Usage Examples

### Basic Prediction
```python
import streamlit as st
from app import load_model

# Load the trained model
model = load_model()

# Make prediction
input_data = {
    'longitude': -119.0,
    'latitude': 36.0,
    'housing_median_age': 28.0,
    'total_rooms': 2000,
    'total_bedrooms': 400,
    'population': 2000,
    'households': 500,
    'median_income': 3.0,
    'ocean_proximity': '<1H OCEAN'
}

prediction = model.predict(pd.DataFrame([input_data]))
print(f"Predicted price: ${prediction[0]:,.0f}")
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** and add tests
4. **Commit your changes**: `git commit -m 'Add feature'`
5. **Push to the branch**: `git push origin feature-name`
6. **Submit a pull request**

### Development Setup
```bash
git clone https://github.com/DevJelvehgar/house_prediction.git
cd house_prediction
pip install -r requirements.txt
streamlit run app.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **DevJelvehgar** - *Initial work* - [GitHub](https://github.com/DevJelvehgar)

## 🙏 Acknowledgments

- **Scikit-learn** team for the excellent ML library
- **Streamlit** team for the amazing web framework
- **California Housing Dataset** from StatLib
- **Hands-On Machine Learning** by Aurélien Géron for inspiration

## 📞 Support

If you encounter any issues or have questions:

1. **Check the Issues**: Look for existing solutions
2. **Create an Issue**: Describe your problem
3. **Contact**: Reach out via GitHub discussions

## 🔮 Future Enhancements

- [ ] **Advanced Models**: XGBoost, LightGBM, Neural Networks
- [ ] **Real-time Data**: Integration with live housing data APIs
- [ ] **Multi-city Support**: Expand beyond California
- [ ] **Mobile App**: React Native or Flutter mobile version
- [ ] **API Endpoints**: RESTful API for external integrations
- [ ] **Advanced Visualizations**: 3D maps and interactive plots
- [ ] **Model Explainability**: SHAP values and LIME explanations
- [ ] **A/B Testing**: Model comparison and validation

---

⭐ **Star this repository** if you found it helpful!

🐛 **Report bugs** by opening an issue

💡 **Suggest features** through discussions

📧 **Contact**: [Your Email/Contact Info]