import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import tarfile
import urllib.request
from sklearn.compose import make_column_selector, make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import rbf_kernel

# Custom ClusterSimilarity class
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

# Helper functions
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, features_name_in):
    return ["ratio"]

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

def load_housing_data():
    """Load the California housing dataset"""
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

def create_preprocessing_pipeline():
    """Create the preprocessing pipeline"""
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore'))
    
    log_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(np.log, feature_names_out='one-to-one'),
        StandardScaler()
    )
    
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1, random_state=42)
    
    default_num_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler())
    
    preprocessing = ColumnTransformer([
        ('bedrooms', ratio_pipeline(), ['total_bedrooms', 'total_rooms']),
        ('rooms_per_house', ratio_pipeline(), ['total_rooms', 'households']),
        ('people_per_house', ratio_pipeline(), ['population', 'households']),
        ('log', log_pipeline, ["total_bedrooms", "total_rooms", "population",
                                "households", "median_income"]),
        ('geo', cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)
    
    return preprocessing

def load_model():
    """Load the trained model"""
    try:
        model = joblib.load('california_districts_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first by running the notebook.")
        return None

def train_model():
    """Train a new model if the saved one doesn't exist"""
    st.info("Training a new model... This may take a few minutes.")
    
    # Load data
    housing = load_housing_data()
    
    # Create income categories for stratified sampling
    housing['income_cat'] = pd.cut(housing['median_income'],
                                  bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                  labels=[1, 2, 3, 4, 5])
    
    # Split data
    from sklearn.model_selection import train_test_split
    strat_train_set, strat_test_set = train_test_split(
        housing, stratify=housing['income_cat'], test_size=0.2, random_state=42)
    
    # Remove income_cat column
    for set_ in (strat_train_set, strat_test_set):
        set_.drop('income_cat', axis=1, inplace=True)
    
    # Prepare features and labels
    housing_features = strat_train_set.drop('median_house_value', axis=1)
    housing_labels = strat_train_set['median_house_value'].copy()
    
    # Create preprocessing pipeline
    preprocessing = create_preprocessing_pipeline()
    
    # Create full pipeline
    full_pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('random_forest', RandomForestRegressor(random_state=42, n_estimators=100))
    ])
    
    # Train model
    full_pipeline.fit(housing_features, housing_labels)
    
    # Save model with protocol=4 for better compatibility
    try:
        joblib.dump(full_pipeline, 'california_districts_model.pkl', protocol=4)
    except Exception as e:
        st.warning(f"Could not save model: {e}")
        st.info("Model will be retrained on each app restart.")
    
    return full_pipeline

def main():
    st.set_page_config(
        page_title="California Housing Price Prediction",
        page_icon="🏠",
        layout="wide"
    )
    
    st.title("🏠 California Housing Price Prediction")
    st.markdown("Predict median house values in California districts using machine learning")
    
    # Load or train model
    model = load_model()
    if model is None:
        model = train_model()
    
    if model is None:
        st.error("Failed to load or train model. Please check the error messages above.")
        return
    
    # Sidebar for input parameters
    st.sidebar.header("🏘️ District Information")
    
    # Input fields
    longitude = st.sidebar.slider(
        "Longitude", 
        min_value=-124.5, 
        max_value=-114.0, 
        value=-119.0, 
        step=0.1,
        help="Longitude coordinate of the district"
    )
    
    latitude = st.sidebar.slider(
        "Latitude", 
        min_value=32.5, 
        max_value=42.0, 
        value=36.0, 
        step=0.1,
        help="Latitude coordinate of the district"
    )
    
    housing_median_age = st.sidebar.slider(
        "Housing Median Age", 
        min_value=1.0, 
        max_value=52.0, 
        value=28.0, 
        step=1.0,
        help="Median age of houses in the district"
    )
    
    total_rooms = st.sidebar.number_input(
        "Total Rooms", 
        min_value=1, 
        max_value=40000, 
        value=2000,
        help="Total number of rooms in the district"
    )
    
    total_bedrooms = st.sidebar.number_input(
        "Total Bedrooms", 
        min_value=1, 
        max_value=6000, 
        value=400,
        help="Total number of bedrooms in the district"
    )
    
    population = st.sidebar.number_input(
        "Population", 
        min_value=1, 
        max_value=40000, 
        value=2000,
        help="Total population in the district"
    )
    
    households = st.sidebar.number_input(
        "Households", 
        min_value=1, 
        max_value=6000, 
        value=500,
        help="Total number of households in the district"
    )
    
    median_income = st.sidebar.slider(
        "Median Income", 
        min_value=0.5, 
        max_value=15.0, 
        value=3.0, 
        step=0.1,
        help="Median income in the district (in tens of thousands of dollars)"
    )
    
    ocean_proximity = st.sidebar.selectbox(
        "Ocean Proximity",
        options=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
        help="Proximity to the ocean"
    )
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        'ocean_proximity': [ocean_proximity]
    })
    
    # Prediction button
    if st.sidebar.button("🔮 Predict House Value", type="primary"):
        try:
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display results
            st.success(f"**Predicted Median House Value: ${prediction:,.0f}**")
            
            # Additional insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Predicted Value",
                    f"${prediction:,.0f}",
                    help="Predicted median house value for this district"
                )
            
            with col2:
                rooms_per_house = total_rooms / households if households > 0 else 0
                st.metric(
                    "Rooms per House",
                    f"{rooms_per_house:.1f}",
                    help="Average number of rooms per household"
                )
            
            with col3:
                bedrooms_ratio = total_bedrooms / total_rooms if total_rooms > 0 else 0
                st.metric(
                    "Bedroom Ratio",
                    f"{bedrooms_ratio:.2f}",
                    help="Ratio of bedrooms to total rooms"
                )
            
            # Feature importance
            if hasattr(model['random_forest'], 'feature_importances_'):
                st.subheader("🔍 Feature Importance")
                feature_importances = model['random_forest'].feature_importances_
                feature_names = model['preprocessing'].get_feature_names_out()
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importances
                }).sort_values('Importance', ascending=False).head(10)
                
                st.bar_chart(importance_df.set_index('Feature'))
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Main content area
    st.subheader("📊 Dataset Overview")
    
    # Load and display sample data
    try:
        housing_data = load_housing_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Statistics:**")
            st.write(f"- Total districts: {len(housing_data):,}")
            st.write(f"- Features: {len(housing_data.columns)}")
            st.write(f"- Missing values: {housing_data.isnull().sum().sum()}")
        
        with col2:
            st.write("**Price Statistics:**")
            st.write(f"- Mean: ${housing_data['median_house_value'].mean():,.0f}")
            st.write(f"- Median: ${housing_data['median_house_value'].median():,.0f}")
            st.write(f"- Max: ${housing_data['median_house_value'].max():,.0f}")
        
        # Display sample data
        if st.checkbox("Show Sample Data"):
            st.dataframe(housing_data.head(10))
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
    
    # Instructions
    st.subheader("📝 How to Use")
    st.markdown("""
    1. **Adjust the parameters** in the sidebar to match your district of interest
    2. **Click "Predict House Value"** to get the predicted median house value
    3. **Explore the feature importance** chart to understand which factors most influence house prices
    4. **Use the dataset overview** to understand the data distribution
    """)
    
    # Model information
    st.subheader("🤖 Model Information")
    st.markdown("""
    - **Algorithm**: Random Forest Regressor
    - **Features**: Geographic location, demographics, housing characteristics
    - **Preprocessing**: Handles missing values, creates feature ratios, applies clustering
    - **Training**: Uses stratified sampling and cross-validation
    """)

if __name__ == "__main__":
    main()
