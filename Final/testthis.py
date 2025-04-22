import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def load_and_clean_data(file_path):
    """
    Load and clean the network flow dataset
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Display initial info
    print("Initial dataset shape:", df.shape)
    print("\nInitial columns:", df.columns.tolist())
    
    # Convert timestamp to datetime object
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    
    # Handle missing values
    # Drop rows with critical missing values
    critical_cols = ['Source IP', 'Destination IP', 'Protocol', 'Flow Duration']
    df = df.dropna(subset=critical_cols)
    
    # For numerical columns, fill missing with median
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # For categorical columns, fill missing with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Feature engineering: Extract time-based features
    if 'Timestamp' in df.columns:
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Drop original timestamp if it exists
    if 'Timestamp' in df.columns:
        df = df.drop('Timestamp', axis=1)
    
    # Clean IP addresses (basic validation)
    ip_cols = ['Source IP', 'Destination IP']
    for col in ip_cols:
        if col in df.columns:
            # Simple IP validation (very basic)
            df = df[df[col].str.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', na=False)]
    
    # Clean port numbers (valid range)
    port_cols = ['Source Port', 'Destination Port']
    for col in port_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df[(df[col] >= 0) & (df[col] <= 65535)]
    
    # Protocol normalization
    if 'Protocol' in df.columns:
        df['Protocol'] = df['Protocol'].astype(str).str.upper().str.strip()
    
    # Remove duplicate flows
    df = df.drop_duplicates(subset=['Flow ID'], keep='first')
    
    # Display cleaned dataset info
    print("\nCleaned dataset shape:", df.shape)
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    
    return df

def vectorize_data(df):
    """
    Vectorize the cleaned dataset for machine learning
    """
    # Identify feature types
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target if it exists (for supervised learning)
    target = None
    if 'Label' in df.columns:
        target = df['Label']
        numeric_features.remove('Label')
    
    # Remove Flow ID as it's an identifier
    if 'Flow ID' in df.columns:
        df = df.drop('Flow ID', axis=1)
        if 'Flow ID' in categorical_features:
            categorical_features.remove('Flow ID')
    
    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    # Apply transformations
    X = preprocessor.fit_transform(df)
    
    # Get feature names
    numeric_features_out = numeric_features
    categorical_features_out = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_features = np.concatenate([numeric_features_out, categorical_features_out])
    
    print(f"\nFinal vectorized shape: {X.shape}")
    print(f"\nNumber of numeric features: {len(numeric_features_out)}")
    print(f"Number of categorical features: {len(categorical_features_out)}")
    print("\nFirst 5 feature names:", all_features[:5])
    
    return X, target, all_features

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = 'network_flows.csv'
    
    # Step 1: Load and clean data
    cleaned_df = load_and_clean_data(file_path)
    
    # Step 2: Vectorize data
    X, y, feature_names = vectorize_data(cleaned_df)
    
    # You can now use X and y for machine learning
    # Example: pd.DataFrame(X, columns=feature_names)