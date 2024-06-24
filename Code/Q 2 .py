import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv('heart.csv', header=0, delimiter=';')

# Display the first few rows of the dataframe to inspect the structure
print("Initial DataFrame:")
print(df.head())

# Check the shape of the DataFrame
print("Shape of the DataFrame:", df.shape)

# Assuming 'NA' or empty strings indicate missing values
df.replace('NA', np.nan, inplace=True)

# Convert columns to appropriate data types
df = df.apply(pd.to_numeric, errors='coerce')  # use 'coerce' to convert invalid parsing to NaN

# Fill missing values for numerical columns with mean
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df)

# Convert the imputed data back to a DataFrame
df_imputed = pd.DataFrame(df_imputed, columns=df.columns)

# Scaling numerical features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_imputed)

# Convert the preprocessed dataframe back to a DataFrame for plotting
df_preprocessed = pd.DataFrame(df_scaled, columns=df.columns)

# Define categorical variables for plotting
categorical_variables = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Plotting the distribution of classes for the categorical variables
for var in categorical_variables:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df[var], hue=df['target'])
    plt.title(f'Distribution of {var} based on target')
    plt.show()

# Plotting the distribution of classes for the numerical variables using boxplot
numerical_variables = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

for var in numerical_variables:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df['target'], y=df[var])
    plt.title(f'Distribution of {var} based on target')
    plt.show()
