import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


df = pd.read_csv('heart.csv', delimiter=';')
df.replace('NA', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')


X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Support Vector Machine', SVC(kernel='rbf', random_state=42))
]


best_model = None
best_accuracy = 0.0

for model_name, model in models:
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    
    clf.fit(X_train, y_train)
    
    
    y_pred = clf.predict(X_test)
    
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = clf
        best_model_name = model_name


if best_model:
    joblib.dump(best_model, 'best_model_heart_disease_prediction.joblib')
    print(f"\nBest Model ({best_model_name}) saved to disk.")

