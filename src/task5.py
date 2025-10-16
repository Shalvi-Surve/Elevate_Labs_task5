# Elevate Labs Internships
# Task - 5: Decision Trees and Random Forests
# Dataset Used : Heart Disease Dataset
# Tools : Scikit-learn, Graphviz

# Task 5: Decision Trees and Random Forests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

# Load dataset (assuming downloaded as heart_disease.csv)
data_path = r'C:\Users\Shalvi\OneDrive\Desktop\ELEVATE LABS (INTERNSHIP)\5. TASK - 5\data\input\heart_disease.csv'
if not os.path.exists(data_path):
    print(f"Error: File not found at {data_path}. Please download heart_disease.csv and place it in the data/input folder.")
else:
    df = pd.read_csv(data_path)

    # Assume 'target' is the target (1=disease, 0=no disease); adjust if different
    X = df.drop('target', axis=1)  # Features
    y = df['target']  # Binary target

    # Handle missing values (fill with median for numeric)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

    # Split and save data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    output_dir = r'C:\Users\Shalvi\OneDrive\Desktop\ELEVATE LABS (INTERNSHIP)\5. TASK - 5\data\output'
    os.makedirs(output_dir, exist_ok=True)
    train_df = X_train.copy()
    train_df['target'] = y_train
    test_df = X_test.copy()
    test_df['target'] = y_test
    train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)

    # Decision Tree (default depth)
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    print(f"Decision Tree Accuracy (default): {dt_accuracy:.2f}")

    # Control overfitting (limit depth to 3)
    dt_controlled = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_controlled.fit(X_train, y_train)
    dt_controlled_pred = dt_controlled.predict(X_test)
    dt_controlled_accuracy = accuracy_score(y_test, dt_controlled_pred)
    print(f"Decision Tree Accuracy (max_depth=3): {dt_controlled_accuracy:.2f}")

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

    # Feature Importances
    importances = rf_model.feature_importances_
    feat_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feat_importance = feat_importance.sort_values('Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_importance)
    plt.title('Feature Importances (Random Forest)')
    visuals_dir = r'C:\Users\Shalvi\OneDrive\Desktop\ELEVATE LABS (INTERNSHIP)\5. TASK - 5\visuals'
    os.makedirs(visuals_dir, exist_ok=True)
    plt.savefig(os.path.join(visuals_dir, 'feature_importance.png'))
    plt.close()

    # Cross-validation
    dt_cv_scores = cross_val_score(dt_controlled, X, y, cv=5)
    rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)
    print(f"Decision Tree CV Scores (mean): {dt_cv_scores.mean():.2f} (+/- {dt_cv_scores.std() * 2:.2f})")
    print(f"Random Forest CV Scores (mean): {rf_cv_scores.mean():.2f} (+/- {rf_cv_scores.std() * 2:.2f})")

    print("\nTask 5 completed. Check visuals folder for feature importance plot.")