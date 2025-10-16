# <img width="190" height="98" alt="image" src="https://github.com/user-attachments/assets/dac52983-c1c0-4a13-b1a3-4f0dee953731" />                                                Elevate Labs AI & ML Internship

# Task 5: Decision Trees and Random Forests

## Objective
Learn tree-based models for classification using the Heart Disease Dataset.


## Key Learnings
- Binary classification with logistic regression.
- Importance of standardization, evaluation metrics (precision, recall, ROC-AUC), and threshold tuning.
- Visualization of sigmoid function and model performance.
  

## Dataset Info
- **Name:** Heart Disease Dataset
- **Source:** Downloaded from `https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset`
- Saved at: `data/input/heart_disease.csv`
- **Target:** target (1=disease, 0=no disease)


## Tools Used
- Python
- Scikit-learn
- Graphviz


## Steps Completed
1. **Load/Preprocess**: Loaded dataset, dropped unused 'Unnamed: 32' column, handled missing values (numeric with median, categorical with mode), and standardized features.
2. **Split Data**: 80-20 train-test split, saved as CSV files with 'diagnosis' included.
3. **Fit Model**: Trained Logistic Regression with max_iter=1000.
4. **Evaluate**: Computed confusion matrix, precision, recall, and ROC-AUC.
5. **Tune Threshold**: Adjusted threshold to 0.3 as an example.
6. **Visualize**: Plotted confusion matrix, ROC curve, and sigmoid function.


## Final Output
- Task Completed
  
  <img width="359" height="264" alt="image" src="https://github.com/user-attachments/assets/b1d37cff-e942-403c-8067-8452e287ced4" />


## Output Files
- `data/output/train_data.csv`: Training set with standardized features and diagnosis.
- `data/output/test_data.csv`: Test set with standardized features and diagnosis.


## Output Visualizations
- Confusion Matrix (Default Threshold):
  
  ![Confusion Matrix](visuals/confusion_matrix.png)
  
- ROC Curve:

  ![ROC Curve](visuals/roc_curve.png)

- Sigmoid Function:

  ![Sigmoid Curve](visuals/sigmoid_curve.png)


## Folder Structure
<img width="324" height="306" alt="image" src="https://github.com/user-attachments/assets/aa3182b2-e89e-4315-a675-a7192931295c" />


## Challenges & Solutions
- **Missing Data**: Used median imputation; consider domain-specific methods if needed.
- **Feature Selection**: Chose Size, Bedrooms, Age; adjust based on dataset.
- **Model Fit**: Ensured split and evaluation metrics align with task requirements.


## Code Adjustments for GitHub
- dataset_path = r'data/input/breast_cancer.csv'
- visuals_dir = 'visuals'
- confusion_matrix_path = f'{visuals}/confusion_matrix.png'
- roc_curve_path = f'{visuals}/roc_curve.png'
- sigmoid_curve_path = f'{visuals}/sigmoid_curve.png'
- output_dir = 'data/output'
- train_data_path = f'{output_dir}/train_data.csv'
- test_data_path = f'{output_dir}/train_data.csv'


## Submission
- GitHub Repository: https://github.com/Shalvi-Surve/Elevate_Labs_task4
