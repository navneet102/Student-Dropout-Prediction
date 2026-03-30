# Student's Dropout Prediction

This project uses Supervised Machine Learning Classifiers to predict student dropout rates based on various demographic, socioeconomic, and academic performance factors.

## Dataset Overview

The dataset (`student's dropout dataset.csv`) contains multiple columns representing student information, including:
- **Demographics**: Marital status, Nationality, Gender, Age at enrollment.
- **Socioeconomic factors**: Mother's and Father's qualification/occupation, Scholarship holder, Debtor, Tuition fees up to date.
- **Academic Performance**: Curricular units for 1st and 2nd semesters (enrolled, approved, grade, etc.), Course, Daytime/evening attendance.
- **Economic Indicators**: Unemployment rate, Inflation rate, GDP.
- **Target**: The prediction goal (Dropout, Graduate, or Enrolled).

## Methodology

The analysis follows a standard machine learning pipeline implemented in `Student's_dropout.ipynb`:
1. **Data Wrangling**: Loading and cleaning the dataset (e.g., correcting typographical errors).
2. **Data Visualization**: Using Plotly and Matplotlib to explore distributions and correlations.
3. **Pre-processing**: Scaling features using `StandardScaler` and splitting data into training and testing sets.
4. **Modeling**: Implementing various classifiers:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
   - k-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
5. **Hyperparameter Tuning**: Using `GridSearchCV` to optimize model performance.
6. **Feature Selection**: Analyzing feature importance using PCA and model coefficients.

## Results

- **Key Predictors**: Analysis of feature importance across different models consistently highlighted "Course" and "Age at enrollment" as strong predictors of student outcomes.
- **Model Performance**: The best performing model achieved an AUC (Area Under the Curve) score of 0.78, indicating good predictive capability.

## Libraries Used

- pandas
- numpy
- scikit-learn
- matplotlib
- plotly
- tqdm
## How to Run

1. Ensure you have the required Python libraries installed.
2. Open the `Student's_dropout.ipynb` notebook in a Jupyter environment.
3. Run the cells sequentially to reproduce the analysis and results.
