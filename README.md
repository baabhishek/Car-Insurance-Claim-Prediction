# Car-Insurance-Claim-Prediction
The aim of this project is to predict whether a customer will file a claim on their car insurance policy based on various factors, including policyholder demographics, car details, and policy characteristics. The project involves end-to-end data processing, exploratory data analysis (EDA), feature selection, data balancing, and predictive modeling.

## Table of Contents

- Dataset Description

- Data Preprocessing

- Exploratory Data Analysis (EDA)

- Handling Imbalanced Data

- Model Building and Evaluation

- Results

- Conclusion

## Dataset Description

The dataset contains 44 features, which include details about the policy, the vehicle, the policyholder, and other technical specifications. The target variable is is_claim, which indicates whether a claim was made (1) or not (0).

## Data Preprocessing

- Steps Included:

### Data Cleaning:

- Handling missing values.

- Standardizing column names.

### Feature Selection:

- Statistical methods such as ANOVA and chi-square tests.

- Multi-collinearity checks using Corr matrix.

- Random Forest feature importance.

## Column Transformation:

- Used pipelines for consistent preprocessing.

- Handled categorical variables using one-hot encoding and label encoding.

## Exploratory Data Analysis (EDA)

- Performed EDA to uncover insights and understand data distribution.

### Techniques:

- Univariate Analysis: Examined distributions of individual features.

- Bivariate Analysis: Analyzed relationships between features and the target variable.

- Multivariate Analysis: Explored feature interactions and correlations.

### Key visualizations include:

- Count plots for categorical variables.

- Box plots for numerical variables.

- Heatmaps for correlation.

- Handling Imbalanced Data

- The dataset was imbalanced with a skewed distribution of the is_claim target variable. This was addressed using SMOTE (Synthetic Minority Over-sampling Technique).

### Visualizations were used to:

- Compare class distributions before and after balancing.

- Validate the effectiveness of SMOTE.

## Model Building and Evaluation

Multiple models were built and evaluated to find the best-performing one. The following algorithms were tested:

- Random Forest

- Logistic Regression

- Support Vector Machine (SVM)

- XGBoost Classifier

Performance Metrics:

Models were evaluated using the following metrics:

1. Accuracy 2. Precision 3. Recall 4. F1-Score

Results:

## Results
![Results Visualization](/Users/abhisheksenapati/Desktop/Screenshot 2568-01-15 at 4.52.00 PM.png)


- The XGBoost Classifier outperformed other models with the highest accuracy (92.1%) and robust precision, recall, and F1-score. Random Forest also performed well with an accuracy of 91.6%.


