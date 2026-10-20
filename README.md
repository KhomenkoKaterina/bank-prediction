#  Bank marketing campaign: customer segmentation & term deposit prediction

##  Project Overview
This project implements an end-to-end data analysis and machine learning pipeline on the **Bank Marketing Dataset**. The primary business objective is to predict whether a bank client will subscribe to a term deposit following a direct marketing campaign, as well as to explore customer segmentation using unsupervised learning.

Optimizing marketing calls is critical for banking operations:
* **High Recall** minimizes lost revenue by ensuring potential subscribers are not missed.
* **High Precision** optimizes call center resources by reducing unproductive calls to uninterested clients.

---

##  Dataset description
The dataset contains **11,162 records** and **17 features** (7 numerical, 10 categorical) describing client demographics, financial status, and previous marketing interaction history.
* **Target Variable:** deposit (yes / no), which is well-balanced (**47.4% positive** vs. **52.6% negative**).
* **Data Quality:** Zero explicit NaN values or duplicate rows  . However, hidden missing values encoded as 'unknown' were identified in job (0.63%),  education  (4.45%), contact (21.02%), and poutcome (74.59%).

---

##  Methodology & Pipeline

### 1. EDA & Visualization
* Analyzed distributions and identified heavy right-skewness and outliers in balance, duration, campaign, and previous  .
* Investigated feature relationships via correlation heatmaps (confirming no severe multicollinearity among numerical features) 
* Applied **Principal Component Analysis (PCA)** (2 components), demonstrating that the target classes are not linearly separable and require non-linear classification algorithms 

### 2. Data preprocessing & feature engineering 
* **Leakage Prevention:** Dropped call  duration  prior to modeling  While highly correlated with the target, call duration is not known *before* a call is placed and cannot be used for real-world pre-call prediction 
* **Stratified Splitting:** Divided data into **Train (70%)**, **Validation (15%)**, and **Test (15%)** sets  All statistical bounds, imputers, and scalers were fitted **strictly on the training set** 
* **Outlier Handling:** Capped extreme values in  age ,  balance ,  campaign , and  previous  using the Interquartile Range (**1.5 × IQR**) method 
* **Feature Engineering:**
  *  was_contacted_before : Binary indicator extracted from  pdays != -1  to separate new leads from previously contacted clients 
  *  contact_intensity : Ratio of current campaign contacts to previous contacts ( campaign / (previous + 1) ) to capture interaction pressure 
* **Encoding & Scaling:** Mode imputation for low-frequency  'unknown'  values in job, One-Hot Encoding for categorical variables, and StandardScaler for numerical features.

### 3. Unsupervised learning: customer segmentation
* Applied **K-Means Clustering** on the preprocessed training set 
* Determined the optimal number of clusters (**k = 3**) using both the **Elbow Method** (inertia drop from ~238k to ~146k) and **Silhouette Score** 
* **Key Insight:** While clusters separated clients primarily across categorical dimensions (such as mortgage status housing), conversion rates across all three clusters remained nearly identical (**45.2% to 50.0%**). High-dimensional binary features from One-Hot Encoding dominated the Euclidean distance metric over behavioral numerical features 

### 4. Supervised learning: classification & tuning
Trained and tuned three models using **5-fold Cross-Validation** optimizing for **F1-score**  :
1. **Logistic Regression** ( GridSearchCV  -> Best:  C=0.1 )  
2. **Decision Tree** ( GridSearchCV  -> Best:  max_depth=None, min_samples_leaf=20 )  
3. **Random Forest** ( RandomizedSearchCV  -> Best:  n_estimators=200, min_samples_leaf=5, max_depth=None )  

---

##  Model performance & Results

Metrics evaluated on the hold-out **Validation** and **Test** sets  :

| Model | CV Train F1 | Validation F1 | Test F1 | Test Accuracy | Test Precision | Test Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.6630 | 0.6539 | 0.6602 | 70.93% | 74.02% | 59.57% |
| **Decision Tree** | 0.6568 | 0.6746 | 0.6487 | 68.12% | 67.91% | 62.09% |
| **Random Forest** | **0.6839** | **0.6858** | **0.6834** | **72.84%** | **76.36%** | **61.84%** |

*Note: Retraining the optimal Random Forest model on the combined Train + Validation set yielded a final **Test F1-score of 0.6848** *

### Key takeaways:
* **Random Forest** outperformed linear and single-tree baselines, confirming the non-linear nature of the feature space observed during PCA 
* Minimal variance (< 0.026) between Cross-Validation, Validation, and Test F1-scores confirms strong generalization and the absence of severe overfitting 

---

##  Future improvements
* Experiment with Gradient Boosting frameworks (**XGBoost, LightGBM, CatBoost**) 
* Perform clustering strictly on scaled numerical behavioral features (or use **K-Prototypes** for mixed data types) to prevent One-Hot encoded features from dominating Euclidean distances 
* Tune the classification probability threshold based on a custom business cost matrix (balancing the cost of a phone call vs. the expected value of a deposit)  .

