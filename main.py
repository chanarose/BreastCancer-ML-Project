"""
Ori Malca, 315150599
Chana Rosenblum, 206789711
"""
# Pre-requirements & Environment Setup
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import shap
import warnings # removing sns style warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (precision_score, recall_score, accuracy_score, f1_score, confusion_matrix,
                             adjusted_rand_score, normalized_mutual_info_score)
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from sklearn.cluster import KMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, ClassifierMixin

# CONSTANTS
ID = 'id'
LABEL = 'diagnosis'
MALIGNANT_VALUE = 'M'
BENIGN_VALUE = 'B'
MALIGNANT_NAME = 'Malignant' # cells/tumors that are cancerous
BENIGN_NAME = 'Benign' # cells/tumors that are non-cancerous
PALETTE = 'coolwarm'
SEED = 0
NUM_CLUSTERS = 2
# Visualization Hyperparameters
TITLE_FONT_SIZE = 12
PLOT_TEXT_SIZE = 11
NUM_COLUMNS = 5
# Label Occurences
FIGURE_SIZE = (6, 4)
# Corr Matrix
CORR_FIGURE_SIZE = (18, 10)
VALUE_FORMATING = ".2f"
CORR_PALETTE = "ch:s=-.2,r=.6"
# Box plots
coolwarm_palette = sns.color_palette(PALETTE, n_colors=2)
BOX_PALETTE = {BENIGN_VALUE: coolwarm_palette[0], MALIGNANT_VALUE: coolwarm_palette[-1]}
WHISKER_LEN = 2.
BOX_PLOT_FIGURE_SIZE = (15,5)
# KDE
WIDTH_PER_SUBPLOT = HEIGHT_PER_SUBPLOT = 5
LOG_SCALED = False
BENIGN_COLOR = 'blue'
MALIGNANT_COLOR = 'red'
# Outliers DIST
COLOR = 'skyblue'
ALPHA = 0.6
# Training
TEST_PORTION = 0.1
METRIC_FP_PRECISION = 3
METRIC_CRITERIA = 'recall'
SECOND_METRIC_CRITERIA = 'f1'
NUM_KFOLD = 10
NUM_PCA_COMPONENTS = 15
## Metrics we will use in training and inference
METRICS_DICT = { # metrics which requires labels
    'accuracy': lambda labels, preds: accuracy_score(labels, preds),
    'precision': lambda labels, preds: precision_score(labels, preds, zero_division=1),
    'recall': lambda labels, preds: recall_score(labels, preds, zero_division=1),
    'f1': lambda labels, preds: f1_score(labels, preds, zero_division=1),
    'ari': lambda labels, preds: adjusted_rand_score(labels, preds),
    'nmi': lambda labels, preds: normalized_mutual_info_score(labels, preds),
}

## Loading dataset
file_path = 'data/breast-cancer.csv'
df = pd.read_csv(file_path)

"""
Dataset Analysis:
The Breast Cancer Dataset contains measurements derived from breast cancer x-rays.   
Each entry includes features such as the radius, texture, and area of tumor cells, which are taken from multiple observations or parts of the tumor.
Therefore, every feature is an aggregation of statistical parameters such as mean, standard error, and maximum (often referred to as "worst").
"""

## Data Structure
# Dataset columns
df_dtypes = pd.DataFrame(df.dtypes).reset_index()
df_dtypes.columns = ['Column', 'Data Type']
print(df_dtypes)

# Our data consists of numeric measurements [floating-point numbers]
non_numeric_cols = [ID, LABEL]
numerical_df = df.drop(non_numeric_cols, axis=1)
f"Number of Samples: {numerical_df.shape[0]}"

## Data Distribution Description
numerical_df.describe().drop('count', axis=0).T

"""
Observations:
1. Data range variability, like in `area_worst` (185.2 to 4254.0) and `area_mean` (143.5 to 2501.0), suggests potential outliers and the need for normalization.
2. Outliers: Significant difference between the Q3 and max value (or min value and Q1) suggest outliers' existence.  
Examples:
  - `concavity_worst` Q3=0.38, max=1.25
  - `area_worst` Q3=1084.0, max=4254.0
  - `area_se` Q3=45.2, max=542.2
  - `perimeter_se` Q3=3.35, max=21.98
  - `area_mean` Q3=782.7, max=2501.0
Alignment with Related Works & Datasets:  
Actual Dataset source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic.
Upon reviewing related works, the distribution appears reliable and originates from the source dataset. However, it's unclear whether the skewed values are due to measurement errors.
"""

## Quantitative Measure of Skewness
skewness_rate_df = numerical_df.skew().sort_values(ascending=False)
skewness_rate_df
# All features are right-skewed in different scales, especially standard error measurements. Further Outliers & Skewness investigation is needed.

## Data Quality and Nullness
numerical_df.isnull().sum()
# Based on the null counts provided for the dataset, it can be concluded that there are no missing values across all features listed.

"""
Definition of Prediction Problem using Unsupervised Models:
- The Objective: Predicting the diagnosis of breast tissues as either malignant (M) or benign (B). This task involves utilizing a comprehensive set of diagnostic measurements.
- The Criteria Metric - Recall: It is crucial to accurately identify as many true positives as possible to ensure that individuals with cancer receive the necessary treatment.   Concurrently, maintaining a high true negative rate is important to prevent misdiagnosing healthy individuals, thereby safeguarding them from receiving unnecessary and potentially harmful treatments.
"""

# Data Visualizations
## Label Occurences
full_names = {MALIGNANT_VALUE: MALIGNANT_NAME, BENIGN_VALUE: BENIGN_NAME}
label_counts = df[LABEL].value_counts().rename(index=full_names)
plt.figure(figsize=FIGURE_SIZE)
plt.pie(label_counts, labels=label_counts.index, autopct=lambda p: '{:.0f}\n({:.1f}%)'.format(p * sum(label_counts) / 100, p), colors=sns.color_palette(PALETTE, len(label_counts)))
plt.title('Frequency of Each Diagnosis in the Dataset', fontsize=TITLE_FONT_SIZE)
plt.show()
# An imbalance between the two classes, with benign cases being more common than malignant.

## Pair-wise Pearson Correlation Heatmap
numerical_df[LABEL] = df[LABEL].apply(lambda y: (float)(y == MALIGNANT_VALUE))
corr = numerical_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=CORR_FIGURE_SIZE)
sns.heatmap(corr, mask=mask, cmap=sns.color_palette(CORR_PALETTE, as_cmap=True), annot=True, fmt=VALUE_FORMATING)
plt.show()
"""
Observations:
1. **Key Predictors**: `radius_mean`, `perimeter_mean`, `area_mean`,
   `concavity_mean`, `concave points_mean`, and `concave points_worst`
   have significant correlations with diagnosis, showing their
   importance for the predictive model.
2. **Redundant Features and Multicollinearity**: Strong correlations
   among size-related features (`radius_mean`, `perimeter_mean`,
   `area_mean`) and shape descriptors (`compactness_mean`,
   `compactness_worst`, `concavity_mean`, `concavity_worst`,
   `concave points_mean`, `concave points_worst`) indicate
   multicollinearity. This suggests these features provide overlapping
   information, making it possible to reduce them to streamline the
   model without compromising its predictive power.
3. **Weak Correlations**: Features like `fractal_dimension_mean` and
   specific `texture_se` and `symmetry_se` metrics exhibit minimal
   correlation with diagnosis, making them potential candidates for
   exclusion to simplify the model and possibly enhance performance.
"""

## Identifying Outliers Across TOP10 Skewed Distributions (Boxplot)
features_lexico_sorted = sorted(numerical_df.drop(LABEL, axis=1).columns)
top_10_skewed_features = skewness_rate_df.nlargest(10).index.tolist()
features_filtered = [feature for feature in features_lexico_sorted if feature in top_10_skewed_features]
num_features = len(features_filtered)
num_rows = num_features // NUM_COLUMNS + (1 if num_features % NUM_COLUMNS > 0 else 0)
# Create a grid of subplots
fig, axes = plt.subplots(num_rows, NUM_COLUMNS, figsize=BOX_PLOT_FIGURE_SIZE)
for i, feature in enumerate(features_filtered):
    row, col = divmod(i, NUM_COLUMNS)
    sns.boxplot(y=LABEL, x=feature, data=df, ax=axes[row, col], whis=WHISKER_LEN, palette=BOX_PALETTE)
# Hide any unused subplots
for j in range(i + 1, num_rows * NUM_COLUMNS):
    axes.flat[j].set_visible(False)
plt.tight_layout()
plt.show()
"""
Observations:

Outliers Reinforcing Distinction
(Area_worst, Perimeter_se, Area_se, Radius_se):  
- Outliers emphasize the existing trend of much higher values in Malignant cases, aiding in the distinction without negatively impacting it.

Outliers Adding Noise
(Fractal_dimension_se, Fractal_dimension_worst, Smoothness_se, Symmetry_se, Compactness_se):  
- Outliers mix with both diagnoses, adding noise and complicating classification.

(Concavity_se):  
- While it generally indicates malignancy with higher values, outliers do not distinctly enhance or weaken the separation between diagnoses, but adds some small effect.
-------------------------------------------------------------------------------------

Hypothesis: Same Outlier Data Points:
Due to the high correlation (collinearity) among many skewed features, it is highly likely that outliers in one feature are also outliers in another correlated feature. For instance, it makes sense for `Area_se` and `Area_worst` to share outliers, as a large area might also result in significant size variability (standard error) across different measurements.
"""

## Comparative Kernel Density Estimation: Top 5 & Bottom 5 Correlated Features for Benign vs. Malignant
correlations = corr[LABEL].drop(LABEL).abs().sort_values(ascending=False)
top_features = correlations.head(NUM_COLUMNS).index.tolist() + correlations.tail(NUM_COLUMNS).index.tolist()
num_rows = (len(top_features) // NUM_COLUMNS + (1 if len(top_features) % NUM_COLUMNS else 0))
fig, axes = plt.subplots(num_rows, NUM_COLUMNS, figsize=(WIDTH_PER_SUBPLOT * NUM_COLUMNS, HEIGHT_PER_SUBPLOT * num_rows), squeeze=False)
for index, feature in enumerate(top_features):
    ax = axes[index // NUM_COLUMNS, index % NUM_COLUMNS]
    sns.kdeplot(numerical_df[df[LABEL] == BENIGN_VALUE][feature], ax=ax, label=BENIGN_NAME, shade=True, color=BENIGN_COLOR, log_scale=LOG_SCALED)
    sns.kdeplot(numerical_df[df[LABEL] == MALIGNANT_VALUE][feature], ax=ax, label=MALIGNANT_NAME, shade=True, color=MALIGNANT_COLOR, log_scale=LOG_SCALED)
    ax.set_title(f'{"Highly" if index < NUM_COLUMNS else "Weakly"} Correlated', fontsize=TITLE_FONT_SIZE)
    ax.legend(fontsize=PLOT_TEXT_SIZE)
    ax.tick_params(axis='both', labelsize=PLOT_TEXT_SIZE)
for ax in axes.flat[index + 1:]:
    ax.set_visible(False)
plt.tight_layout()
plt.show()
"""
Observations:

Correlation & KDE Overlapping:
- Features with smaller overlaps between the benign and malignant KDE distributions generally show a stronger correlation with the diagnosis, and vice versa.

Highly Correlated Features:
- The separation between benign and malignant distributions in `concave points_worst/mean`, `perimeter_worst/mean` and `radius_worst` features indicates these are strong predictors of malignancy.

Weakly Correlated Features:
- Overlapping densities in `fractal dimension_se/mean`, `smoothness_se`, `texture_se`, and `symmetry_se` metrics suggest they have low discriminatory power.  
- Removing features with the most overlapping distributions, such as `fractal_dimension_se`, can reduce noise and simplify the model.
- We can see that most of the weakly has strong right tail, log transform can considered.
"""

## Collecting & Visualizing the Distribution of Outliers in the Dataset:
## Our goal is to identify outlier data points (with `id`)  from previous visualizations. Given the high correlation among features, it is highly likely that some data points are outliers across multiple features. Understanding this distribution is key to determining our approach to managing outliers.
### Collecting Outliers using configured Whishker and IQR
outliers_mask = pd.DataFrame(False, index=df[ID], columns=features_lexico_sorted)
def mark_outliers_by_diagnosis_with_id(df, features, whis=2.0):
    # Function to calculate and mark outliers using IQR, preserving sample IDs
    for diagnosis in df[LABEL].unique():
        diagnosis_group = df[df[LABEL] == diagnosis]
        for feature in features:
            Q1 = diagnosis_group[feature].quantile(0.25)
            Q3 = diagnosis_group[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - whis * IQR
            upper_bound = Q3 + whis * IQR
            # Mark outliers in the mask DataFrame, using id col to reference the correct samples
            outliers_indices = diagnosis_group[(diagnosis_group[feature] < lower_bound) | (diagnosis_group[feature] > upper_bound)][ID]
            outliers_mask.loc[outliers_indices, feature] = True
mark_outliers_by_diagnosis_with_id(df, features_lexico_sorted, WHISKER_LEN)
outliers_samples = outliers_mask[outliers_mask.any(axis=1)]
outliers_samples = outliers_samples.reset_index()

### Visualizing Outliers
outlier_counts = outliers_samples.drop(columns=[ID]).sum(axis=1)
plt.figure(figsize=FIGURE_SIZE)
bins = np.arange(outlier_counts.min(), outlier_counts.max() + 2) - 0.5
sns.histplot(outlier_counts, bins=bins, stat='count', color=COLOR, alpha=ALPHA)
total_samples = len(outlier_counts)
plt.text(0.5, 0.95, f'Total Samples: {total_samples}', ha='center', va='center',
         transform=plt.gca().transAxes, fontsize=10, bbox=dict(facecolor='white'))
plt.title('Distribution of Number of Outlier Features per Sample', fontsize=TITLE_FONT_SIZE)
plt.xlabel('Number of Outlier Features', fontsize=PLOT_TEXT_SIZE)
plt.ylabel('Frequency', fontsize=PLOT_TEXT_SIZE)
plt.xticks(range(int(outlier_counts.max()) + 1))
plt.grid(axis='y')
plt.show()
"""
Observations:

Outlier Feature Frequency:  
- The majority of samples exhibit a low number of outliered features.

Implications for Model Robustness:  
- Considerable number of data points has multiple outlier features, it can weaken the reliability of a model. We can evaluate different strategies for handling outliers for these data points.
"""


# Baseline Model Training - Birch
# Shifting from EDA to training a clustering algorithm with 10-fold cross-validation and use it as a classifier for our prediction problem.

## Data Preparation
def data_preparation(training_df, labels, test_size=TEST_PORTION, random_state=SEED):
    X = training_df
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, Y_train, Y_test

X_train_with_id, X_test_with_id, _, _ = data_preparation(df.drop([LABEL],axis=1), df[LABEL])
X_train_with_id, X_test_with_id = X_train_with_id.values, X_test_with_id.values

X_train_df, X_test_df, Y_train, Y_test = data_preparation(df.drop([ID,LABEL],axis=1), df[LABEL])
X_train, X_test = X_train_df.values, X_test_df.values
training_columns = df.columns.drop([ID,LABEL]).to_list()

## Training
### Set Cross-Validation for training
kf = KFold(n_splits=NUM_KFOLD, shuffle=False)
### Set Birch parameters
birch_params = {'n_clusters': NUM_CLUSTERS}

### Evaluation Function
# Our Function evaluates custom metrics per fold to uncover these trends.
def eval_metrics(model, cluster_labels, x, y, metrics_dict=METRICS_DICT):
    # Predict on set
    clusters = model.predict(x)
    preds = np.array([cluster_labels[c] for c in clusters])
    # Calc metrics
    metrics_results = {}
    for metric_name, func in metrics_dict.items():
        metrics_results[metric_name] = func(y, preds)
    return metrics_results

### Training Loop
# This loop facilitates the training and evaluation of 10 Birch models through K-Fold cross-validation:
# 1. Splitting the data per fold into train and validation sets.
# 2. Training the model with both data sets and employing metrics for evaluation.
# 3. Evaluating each model on a validation set to update the metrics summary and determine the best model based on predefined criteria.
def assign_cluster_labels(cluster_assignments, y_train):
    cluster_labels = {}
    for c in range(NUM_CLUSTERS):
        labels_in_cluster_c = y_train[cluster_assignments == c]
        if len(labels_in_cluster_c) == 0: # Edge case if cluster c is empty
            cluster_labels[c] = 0  # Default to 0 (or 1) if you prefer
        else: # Find majority label in cluster c
            majority_label = np.bincount(labels_in_cluster_c).argmax()
            cluster_labels[c] = majority_label
    return cluster_labels
def training_loop(cluster_model_class, cluster_model_params, X_train, Y_train, kf=kf, metrics_dict=METRICS_DICT):
    best_first_metric_criteria, best_second_metric_criteria = float('-inf'), float('-inf')
    for train_index, val_index in kf.split(X_train):
        x_train, x_val = X_train[train_index], X_train[val_index]
        y_train, y_val = Y_train[train_index], Y_train[val_index]
        # Instantiate the cluster model dynamically using cluster_model_class
        model = cluster_model_class(**cluster_model_params)
        model.fit(x_train)
        # Assign majority label for each cluster
        train_clusters = model.predict(x_train)
        cluster_labels = assign_cluster_labels(train_clusters, y_train)
        # Evaluate metrics on validation set
        fold_metrics = eval_metrics(model, cluster_labels, x_val, y_val, metrics_dict)
        # Update if this fold is better
        if (fold_metrics[METRIC_CRITERIA] > best_first_metric_criteria or
            (fold_metrics[METRIC_CRITERIA] == best_first_metric_criteria and
             fold_metrics[SECOND_METRIC_CRITERIA] > best_second_metric_criteria)):
            best_first_metric_criteria = fold_metrics[METRIC_CRITERIA]
            best_second_metric_criteria = fold_metrics[SECOND_METRIC_CRITERIA]
            best_model = model
            best_cluster_labels = cluster_labels
    return best_model, best_cluster_labels
best_model, best_cluster_labels = training_loop(Birch, birch_params, X_train, Y_train)

## Evaluation Metrics and Results
### Summary of best fold performance
### Reminder:
### - We want to reduce the number of false negatives so `recall` is our first criteria and `f1` is our second one.
### - a considerable/large number of false positives within the fold is acceptable given our focus on maximizing `recall` to minimize false negatives.
def visualize_test_results(X_test=None, Y_test=None, best_model=None, best_cluster_labels=None, result_metrics=None):
    if not result_metrics: # ignoring X_test and Y_test (using given results dict)
        result_metrics = eval_metrics(best_model, best_cluster_labels, X_test, Y_test)
    rows = [[mname, round(v, METRIC_FP_PRECISION)] for mname,v in result_metrics.items()]
    df_metrics = pd.DataFrame(rows, columns=['Metric', 'Best_Fold_on_Test'])
    return df_metrics
baseline_metrics = visualize_test_results(X_test, Y_test, best_model, best_cluster_labels)
print(baseline_metrics)


# Feature Engineering
def compare_models(dfs, names, postfix='_Best_Fold'):
    modified_dfs = []
    for df, name in zip(dfs, names):
        renamed_df = df.rename(columns={"Best_Fold_on_Test": f"{name}{postfix}"})
        modified_dfs.append(renamed_df)
    comparison_df = modified_dfs[0]
    for mod_df in modified_dfs[1:]:
        comparison_df = pd.merge(comparison_df, mod_df, on='Metric')
    return comparison_df

## Standardization (Z-score)
## feature standartization in order to prevent dominance of certain features during model training.
scaler = StandardScaler()
scaler.fit(X_train)
# Apply the transformation to both train and test sets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
best_model, best_cluster_labels = training_loop(Birch, birch_params, X_train_scaled, Y_train)
oversampled_metrics = visualize_test_results(X_test_scaled, Y_test, best_model, best_cluster_labels)
standartization_comparison_df = compare_models([baseline_metrics, oversampled_metrics], ['Baseline', 'Standartization'])
print(standartization_comparison_df)

## Positive Class Over Sampling
oversampler = RandomOverSampler(random_state=SEED)
OS_X_train_balanced, OS_Y_train_balanced = oversampler.fit_resample(X_train, Y_train)
best_model, best_cluster_labels = training_loop(Birch, birch_params, OS_X_train_balanced, OS_Y_train_balanced)
oversampled_metrics = visualize_test_results(X_test, Y_test, best_model, best_cluster_labels)
oversampled_comparison_df = compare_models([baseline_metrics, oversampled_metrics], ['Baseline', 'OverSampling'])
print(oversampled_comparison_df)
# As mentioned earlier, our dataset is heavily imbalanced, like many medical datasets. Oversampling the positive instances (cancer samples) is not effective in that case.

## Feature Selection
DIAGNOSIS_CORR_THRESHOLD = 0.25
corr_with_diagnosis = corr['diagnosis']
low_corr_features = corr_with_diagnosis[abs(corr_with_diagnosis) < DIAGNOSIS_CORR_THRESHOLD].index.tolist()
print("Dropped features with absolute correlation value less than", DIAGNOSIS_CORR_THRESHOLD, ":", low_corr_features)
# Get column indices to drop
drop_indices = [training_columns.index(feature) for feature in low_corr_features]
# Apply feature selection on the existing splits
FS_X_train = np.delete(X_train, drop_indices, axis=1)
FS_X_test = np.delete(X_test, drop_indices, axis=1)
best_model, best_cluster_labels = training_loop(Birch, birch_params, FS_X_train, Y_train)
fs_metrics = visualize_test_results(FS_X_test, Y_test, best_model, best_cluster_labels)
fs_comparison_df = compare_models([baseline_metrics, fs_metrics], ['Baseline', 'Feature_Selection'])
print(fs_comparison_df)


## Feature Extraction (Dimentionality Reduction) - PCA
## As previously discussed, our dataset has multiple correlated features, indicating multicollinearity. Some features have low correlation, and selecting just a few from each subgroup could lead to information loss. A better approach might be using PCA, which creates fewer, orthogonal features from linear combinations of the original ones. This approach offers:
## - Noise Reduction
## - Lower Data Complexity
## - Independent Features

# PCA transformation
pca = PCA(n_components=NUM_PCA_COMPONENTS)
PCA_X_train = pca.fit_transform(X_train)
PCA_X_test = pca.transform(X_test)
# Identify the most contributing features for naming
pca_columns = [X_train_df.columns[np.argmax(np.abs(pca.components_[i]))] for i in range(pca.n_components)]
# Training with PCA-transformed data
best_model, best_cluster_labels = training_loop(Birch, birch_params, PCA_X_train, Y_train)
pca_metrics = visualize_test_results(PCA_X_test, Y_test, best_model, best_cluster_labels)
pca_comparison_df = compare_models([baseline_metrics, pca_metrics], ['Baseline', 'PCA'])
print(pca_comparison_df)

## Hard-Outliers Removal
## The outliers in our dataset represent extreme data points across multiple features (as shown in the histogram above). Though not ideal, we can remove these data points and label them as abnormal, as they might not be representative of typical data. To decide which points to remove, we set a threshold based on the number of outlier features in each data point. However, this should be a last resort, as we aim to work with all data. We've explained above that we couldn't found consistent criteria for abnormal data.
NUM_OF_OUTLIER_FEATURES_THRESHOLD = 6
# Identify outliers in 'outliers_samples' ('id' is in first col)
outlier_count = outliers_samples.drop(columns=['id']).sum(axis=1)
filtered_samples = outliers_samples[outlier_count > NUM_OF_OUTLIER_FEATURES_THRESHOLD]
indices_to_remove = filtered_samples['id'].values
# Apply the removal on the training set only
train_mask = ~np.isin(X_train_with_id[:, 0], indices_to_remove)  # Mask for non-outliers
test_mask = ~np.isin(X_test_with_id[:, 0], indices_to_remove)
# Filter samples with corresponding id
OR_X_train = X_train_with_id[train_mask][:, 1:]
OR_Y_train = Y_train[train_mask]
OR_X_test = X_test_with_id[test_mask][:, 1:]
OR_Y_test = Y_test[test_mask]
print(f"{len(indices_to_remove)} outliers have been removed from the training set.")
best_model, best_cluster_labels = training_loop(Birch, birch_params, OR_X_train, OR_Y_train)
outliers_metrics = visualize_test_results(OR_X_test, OR_Y_test, best_model, best_cluster_labels)
outliers_comparison_df = compare_models([baseline_metrics, outliers_metrics], ['Baseline', 'Outliers_Removal'])
print(outliers_comparison_df)

"""
Summary For our baseline model:
| Method                             | Effect                     |
|------------------------------------|----------------------------|
| Z-Score Normalization              | Significant - Good         |
| Positive Class Oversampling        | Significant - Bad          |
| Principal Component Analysis (PCA) | No Effect                  |
| Feature Selection                  | No Effect                  |
| Outlier Removal                    | Significant - Bad          |
|------------------------------------|----------------------------|
"""

# Optimal Model and Feature Engineering Combinations (Grid Search)
# We will explore 2 more alternatives for our model. For each we'll run a grid search across all the datasets and choose the one which performed best in terms of `recall`:
# The models we will explore are:
# - GMM
# - KMeans
models = {
    KMeans: {'n_clusters': NUM_CLUSTERS, 'random_state': SEED},
    GaussianMixture: {'n_components': NUM_CLUSTERS, 'random_state': SEED, 'covariance_type': 'full', 'reg_covar': 1e-3},
    Birch: birch_params,
}
datasets = {
    'Feature Selection': (FS_X_train, Y_train, FS_X_test, Y_test),
    'Baseline Data': (X_train, Y_train, X_test, Y_test),
    'Z-Score': (X_train_scaled, Y_train, X_test_scaled, Y_test),
    'PCA': (PCA_X_train, Y_train, PCA_X_test, Y_test),
    'Hard-Outliers Removal': (OR_X_train, OR_Y_train, OR_X_test, OR_Y_test),
    'Positive Class Over Sampling': (OS_X_train_balanced, OS_Y_train_balanced, X_test, Y_test),
}
models_results = {}
for model_class, model_params in models.items():
    best_first_metric_criteria, best_second_metric_criteria = float('-inf'), float('-inf')
    for dataset_name, (_X_train, _Y_train, _X_test, _Y_test) in tqdm(datasets.items(), total=len(datasets), desc=f'{model_class.__name__}: Grid Search Over Datasets'):
        model, cluster_labels = training_loop(model_class, model_params, _X_train, _Y_train) # train
        result_metrics = eval_metrics(model, cluster_labels, _X_test, _Y_test) # test
        if (result_metrics[METRIC_CRITERIA] > best_first_metric_criteria or
            (result_metrics[METRIC_CRITERIA] == best_first_metric_criteria and
             result_metrics[SECOND_METRIC_CRITERIA] > best_second_metric_criteria)):
            best_first_metric_criteria = result_metrics[METRIC_CRITERIA]
            best_second_metric_criteria = result_metrics[SECOND_METRIC_CRITERIA]
            best_model = model
            best_cluster_labels = cluster_labels
            best_metrics_results = result_metrics
            best_dataset = dataset_name
    models_results[model_class] = (best_model, best_cluster_labels, best_metrics_results, best_dataset)
mnames_fe = [f"{m.__name__} + {r[3]}" for m, r in models_results.items()]
metrics_results = [visualize_test_results(result_metrics=best_res) for _,_,best_res,_ in models_results.values()]
combination_comparison_df = compare_models(metrics_results, mnames_fe, postfix='')
print(combination_comparison_df)
"""
Observations:
Based on our comprehensive evaluation of different models
1. **GaussianMixture + PCA**
2. **KMeans + Z-Score Data**
3. **Birch + Z-Score Data**
The best combination to maximize Recall (1st criterion) and F1 (2nd criterion) scores is **GaussianMixture** trained on the Baseline dataset.
"""

# Error and Model Performance Analysis
## Wrong Predictions
model, cluster_labels, metrics_results, dataset_name = models_results[GaussianMixture]
_X_train, _Y_train, _X_test, _Y_test = datasets[dataset_name]
def plot_confusion_matrix(predictions, ground_truth, set_type, cmap='coolwarm', annot_size=12):
    cm = confusion_matrix(ground_truth, predictions)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, square=True, cbar=False, annot_kws={"size": annot_size})
    ax.set_title(f"Confusion Matrix ({set_type})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
train_clusters = model.predict(_X_train)
train_predictions = np.array([cluster_labels[c] for c in train_clusters])
test_clusters = model.predict(_X_test)
test_predictions = np.array([cluster_labels[c] for c in test_clusters])
plt.figure(figsize=(6, 3))
plt.subplot(1,2,1)
plot_confusion_matrix(train_predictions, _Y_train, "Training", cmap='YlGnBu', annot_size=10)
plt.subplot(1,2,2)
plot_confusion_matrix(test_predictions, _Y_test, "Test", cmap='YlGnBu', annot_size=10)
plt.tight_layout()
plt.show()
"""
It seems we've captured most of the small dataset. It's important to note that the dataset is small, and the test set wasn't seen during training.
"""


# Lets take a look at the missed samples.
# Train Misses:
incorrect_train_predictions_idx = train_predictions != _Y_train
incorrect_train_predictions_samples = pd.DataFrame(_X_train[incorrect_train_predictions_idx], columns=pca_columns)
print(incorrect_train_predictions_samples)
# Test Misses:
incorrect_test_predictions_idx = test_predictions != _Y_test
incorrect_test_predictions_samples = pd.DataFrame(_X_test[incorrect_test_predictions_idx], columns=pca_columns)
print(incorrect_test_predictions_samples)
# Let's take a look at the standardized values.
numerical_description = numerical_df.describe().drop('count', axis=0)
numerical_description = numerical_description.apply(pd.to_numeric, errors='coerce')
mean_std_description = numerical_description.loc[['mean', 'std']][incorrect_test_predictions_samples.columns]
incorrect_test_predictions_samples_zscore = (incorrect_test_predictions_samples - mean_std_description.loc['mean']) / mean_std_description.loc['std']
considerable_zscore = incorrect_test_predictions_samples_zscore.stack()[abs(incorrect_test_predictions_samples_zscore).stack() > 1]
print(considerable_zscore)


## Visualize Best Model Clusters
### 2D & 3D Plot Functions
def plot_clusters_2D(X_train, X_test, Y_train, Y_test, incorrect_train_idx=None, incorrect_test_idx=None, title='Clusters - 2D Visualization'):
    pca = PCA(n_components=2)
    PCA_2D_X_train = pca.fit_transform(X_train)
    PCA_2D_X_test = pca.transform(X_test)
    # Mask correct predictions
    correct_train_idx = ~incorrect_train_idx if incorrect_train_idx is not None else np.ones_like(Y_train, dtype=bool)
    correct_test_idx = ~incorrect_test_idx if incorrect_test_idx is not None else np.ones_like(Y_test, dtype=bool)
    # Stack correct predictions
    correct_X_combined = np.vstack((PCA_2D_X_train[correct_train_idx], PCA_2D_X_test[correct_test_idx]))
    correct_y_combined = np.hstack((Y_train[correct_train_idx], Y_test[correct_test_idx]))
    # Plot correct predictions by class
    plt.figure(figsize=(10,6))
    correct_class_0 = correct_y_combined == 0
    correct_class_1 = correct_y_combined == 1
    plt.scatter(correct_X_combined[correct_class_0, 0], correct_X_combined[correct_class_0, 1], color='red', alpha=0.7, label=f"{BENIGN_NAME} Correct Predictions")
    plt.scatter(correct_X_combined[correct_class_1, 0], correct_X_combined[correct_class_1, 1], color='blue', alpha=0.7, label=f"{MALIGNANT_NAME} Correct Predictions")
    # Plot incorrect predictions
    if incorrect_train_idx is not None and incorrect_test_idx is not None:
        incorrect_X_combined = np.vstack((PCA_2D_X_train[incorrect_train_idx], PCA_2D_X_test[incorrect_test_idx]))
        plt.scatter(incorrect_X_combined[:, 0], incorrect_X_combined[:, 1], color='black', marker='x', alpha=0.9, label="Incorrect Predictions")
    plt.xticks([]); plt.yticks([])
    plt.title(title)
    plt.legend()
    plt.show()
def plot_clusters_3D(X_train, X_test, Y_train, Y_test, incorrect_train_idx=None, incorrect_test_idx=None, title="Clusters - 3D Visualization"):
    pca = PCA(n_components=3)
    PCA_3D_X_train = pca.fit_transform(X_train)
    PCA_3D_X_test = pca.transform(X_test)
    # Mask correct predictions
    correct_train_idx = ~incorrect_train_idx if incorrect_train_idx is not None else np.ones_like(Y_train, dtype=bool)
    correct_test_idx = ~incorrect_test_idx if incorrect_test_idx is not None else np.ones_like(Y_test, dtype=bool)
    correct_X_combined = np.vstack((PCA_3D_X_train[correct_train_idx], PCA_3D_X_test[correct_test_idx]))
    correct_y_combined = np.hstack((Y_train[correct_train_idx], Y_test[correct_test_idx]))
    # Plot correct predictions by class
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    correct_class_0 = correct_y_combined == 0
    correct_class_1 = correct_y_combined == 1
    ax.scatter(correct_X_combined[correct_class_0, 0], correct_X_combined[correct_class_0, 1], correct_X_combined[correct_class_0, 2],
               color='red', alpha=0.7, label=f"{BENIGN_NAME} Correct Predictions")
    ax.scatter(correct_X_combined[correct_class_1, 0], correct_X_combined[correct_class_1, 1], correct_X_combined[correct_class_1, 2],
               color='blue', alpha=0.7, label=f"{MALIGNANT_NAME} Correct Predictions")
    # Plot incorrect predictions
    if incorrect_train_idx is not None and incorrect_test_idx is not None:
        incorrect_X_combined = np.vstack((PCA_3D_X_train[incorrect_train_idx], PCA_3D_X_test[incorrect_test_idx]))
        ax.scatter(incorrect_X_combined[:, 0], incorrect_X_combined[:, 1], incorrect_X_combined[:, 2],
                   color='black', marker='x', alpha=0.9, label="Incorrect Predictions")
    # Remove tick labels + Labels and title + Adjust view angle
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_title(title, pad=10)
    ax.view_init(elev=20, azim=-135)
    ax.legend(loc='upper left', bbox_to_anchor=(-0.01, 1.015))  # Move legend to the left
    plt.show()
### Get clusters and incorrect predictions with best model
gmm_best_model, gmm_best_cluster_labels, _, _ = models_results[GaussianMixture]
#### Get clusters
train_clusters = gmm_best_model.predict(_X_train)
test_clusters = gmm_best_model.predict(_X_test)
train_preds = np.array([gmm_best_cluster_labels[c] for c in train_clusters])
test_preds = np.array([gmm_best_cluster_labels[c] for c in test_clusters])
#### Incorrect predictions
incorrect_train_predictions_idx = train_preds != _Y_train
incorrect_test_predictions_idx = test_preds != _Y_test
### Plot predicted clusters with incorrect predictions marked
plot_clusters_2D(X_train, X_test, train_preds, test_preds,
                 incorrect_train_predictions_idx, incorrect_test_predictions_idx,
                 title='Predicted Clusters - 2D Visualization')
plot_clusters_3D(X_train, X_test, train_preds, test_preds,
                 incorrect_train_predictions_idx, incorrect_test_predictions_idx,
                 title='Predicted Clusters - 3D Visualization')
### Plot real clusters
plot_clusters_2D(X_train, X_test, Y_train, Y_test, title='Real Clusters - 2D Visualization')
plot_clusters_3D(X_train, X_test, Y_train, Y_test, title='Real Clusters - 3D Visualization')


## Explainability
## Utilizing SHAP (stands for: SHapley Additive exPlanations) is a method in machine learning that helps us understand how our model works. It breaks down each prediction into parts and tells us which features are most important for that prediction.

### SHAP Summary Plots
#### Creating a wrapper for GaussianMixture because SHAP by default doesn't know how to handle clustering algorithms
class GMMBasedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=NUM_CLUSTERS, random_state=SEED, covariance_type='full', reg_covar=1e-3):
        self.n_components = n_components
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
    def fit(self, X, y):
        # create a GMM and train
        self.gmm_ = GaussianMixture(n_components=self.n_components, random_state=self.random_state,
                                    covariance_type=self.covariance_type, reg_covar=self.reg_covar)
        self.gmm_.fit(X)
        clusters = self.gmm_.predict(X) # Find the cluster assignments
        self.cluster_labels_ = {} # Determine majority label per cluster
        self.cluster_fraction_of_1_ = np.zeros(self.n_components) # Optionally store fraction_of_1s_in_cluster for predict_proba
        for c in range(self.n_components):
            mask = (clusters == c)
            labels_in_c = y[mask]
            if len(labels_in_c) == 0: # default
                self.cluster_labels_[c] = 0
            else:
                majority_label = np.bincount(labels_in_c).argmax()
                self.cluster_labels_[c] = majority_label
                self.cluster_fraction_of_1_[c] = np.mean(labels_in_c)
        return self
    def predict(self, X):
        clusters = self.gmm_.predict(X)
        return np.array([self.cluster_labels_[c] for c in clusters])
    def predict_proba(self, X):
        clusters_proba = self.gmm_.predict_proba(X)
        p_label1 = (clusters_proba * self.cluster_fraction_of_1_).sum(axis=1)
        p_label0 = 1.0 - p_label1
        return np.vstack([p_label0, p_label1]).T

# create dataframes for SHAP
_X_train_df = pd.DataFrame(_X_train, columns=pca_columns)
_X_test_df = pd.DataFrame(_X_test, columns=pca_columns)
# create wrapt GMM
gmm_model = GMMBasedClassifier(**models[GaussianMixture])
gmm_model.fit(_X_train_df, _Y_train)
# create SHAP
explainer = shap.Explainer(gmm_model.predict_proba, _X_train_df, algorithm="permutation")
# run SHAP on train set
shap_values_train = explainer(_X_train_df, silent=False)
plt.figure()
shap.summary_plot(shap_values_train[:,:,1], _X_train_df, plot_type='dot', show=False) # we are interested in what effects positive predictions
plt.show()
# run SHAP on test set
shap_values_test = explainer(_X_test_df, silent=True)
plt.figure()
shap.summary_plot(shap_values_test[:,:,1], _X_test_df, plot_type='dot', show=False) # we are interested in what effects positive predictions
plt.show()
"""
This SHAP summary plots indicates the impact of various features on the model's predictions. It uses a dot plot to show SHAP values for individual samples, with colors representing the feature's value (blue for low, red for high), and the dots' position indicating the impact on the model's output (left for Benign, right for Malignant).
---------------------------------------------------------------------------------------------------------
Observations:
1. The most significant features, according to this plot, are those that create a clear separation between red and blue dots. This aligns with our exploratory data analysis (EDA), suggesting that the model has learned the desired distribution of these features.
2. Some features do not provide a strong separation between classes. For instance, `fractal_dimension_worst` shows less distinct division, implying it might be less useful for classification. However, it does seem to contribute to the classification of Malignant data points, so its removal requires careful consideration.
3. Both summaries have similar distributions of SHAP values, indicating that the model's explanations for test predictions align with its explanations for training predictions.
"""

### Feature Importance
plt.figure()
shap.plots.bar(shap_values_train[:,:,1])
"""As we already mentioned in Summary Plot's observations, it seems that the most significant features in decision made were some of the most correlated features."""

#### Reminder: TOP10 Correlated Features
print(top_features)

### Test Wrong Prediction Analysis
plt.figure()
shap.plots.bar(explainer(incorrect_test_predictions_samples, silent=True)[:,:,1])
"""
We can see that `area_worst` had the most significant impact on incorrect predictions. Based on the training distribution (which we will check to see if it's approximately normal), we will calculate the probability of having the same or a higher value for `area_worst`.

Using Z-Score Tables:
To find the cumulative probability for a Z-score of -1.210041, you would look for the corresponding value in a standard normal distribution table.

Checking whether `area_worst` follows an approximately normal distribution, using QQ plot of our training data and normal distribution.
"""
check_col = "area_worst"
plt.figure()
stats.probplot(pd.DataFrame(_X_train, columns=pca_columns)[check_col], dist="norm", plot=plt)
plt.title(f"Q-Q plot for {check_col}")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Ordered Values")
plt.show()
"""
We can see that the data doesn't follow a normal distribution (we already knew that many features are skewed), so we won't attempt to approximate the probability of a Z-score of -1.21. Overall, this value for `area_worst` seems to have a significant impact on the prediction, and it appears to be less common compared to the other 24 features.
Possible Solution:
Consider adding more data with similar values [which we currently don't have] into the training splits.
This might change many of our assumptions and potentially improve model performance, focusing less on uncommon values of medium-correlated features.
"""


"""
# Summary
We chose to work with the [Wisconsin Breast Cancer Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), which aims to distinguish between malignant and benign tumors using 30 numerical features and a given ground truth (gt).

As we analyzed the data, we noticed that some of these 30 features were crucially highly correlated with the target variable.

We divided the dataset into train and test, with the test split kept unobserved by the model to mimic real-world scenarios.

We used several visualization and analysis methods in our pipeline, such as:

* Target Distribution (Pie Chart) – Revealed an imbalance between classes.
* Pearson Correlation – Identified less correlated features.
* Skewness Analysis using Box and KDE Plots – Revealed distinct patterns in feature distributions.
* Outlier Distribution Analysis – Identified hard-outlier data points.

Based on these analyses, we employed the following feature engineering techniques:

* Oversampling
* Feature Selection
* Hard-Outliers Removal
* Principal Component Analysis

We trained a Birch clustering model, achieving fairly good results, though recall wasn't ideal—false-negative predictions in a medical context can have severe consequences.
We also trained a KMeans and Gaussian Mixture models.

After training and applying feature engineering, we found that the PCA combined with Gaussian Mixture model provided the best predictions on the test split.

Continuing our analysis, we explored:
1. False predictions and their underlying causes.
2. Feature importance in these incorrect predictions.
3. SHAP summary plots for both test and training data to identify and compare key differences.
4. Potential solutions for false predictions, possibly through acquiring more data.
5. Feature importance across the training split.

Overall, this medical dataset, sourced from a well-known global hospital, turned out interesting. Unfortunately, we couldn't find additional data samples or extensions for this dataset.
It was intriguing to explore and apply the concepts learned in class.
"""


"""
# References
## Academic Papers
These papers utilize the Wisconsin Breast Cancer Dataset in their studies:

- [Paper 1](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9398810/)
- [Paper 2](https://www.mdpi.com/2673-7426/3/3/42)

## Official Dataset Repository
The original source for the dataset:

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

## Code Notebooks from Kaggle
Here are some notable Kaggle notebooks that explore the Wisconsin Breast Cancer Dataset:

- [Notebook 1](https://www.kaggle.com/code/youdayyy/breast-cancer-classification-xgb-96)
- [Notebook 2](https://www.kaggle.com/code/devendrasingh22/logistic-regression)
- [Notebook 3](https://www.kaggle.com/code/shaikhabdulrafay03/perceptron-using-gradient-descent)

## Important Notes
- Our approach focuses on real-world scenarios, emphasizing higher recall to minimize the risk of misdiagnosis and the potential negative consequences for patients.
- We did not rely on any existing Kaggle notebooks for our analysis.
- We explored additional research sources to better understand valid values for our sample distributions.
- Most Kaggle notebooks do not conduct thorough error analysis or extensive Exploratory Data Analysis (EDA) as we did.
"""
