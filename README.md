# 🧠 Kaggle Breast Cancer Diagnosis (ML Project)

This is a collaborative machine learning project developed independently by myself and a colleague.  
The project focuses on building interpretable models for predicting breast cancer diagnoses using the **Breast Cancer Wisconsin Diagnostic Dataset**.

Our goal was to apply a full ML pipeline—from EDA and preprocessing to model training, explainability, and evaluation—while making the project easily reproducible via Google Colab.


## Quick start <a name="start"></a>

### Run on Google Colab <a name="colab"></a>
We provide a documented implementation on Google Colab [here](https://drive.google.com/file/d/1va1-uIUx7jdbN0UBR7v5VKbwFhjx-AHq/view?usp=sharing). This includes detailed notes and experiments that outline our thought process. Once you access the notebook, simply click "Run All" to execute it. The environment will set up automatically and the code will begin running. Additionally, a copy of the notebook is provided under the name `main.ipynb`.

<u>NOTE</u>: We strongly encourage using Google Colab rather than running the code locally to visually follow the experiments and reasoning process that guided the development and decision making of this project.

### Running Locally <a name="local_imp"></a>

Create a new environment using one of the following commands:

Create a new python environment using `conda`:
```bash
conda create -y -n kaggle_breast_cancer python=3.9
conda activate kaggle_breast_cancer
```

Alternatively, you can create a new python environment using `virtualenv`:
```bash
pip install virtualenv # in case virtualenv need to be installed first
virtualenv -p=python3.9 venv
source ./venv/bin/activate
```

Install the required code libraries:

```bash
pip install -r requirements.txt
```

## Overview <a name="overview"></a>
This Colab notebook hosts the Kaggle Breast Cancer Detection project, focusing on predictive modeling for diagnosing breast cancer. The project revolves around leveraging machine learning techniques to accurately predict the presence of breast cancer using diagnostic measurements.

## Dataset <a name="dataset"></a>
- **Dataset Name**: Breast Cancer Dataset
- **Description**: The Breast Cancer Dataset comprises detailed medical data crucial for predictive modeling in breast cancer diagnosis. It includes a wide range of features extracted from digitized images of fine needle aspirates (FNA) of breast masses.
- **Attributes**: The dataset contains 569 rows and 32 attributes, including diagnostic measurements like radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.
- **Label Type**: 'diagnosis' (M = malignant, B = benign)
- **Missing Values**: None
- **Kaggle Link**: [Breast Cancer Dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)

## Challenges <a name="challenges"></a>
- **Class Imbalance**: The dataset exhibits class imbalance, with a majority of samples belonging to the 'B' (benign) class.
- **Data Sensitivity**: Addressing privacy concerns and handling the sensitivity associated with medical diagnostic data.
- **Feature Selection**: The dataset contains a large number of diverse diagnostic measurements, necessitating feature selection and dimensionality reduction techniques to identify the most impactful features.
- **Limited Samples**: Due to the limited number of data points, building robust predictive models poses challenges such as overfitting.

## Analysis Overview <a name="analysis-overview"></a>
The analysis involves exploratory data analysis (EDA), data preprocessing, model training, and evaluation. Key steps include:

1. **Exploratory Data Analysis (EDA)**: Understanding the distribution of features, correlation analysis, and visualization of diagnostic measurements.
2. **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical features.
3. **Model Training**: Utilizing various clustering algorithms such as Birch, KMeans, Gaussian Mixture Model.
4. **Model Evaluation**: Assessing model performance using metrics like accuracy, precision, recall, F1-score.

## Explainability Methods <a name="explainability-methods"></a>
To enhance the interpretability of the predictive models, explainability methods such as feature importance analysis, SHAP (SHapley Additive exPlanations), and LIME (Local Interpretable Model-agnostic Explanations) are employed. These methods help understand the contribution of different features towards model predictions and provide insights into the decision-making process.

## References <a name="references"></a>
- Breast Cancer Wisconsin (Diagnostic) Data Set. Retrieved from [Kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. Retrieved from [arXiv](https://arxiv.org/abs/1705.07874)
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. Retrieved from [arXiv](https://arxiv.org/abs/1602.04938)
