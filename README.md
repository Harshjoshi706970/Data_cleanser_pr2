# 🧹 Data Cleanser – Patient Health Records

## 📌 Video Explanation
https://drive.google.com/file/d/1wVsTbAOu6S_43GdVXh-nJdIstN0pFe0m/view?usp=sharing

## 📌 Project Overview

**Data Cleanser** is a Python-based data cleaning project focused on preparing a synthetic patient health records dataset for reliable analysis and machine learning.

The project handles:

* Missing values
* Categorical and numerical data imputation
* Outlier detection
* Outlier treatment
* Data quality comparison
* Export of the final cleaned dataset

The original dataset contains **1,000 patient records and 9 columns**.

---

## 🎯 Objectives

The main objectives of this project are:

1. Identify missing values in the dataset.
2. Apply suitable imputation techniques.
3. Detect and handle numerical outliers.
4. Compare different outlier treatment methods.
5. Produce a clean dataset with no missing values.
6. Save the cleaned dataset for further analysis and machine learning.

---

## 📂 Dataset

### Original Dataset

`patient_health_records_synthetic.csv`

The dataset contains the following columns:

| Column           | Description                |
| ---------------- | -------------------------- |
| `Patient_id`     | Unique patient identifier  |
| `Age`            | Patient age                |
| `Gender`         | Patient gender             |
| `Region`         | Patient region             |
| `BMI`            | Body Mass Index            |
| `Blood_pressure` | Blood pressure measurement |
| `Cholesterol`    | Cholesterol level          |
| `Glucose`        | Glucose level              |
| `Disease_risk`   | Disease-risk indicator     |

The dataset contains **1,000 records**.

---

## 🛠️ Technologies Used

* **Python 3.12.10**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Jupyter Notebook**

### Python Libraries

```python
import numpy as np
import pandas as pd
```

Scikit-learn techniques used include:

```python
SimpleImputer
KNNImputer
IterativeImputer
```

---

# 🔍 Data Cleaning Workflow

```text
Load Dataset
     ↓
Initial Data Inspection
     ↓
Missing Value Analysis
     ↓
Missing Value Imputation
     ↓
Outlier Detection
     ↓
Outlier Treatment
     ↓
Compare Results
     ↓
Final Data Validation
     ↓
Export Clean Dataset
```

---

# 1️⃣ Initial Data Assessment

The dataset initially contained the following missing values:

| Column         | Missing Values | Percentage |
| -------------- | -------------: | ---------: |
| Patient_id     |              0 |         0% |
| Age            |             20 |         2% |
| Gender         |             20 |         2% |
| Region         |             20 |         2% |
| BMI            |             30 |         3% |
| Blood_pressure |              0 |         0% |
| Cholesterol    |             30 |         3% |
| Glucose        |             30 |         3% |
| Disease_risk   |              0 |         0% |

The missing-value report was generated using Pandas.

---

# 2️⃣ Missing Value Treatment

Different imputation strategies were selected for different columns.

| Column      | Method                                     | Purpose                               |
| ----------- | ------------------------------------------ | ------------------------------------- |
| BMI         | Median Imputation                          | Reduce influence of extreme values    |
| Region      | Most Frequent                              | Suitable for categorical data         |
| Gender      | Most Frequent                              | Suitable for categorical data         |
| Age         | Random Sample Imputation                   | Preserve the distribution of Age      |
| Glucose     | KNN Imputation                             | Estimate values using similar records |
| Cholesterol | Iterative Imputation / MICE-style approach | Model relationships between variables |

### BMI – Median Imputation

```python
from sklearn.impute import SimpleImputer

sm = SimpleImputer(strategy="median")
df[["BMI"]] = sm.fit_transform(df[["BMI"]])
```

The 30 missing BMI values were successfully filled.

### Region and Gender – Most Frequent Imputation

```python
si = SimpleImputer(strategy="most_frequent")
df[["Region"]] = si.fit_transform(df[["Region"]])
df[["Gender"]] = si.fit_transform(df[["Gender"]])
```

Both categorical columns were successfully completed.

### Age – Random Sample Imputation

A missing-value indicator was created:

```python
df["miss_Age"] = df["Age"].isnull().astype(int)
```

Missing Age values were then replaced using randomly sampled existing Age values with `random_state=42`.

### Glucose – KNN Imputation

KNN imputation was performed using:

```python
KNNImputer(
    n_neighbors=5,
    weights="distance"
)
```

The missing Glucose values were successfully imputed.

### Cholesterol – Iterative Imputation

The project uses Scikit-learn's `IterativeImputer` for a chained-equations style approach:

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

im = IterativeImputer(
    max_iter=10,
    random_state=0
)
```

The 30 missing Cholesterol values were successfully filled.

---

# 3️⃣ Outlier Detection and Treatment

Three major approaches were evaluated:

* Z-score
* IQR
* Percentile-based filtering
* Winsorization

---

## 📊 Z-score – Cholesterol

The Z-score method used a threshold of `±3`.

```python
df["ZC"] = (
    df["Cholesterol"] - df["Cholesterol"].mean()
) / df["Cholesterol"].std()
```

Result:

* Records before: **1,000**
* Records after filtering: **1,000**
* Mean Cholesterol: **47.738 → 47.738**
* Standard deviation: **15.289838 → 15.289838**

No records were removed using this threshold.

---

## 📊 IQR – BMI

The IQR method was used to identify unusual BMI values.

```python
Q1 = df["BMI"].quantile(0.25)
Q3 = df["BMI"].quantile(0.75)

IQR = Q3 - Q1

upper = Q3 + 1.5 * IQR
lower = Q1 - 1.5 * IQR
```

Result:

* Records before: **1,000**
* Records after filtering: **995**
* Records removed: **5**
* Mean BMI: **27.1662 → 27.11397**
* Standard deviation: **5.122636 → 4.969423**

The IQR method retained most of the original data while reducing the effect of unusual BMI values.

---

## 📊 Percentile Method – Glucose

The executed notebook code uses the **5th and 95th percentiles**:

```python
up_limit = df["Glucose"].quantile(0.95)
lo_limit = df["Glucose"].quantile(0.05)
```

Result:

* Records before: **1,000**
* Records retained: **900**
* Mean Glucose: **105.239914 → 104.706571**
* Standard deviation: **25.375801 → 20.000844**

---

## 📊 Winsorization – Cholesterol

Winsorization was used to cap extreme values rather than remove records.

```python
df["Cholesterol_w"] = df["Cholesterol"].clip(
    lower=lo_limit,
    upper=up_limit
)
```

The dataset remained at **1,000 records**.

---

# 📈 Outlier Method Comparison

| Method        | Variable    | Mean Before | Mean After | Std Before | Std After |
| ------------- | ----------- | ----------: | ---------: | ---------: | --------: |
| Z-score       | Cholesterol |     47.7380 |    47.7380 |    15.2898 |   15.2898 |
| IQR           | BMI         |     27.1662 |    27.1140 |     5.1226 |    4.9694 |
| Percentile    | Glucose     |    105.2399 |   104.7066 |    25.3758 |   20.0008 |
| Winsorization | Cholesterol |     47.7380 |    63.3263 |    15.2898 |    4.4881 |

These values are taken from the comparison generated in the notebook.

---

# 🏆 Key Findings

### Best Imputation Strategy

Different techniques were effective for different columns:

* **Median imputation** worked well for BMI.
* **Most Frequent imputation** handled Gender and Region.
* **Random Sample Imputation** was used for Age.
* **KNN Imputation** was used for Glucose.
* **Iterative Imputation / MICE-style approach** was used for Cholesterol.

Overall, the project demonstrates that one imputation technique does not have to be applied to every variable.

### Best Outlier Method

The **IQR method for BMI** performed best in the project's comparison because only **5 records out of 1,000** were removed while the spread of BMI decreased.

---

# ✅ Final Dataset

The cleaned dataset is exported as:

```text
patient_health_clean_dataset.csv
```

The final validation showed **zero missing values** in all columns present at export time.

The final dataset contains the original columns plus helper columns created during processing, including:

```text
miss_Age
Cholesterol_w
```

---

# 📁 Recommended GitHub Repository Structure

```text
Data-Cleanser/
│
├── Data_Cleanser_pr2(3).ipynb
├── patient_health_records_synthetic.csv
├── patient_health_clean_dataset.csv
├── Data_Cleanser_Project_Documentation.docx
├── README.md
└── requirements.txt
```

---

# 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Open the project folder

```bash
cd Data-Cleanser
```

### 3. Install required libraries

```bash
pip install pandas numpy scikit-learn jupyter
```

Or, if a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the notebook

Open:

```text
Data_Cleanser_pr2(3).ipynb
```

Run the cells in order.

> **Important:** The current notebook uses a local Windows file path for the input CSV. For GitHub portability, change it to a relative path such as:
>
> ```python
> df = pd.read_csv("patient_health_records_synthetic.csv")
> ```

---

# ⚠️ Notes

* The dataset used in this project is a **synthetic patient health records dataset**.
* The notebook's percentile description mentions the 1st/99th percentiles, but the executed code uses the **5th/95th percentiles**.
* The Winsorization step uses `lo_limit` and `up_limit` created in the Glucose percentile section, so these bounds should be reviewed if the method is reused.
* The final exported dataset includes helper columns such as `miss_Age` and `Cholesterol_w`.

---

# 📌 Project Outcome

The Data Cleanser project successfully demonstrates a complete data preprocessing workflow:

```text
Raw Dataset
     ↓
Missing Value Analysis
     ↓
Imputation
     ↓
Outlier Detection
     ↓
Outlier Treatment
     ↓
Data Validation
     ↓
Clean Dataset
```

The final dataset contains **1,000 patient records with zero missing values**, making it more suitable for further statistical analysis, visualization, and machine learning applications.

---

## 👨‍💻 Author

**Your Name**

Data Cleaning & Preprocessing Project

---

## 📄 Documentation

Additional project documentation is available in:

```text
Data_Cleanser_Project_Documentation.docx
```
