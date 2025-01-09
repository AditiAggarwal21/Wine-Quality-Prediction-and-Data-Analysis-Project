# Wine Quality Prediction and Data Analysis Project

## Introduction
This project explores a real-world classification problem using a dataset comprising over 6,000 samples and eleven significant features. The objective is to build predictive models that can classify data into predefined categories, providing insights that are both actionable and valuable for decision-making.

The workflow begins with exploratory data analysis (EDA) to understand the dataset's structure, uncover patterns, and address challenges such as missing values or imbalances. Data preprocessing ensures the dataset is clean and ready for modeling. Various classification algorithms are applied and evaluated based on performance metrics like accuracy, precision, recall, and F1-score.

Beyond model performance, this project addresses ethical considerations, including data bias and privacy concerns, while also highlighting the business relevance of the findings. Actionable recommendations are provided to demonstrate how organizations can leverage these insights to enhance decision-making, optimize processes, and achieve strategic goals.

By the end of this project, we aim to deliver a comprehensive analysis that not only solves the classification problem but also offers meaningful guidance for real-world applications.

## Objectives:
* Perform exploratory data analysis (EDA) and preprocess the dataset to ensure it is clean and suitable for classification tasks.
* Build, evaluate, and compare multiple classification models using performance metrics such as accuracy, precision, recall, and F1-score.
* Derive actionable insights from the model's outputs to support decision-making, improve processes, and address business challenges effectively

## Dataset:
[Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)

## The dataset includes information about:
* Wine chemical properties such as alcohol content, acidity, and pH level, which are key indicators of wine quality.
* Sensory attributes including characteristics like color intensity and taste profiles that impact the overall quality rating.
* Wine quality ratings assigned by experts, typically on a scale from 0 to 10, serving as the target variable for classification.
* Physical characteristics such as residual sugar, chlorides, and sulfur dioxide levels that may influence the wine’s flavor and preservation.
* Dataset size containing over 1,000 samples, ensuring a sufficient amount of data for model training and evaluation.

## Implementation:
**Libraries:** sklearn, Matplotlib, pandas, seaborn, NumPy, Scipy, Imblearn

## Exploratory Data Analysis (EDA)

### 1. Checking for Missing Values
First, I checked for missing values and duplicates in the dataset. It's important to handle missing data before performing any analysis or model building. Here's the  values of missing data and duplicates:
* **Missing Values** = 0
* **Duplicates** = 1,279

### 2. Checking for Imbalanced Dataset
Next, I checked for class imbalance in the target column. An imbalanced dataset can lead to biased models. The following chart shows the distribution of the target variable, as they are imabalanced .We will apply SMOTE technique to solve this problem.

![](img1.png)

### 3. Checking for Outliers
I also checked for outliers in the dataset, as they can significantly impact the analysis. The following boxplot shows the outliers in the numeric columns:

![](img2.png)

### 5. Correlation Analysis
To understand the relationships between numerical features, I checked for correlations. Here's a heatmap showing the correlation between variables:

![](img3.png)

### 6. Pair Plot based on Quality 
I explored the relationships between features based on the 'quality' column. Here's the pair plot that illustrates these relationships:

![](img4.png)  

### 7. Histograms
I plotted histograms for the numeric columns to understand their distributions:

![](img5.png)  

### 8. Histplot with Quality Column
I also created a histplot for the 'quality' column to analyze its distribution:

![](img6.png)  

### 9. Skewness and Kurtosis
Finally, I checked for skewness and kurtosis in the dataset. These metrics help assess the normality of the data. Some of the columns are highly positively skewed, to standerize it we will use the technique of **Log Transformation**. Here's the output for skewness and kurtosis:

![](img7.png) 

## Data Processing ( Feature Engineering)
### 1. Outlier Removal
We have used the **IQR** method to remove outliers that we saw earlier in EDA.

![](img8.png)  

### 2. Removing Skewness
We used the **Log Transformation** technique to remove skewness from our dataset.

![](img9.png) 

### 3. Feature Scaling
I used Standard Scaler to standardize features by centering them around a mean of 0 and scaling to unit variance. This ensures consistency and improves the performance of machine learning models sensitive to feature scaling.

## 4. Encoding Target Values
The target values are encoded as:
* Ratings 3-4 as "Low Quality"(0)                                                                                                            
* Ratings 5-6 as "Medium Quality"(1)                                                                                                          
* Ratings 7-8 as "High Quality"(2)

### 5. Imbalance to Balanced Dataset
![](img10.png) 
![](img11.png) 

## Model Selection and Training
**Chose several classification algorithms to evaluate:**
Logistic Regression
Support Vector Machine(SVM)
Random Forest
K-Nearest Neighbors(KNN)
Decision Tree

![](img8.png) 
![](img8.png) 
![](img8.png) 


