# EXNO2DS
# AIM: To perform Exploratory Data Analysis on the given data set.
### NAME : VINOTH M P
### REG NO : 212223240182
### DATE : 09/09/2024
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
from IPython import get_ipython
from IPython.display import display
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('/content/titanic_dataset.csv')
print(data.isnull().sum())
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
```
![image](https://github.com/user-attachments/assets/82065096-3a04-4152-a43c-26b8e07ff036)

```
sns.boxplot(data=data, x='Age')
plt.title('Boxplot of Age')
plt.show()
```
![image](https://github.com/user-attachments/assets/48d14925-4ab6-4185-bd78-4bc721c96064)

```
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
data_no_outliers = remove_outliers_iqr(data, 'Age')

sns.countplot(data=data_no_outliers, x='Survived')
plt.title('Countplot of Survived')
plt.show()
```
![image](https://github.com/user-attachments/assets/abc15997-70ec-4a2b-9b3c-df6b09606e36)
```
sns.displot(data_no_outliers['Fare'], kde=True)
plt.title('Distplot of Fare')
plt.show()
```
![image](https://github.com/user-attachments/assets/cca2860f-1250-479c-bc70-33cc780207d4)

```
cross_tab = pd.crosstab(data_no_outliers['Pclass'], data_no_outliers['Survived'])
print(cross_tab)
```
![image](https://github.com/user-attachments/assets/1c061120-8441-40fa-a668-49fc2f2da70d)
```
sns.heatmap(cross_tab, annot=True, cmap="YlGnBu")
plt.title('Heatmap of Pclass vs Survived')
plt.show()
```
![image](https://github.com/user-attachments/assets/63e3c06a-3e54-4f23-ba06-9f44b0014cbc)

# RESULT
Thus the program to Perform Exploratory data analysis for the given data set is successfully performed

