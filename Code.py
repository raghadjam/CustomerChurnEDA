# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from scipy.stats import pointbiserialr


# Load the dataset
df = pd.read_csv("customer_data.csv")

# Display the first few rows
print("First five rows of the dataset:")
print(df.head())

# Display general info about the dataset
print("\nDataset Information:")
print(df.info())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values in each column
print("Missing values in each column:")
print(df.isnull().sum())

# Fill missing numeric values with median safely
numeric_cols = ['Age', 'Income', 'Tenure', 'SupportCalls']
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

print("Missing values after imputation:")
print(df.isnull().sum())

#  Handle outliers using IQR capping 
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Capping/flooring
    df[col] = np.where(df[col] > upper_bound, upper_bound,
                       np.where(df[col] < lower_bound, lower_bound, df[col]))


scaler = StandardScaler()

for col in numeric_cols:
    # Fit and transform each column individually
    df[col] = scaler.fit_transform(df[[col]])


for col in numeric_cols:
    plt.figure(figsize=(8,4))
    plt.hist(df[col], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

categorical_cols = ['Gender', 'ChurnStatus', 'ProductType']  

for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=df[col], palette='pastel')
    plt.title(f'Bar Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()

target = 'ChurnStatus'

for col in numeric_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=target, y=col, data=df, palette='pastel')
    plt.title(f'{col} vs {target}')
    plt.xlabel(target)
    plt.ylabel(col)
    plt.show()

categorical_cols = ['Gender', 'ProductType']

for col in categorical_cols:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, hue=target, data=df, palette='pastel')
    plt.title(f'{col} vs {target}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.legend(title=target)
    plt.show()

numeric_cols_corr = ['Age', 'Income', 'Tenure', 'SupportCalls', 'ChurnStatus']

corr_matrix = df[numeric_cols_corr].corr(method='pearson')

# Display the matrix
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
plt.show()


# 6. Data Visualizations 



# Bar Plot - Product Type vs Churn Status
plt.figure()
sns.countplot(x='ProductType', hue='ChurnStatus', data=df, palette='Set2')
plt.title('Churn Status by Product Type')
plt.xlabel('Product Type (0 = Basic, 1 = Premium)')
plt.ylabel('Count')
plt.legend(title='Churn Status', labels=['Stayed', 'Churned'])
plt.show()

# Bar Plot - Income vs Churn Status (using income bins)
plt.figure()
# Create income bins for better visualization
df['IncomeRange'] = pd.cut(df['Income'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
sns.countplot(x='IncomeRange', hue='ChurnStatus', data=df, palette='pastel')
plt.title('Churn Status by Income Range')
plt.xlabel('Income Range')
plt.ylabel('Count')
plt.legend(title='Churn Status', labels=['Stayed', 'Churned'])
plt.show()

# Heatmap - Correlation with Target 
numeric_cols_corr = ['Age', 'Income', 'Tenure', 'SupportCalls', 'ChurnStatus']
corr_matrix = df[numeric_cols_corr].corr()

plt.figure()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Correlation Matrix of Numeric Features')
plt.xlabel('Features')
plt.ylabel('Features')
plt.show()

# Bar Plot - Tenure vs Churn Status 

df['ChurnLabel'] = df['ChurnStatus'].map({0: 'Stayed', 1: 'Churned'}).astype('category')
order = ['Stayed', 'Churned']
palette = {'Stayed': 'orange', 'Churned': 'green'}

# Bin tenure, then plot a grouped bar chart
bins = pd.cut(df['Tenure'], bins=8)
ct = pd.crosstab(bins, df['ChurnLabel'])  # rows: tenure bins, cols: Stayed/Churned

ct[order].plot(kind='bar', figsize=(10,5), color=[palette['Stayed'], palette['Churned']])
plt.title('Churn Status by Tenure Range')
plt.xlabel('Tenure Range (standardized)')
plt.ylabel('Number of Customers')
plt.legend(title='Churn Status')
plt.tight_layout()
plt.show()