
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot styles
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Load dataset
df = pd.read_csv("titanic.csv")

# Initial Exploration
print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
print("\nData Types and Null Values:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe(include='all'))

# Missing Values
missing = df.isnull().sum().sort_values(ascending=False)
missing = missing[missing > 0]
print("\nMissing Values:")
print(missing)

# Plot Missing Values
plt.figure(figsize=(8, 5))
sns.barplot(x=missing.values, y=missing.index, palette='viridis')
plt.title("Missing Values Count")
plt.xlabel("Count")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("missing_values.png")
plt.clf()

# Univariate Analysis
categorical = ['Survived', 'Sex', 'Pclass', 'Embarked']
for col in categorical:
    sns.countplot(data=df, x=col, palette='pastel')
    plt.title(f"Countplot of {col}")
    plt.tight_layout()
    plt.savefig(f"countplot_{col}.png")
    plt.clf()

numerical = ['Age', 'Fare', 'SibSp', 'Parch']
for col in numerical:
    sns.histplot(df[col], kde=True, bins=30, color='teal')
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(f"histogram_{col}.png")
    plt.clf()

# Outlier Detection
for col in ['Age', 'Fare']:
    sns.boxplot(data=df, x=col, color='coral')
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(f"boxplot_{col}.png")
    plt.clf()

# Correlation Heatmap
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.clf()

# Bivariate Analysis
sns.barplot(data=df, x='Sex', y='Survived', palette='Set2')
plt.title("Survival Rate by Gender")
plt.tight_layout()
plt.savefig("survival_by_gender.png")
plt.clf()

sns.barplot(data=df, x='Pclass', y='Survived', palette='Set3')
plt.title("Survival Rate by Passenger Class")
plt.tight_layout()
plt.savefig("survival_by_class.png")
plt.clf()

sns.histplot(data=df, x='Age', hue='Survived', kde=True, element="step")
plt.title("Age Distribution by Survival")
plt.tight_layout()
plt.savefig("age_vs_survival.png")
plt.clf()

# Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
sns.barplot(data=df, x='FamilySize', y='Survived', palette='cool')
plt.title("Survival Rate by Family Size")
plt.tight_layout()
plt.savefig("survival_by_family_size.png")
plt.clf()

print("\nEDA Completed. Visualizations saved as PNG files.")
