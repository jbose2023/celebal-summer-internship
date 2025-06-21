import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset from seaborn
titanic = sns.load_dataset('titanic')

# Display first few rows
print(titanic.head())

# Survival Count
sns.countplot(x='survived', data=titanic)
plt.title('Survival Count (0 = No, 1 = Yes)')
plt.show()

# Survival by Gender
sns.countplot(x='sex', hue='survived', data=titanic)
plt.title('Survival Count by Gender')
plt.show()

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=titanic, x='age', kde=True, bins=30)
plt.title('Age Distribution of Passengers')
plt.show()

# Class-wise Survival
sns.countplot(x='class', hue='survived', data=titanic)
plt.title('Survival Count by Passenger Class')
plt.show()

# Age and Gender vs Survival
plt.figure(figsize=(10, 6))
sns.violinplot(x='sex', y='age', hue='survived', data=titanic, split=True)
plt.title('Age Distribution by Gender and Survival')
plt.show()

