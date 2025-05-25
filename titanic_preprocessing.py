import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("titanic.csv")

# Show basic info
print(df.head())
print(df.info())
print(df.describe())

# Drop columns with too many missing values or not useful for ML
df.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True)

# Handle missing values
# Fill Age with median and Embarked with mode
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert categorical variables to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Normalize selected features
scaler = StandardScaler()
columns_to_scale = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Show final DataFrame
print("\nPreprocessed DataFrame:")
print(df.head())

# Save the cleaned data
df.to_csv("titanic_cleaned.csv", index=False)
print("\nCleaned dataset saved as 'titanic_cleaned.csv'")
