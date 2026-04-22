import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

# 1. Check for missing values and handle them.
# 2. Check for duplicate rows and remove them.
# 3. Inspect the "Sex" column for inconsistencies and standardize values.
# 4. Detect and handle outliers in the "Age" and "Fare" columns.
# 5. Create a new "Family_Size" feature based on "SibSp" and "Parch".
# 6. Convert the "Embarked" column to numerical values using label encoding.
# 7. If a "Ticket_Purchase_Date" column existed, convert it to datetime and extract the year and month.
# 8. Clean the "Name" column by removing titles and converting to lowercase.
# 9. Bin the "Age" column into categories like 'Child', 'Teen', 'Adult', and 'Senior'.

# Load the Titanic dataset
titanic_df = pd.read_csv('titanic.csv')

# 1. Check for missing values and handle them.
# Solution:
missing_values = titanic_df.isnull().sum()
print("Missing values before cleaning:")
print(missing_values)

# Handle missing values
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
titanic_df.dropna(subset=['Embarked'], inplace=True)

# Verify changes
print("\nMissing values after cleaning:")
print(titanic_df.isnull().sum())

# 2. Check for duplicate rows and remove them.
# Solution:
duplicates = titanic_df.duplicated().sum()
print(f"\nNumber of duplicate rows before cleaning: {duplicates}")

# Remove duplicates
titanic_df.drop_duplicates(inplace=True)

# Verify
duplicates_after = titanic_df.duplicated().sum()
print(f"Number of duplicate rows after cleaning: {duplicates_after}")

# 3. Inspect the "Sex" column for inconsistencies and standardize values.
# Solution:
print("\nUnique values in 'Sex' column before cleaning:")
print(titanic_df['Sex'].unique())

# Standardize values (e.g., convert 'male' and 'female' to lowercase)
titanic_df['Sex'] = titanic_df['Sex'].str.lower()

# Verify
print("\nUnique values in 'Sex' column after cleaning:")
print(titanic_df['Sex'].unique())

# 4. Detect and handle outliers in the "Age" and "Fare" columns.
# Solution:
Q1_age = titanic_df['Age'].quantile(0.25)
Q3_age = titanic_df['Age'].quantile(0.75)
IQR_age = Q3_age - Q1_age

Q1_fare = titanic_df['Fare'].quantile(0.25)
Q3_fare = titanic_df['Fare'].quantile(0.75)
IQR_fare = Q3_fare - Q1_fare

# Detect outliers in 'Age' and 'Fare' columns
age_outliers = titanic_df[(titanic_df['Age'] < (Q1_age - 1.5 * IQR_age)) | (titanic_df['Age'] > (Q3_age + 1.5 * IQR_age))]
fare_outliers = titanic_df[(titanic_df['Fare'] < (Q1_fare - 1.5 * IQR_fare)) | (titanic_df['Fare'] > (Q3_fare + 1.5 * IQR_fare))]

# Print outliers
print("\nAge outliers:")
print(age_outliers[['Age', 'Fare']])

print("\nFare outliers:")
print(fare_outliers[['Age', 'Fare']])

# Remove outliers (optional)
titanic_df = titanic_df[~((titanic_df['Age'] < (Q1_age - 1.5 * IQR_age)) | (titanic_df['Age'] > (Q3_age + 1.5 * IQR_age)))]
titanic_df = titanic_df[~((titanic_df['Fare'] < (Q1_fare - 1.5 * IQR_fare)) | (titanic_df['Fare'] > (Q3_fare + 1.5 * IQR_fare)))]

# Verify changes
print(f"\nRows after removing outliers: {titanic_df.shape[0]}")

# 5. Create a new "Family_Size" feature based on "SibSp" and "Parch".
# Solution:
titanic_df['Family_Size'] = titanic_df['SibSp'] + titanic_df['Parch'] + 1

# Verify by displaying the new column
print("\nFamily Size feature:")
print(titanic_df[['SibSp', 'Parch', 'Family_Size']].head())

# 6. Convert the "Embarked" column to numerical values using label encoding.
# Solution:
label_encoder = LabelEncoder()

# Encode 'Embarked' column
titanic_df['Embarked'] = label_encoder.fit_transform(titanic_df['Embarked'])

# Verify
print("\n'Embarked' column after label encoding:")
print(titanic_df[['Embarked']].head())

# 7. If a "Ticket_Purchase_Date" column existed, convert it to datetime and extract the year and month.
# Solution:
# Example: Add a 'Ticket_Purchase_Date' column for demonstration
titanic_df['Ticket_Purchase_Date'] = ['2021-01-10', '2021-05-23', '2021-07-15', '2021-09-09', '2021-11-18']

# Convert to datetime
titanic_df['Ticket_Purchase_Date'] = pd.to_datetime(titanic_df['Ticket_Purchase_Date'])

# Extract year and month
titanic_df['Purchase_Year'] = titanic_df['Ticket_Purchase_Date'].dt.year
titanic_df['Purchase_Month'] = titanic_df['Ticket_Purchase_Date'].dt.month

# Verify
print("\nTicket purchase date and extracted year/month:")
print(titanic_df[['Ticket_Purchase_Date', 'Purchase_Year', 'Purchase_Month']].head())

# 8. Clean the "Name" column by removing titles and converting to lowercase.
# Solution:
titanic_df['Name'] = titanic_df['Name'].apply(lambda x: re.sub(r'(Mr\.|Mrs\.|Dr\.)', '', x))
titanic_df['Name'] = titanic_df['Name'].str.lower()

# Verify
print("\nName column after normalization:")
print(titanic_df[['Name']].head())

# 9. Bin the "Age" column into categories like 'Child', 'Teen', 'Adult', and 'Senior'.
# Solution:
bins = [0, 12, 18, 64, 100]
labels = ['Child', 'Teen', 'Adult', 'Senior']

# Create new 'Age_Group' feature
titanic_df['Age_Group'] = pd.cut(titanic_df['Age'], bins=bins, labels=labels)

# Verify
print("\nAge and corresponding Age Group:")
print(titanic_df[['Age', 'Age_Group']].head())
