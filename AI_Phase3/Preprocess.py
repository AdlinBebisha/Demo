import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your dataset (replace 'your_dataset.csv' with your dataset file)
data = pd.read_csv('diabetes.csv')

# Data Preprocessing
# Step 1: Split the data into features (X) and the target variable (y)
X = data[['Age', 'Glucose', 'BMI', 'Insulin']]
y = data['Outcome']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Feature Scaling (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Output to confirm data preprocessing
print("Data Preprocessing Completed:")
print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")
print(f"Number of features: {X_train.shape[1]}")