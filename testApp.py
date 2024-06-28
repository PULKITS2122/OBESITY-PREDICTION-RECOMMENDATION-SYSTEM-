import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


# Load the new dataset
data = pd.read_csv('obesity-final.csv')  # Update the file name

# Separate features (X) and target variable (y)
X = data.drop(columns='NObeyesdad', axis=1)
y = data['NObeyesdad']

# Identify numeric and categorical columns
numeric_columns = X.select_dtypes(include=np.number).columns
categorical_columns = X.select_dtypes(exclude=np.number).columns

# Create transformers for numeric and categorical columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

# Create a column transformer to apply transformers to the correct columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns),
    ])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2, stratify=y)

# Create a logistic regression model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000)),
])

# Ignore ConvergenceWarning for the purpose of this example
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Model Training
model.fit(x_train, y_train)

# Reset warnings to default
warnings.resetwarnings()

# Evaluate the model
y_train_pred = model.predict(x_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Accuracy on training data: ", train_accuracy)

y_test_pred = model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy on test data: ", test_accuracy)

# Example prediction using new dataset
# Update these values and columns based on the new dataset structure
test_input_values1 = {'Gender': 1, 'Age': 22.0, 'family_history_with_overweight': 1, 'FAVC': 1, 'FCVC': 2.0, 'NCP': 3.0, 'CAEC': 1, 'SMOKE': 0, 'CH2O': 2.0}
test_input_values2 = {'Gender': 1, 'Age': 26.0, 'family_history_with_overweight': 1, 'FAVC': 1, 'FCVC': 3.0, 'NCP': 3.0, 'CAEC': 2, 'SMOKE': 0, 'CH2O': 3.0}

# Convert to DataFrame
test_input1 = pd.DataFrame([test_input_values1])
test_input2 = pd.DataFrame([test_input_values2])

# Make predictions
prediction1 = model.predict(test_input1)
prediction2 = model.predict(test_input2)

print(f'predict1: {prediction1}')
print(f'predict2: {prediction2}')


