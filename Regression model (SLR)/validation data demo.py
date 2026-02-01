import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load Data
dataset = pd.read_csv(r"C:\Users\PC World\OneDrive\Desktop\final1.csv")

# 2. Handle Categorical Data (Do this while it's still a Pandas DataFrame)
# This converts 'Male'/'Female' into 1s and 0s automatically
dataset = pd.get_dummies(dataset, columns=['Gender'], drop_first=True)

# 3. Define X and y 
# Make sure the column indices [2, 3] are still correct after get_dummies
X = dataset.iloc[:, [2, 3]].values 
y = dataset.iloc[:, -1].values

# 4. Split AFTER encoding
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# 5. Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 6. Now you can score it without errors!
bias = model.score(X_train, y_train)
print(f"Training Score (Bias): {bias}")
