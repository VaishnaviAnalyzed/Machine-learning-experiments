import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle  # Using pickle library
from datetime import datetime

# 1. Load the dataset
df = pd.read_csv(r'C:\Users\PC World\OneDrive\Documents\Practice Machine Learning\Regression model (SLR)\MLR\House_data.csv')

# 2. Pre-process: Calculate house 'age'
current_year = datetime.now().year
df['age'] = current_year - df['yr_built']

# 3. Select Features (X) and Target (y)
X = df[['sqft_living', 'bedrooms', 'bathrooms', 'age']]
y = df['price']

# 4. Train the Model
model_object = LinearRegression() 
model_object.fit(X, y)

# 5. Save the model using PICKLE
# 'wb' means Write Binary
with open('model.pkl', 'wb') as file:
    pickle.dump(model_object, file)

print("✅ Success: model.pkl has been created using Pickle!")