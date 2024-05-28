#!/opt/homebrew/bin/python3
# coding: utf-8

# In[2]:


from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Define a model for request body
class Item(BaseModel):
    item: int
    month: int

app = FastAPI()

# Load your trained model
model = XGBRegressor()
model.load_model('your_model.json')

@app.post("/predict")
def predict(item: Item):
    # Preprocess your inputs in the same way as your training data
    input_data = np.array([item.item, item.month]).reshape(1, -1)
    input_data = StandardScaler().fit_transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)

    # Return the result
    return {"prediction": prediction[0]}


# In[ ]:




