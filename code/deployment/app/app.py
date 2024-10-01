# app.py
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import joblib
import sys
import os
sys.path.insert(1, '/Users/arinagoncharova/Documents/InnoUni/PMLDL_MLOps/code/models')
import eda

#from eda import encode_features_one_hot
# from . import eda
# from models.eda import encode_features_one_hot

# Load the trained model
print(os.getcwd())
os.chdir('/Users/arinagoncharova/Documents/InnoUni/PMLDL_MLOps/')
with open("code/models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the FastAPI app
app = FastAPI()


# Define the input data schema
class CarsInput(BaseModel):
    Car_Name: str
    Year: int
    Present_Price: float
    Kms_Driven: int
    Fuel_Type: str
    Seller_Type: str
    Transmission: str
    Owner: int

def preprocess(data):

    data = pd.DataFrame(data, columns=['Car_Name', 'Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner'])
    data.drop(columns=['Car_Name'], inplace=True)
    scaler_filename = "models/scaler.save"

    encoder_filename = "models/encoder.save"
    #print(data)
    print('Data before preprocessing: ', data)
    encoder = joblib.load(encoder_filename) 
    scaler = joblib.load(scaler_filename)
    print('Encoder', encoder.get_feature_names_out())
    features_names = list(data.select_dtypes(exclude='number').columns)
    print('Features names: ', features_names)
    data = eda.encode_features_one_hot(data, features_names, encoder)
    data = pd.DataFrame(scaler.transform(data), columns = data.columns)
    return data

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: CarsInput):
    data = [[
        input_data.Car_Name,
        input_data.Year,
        input_data.Present_Price,
        input_data.Kms_Driven,
        input_data.Fuel_Type,
        input_data.Seller_Type,
        input_data.Transmission,
        input_data.Owner
    ]]
    data = preprocess(data)
    print('Data after preprocessing: ', data)
    prediction = model.predict(data)
    # print("Prediction", prediction)
    return {"prediction": prediction[0]}
    # return {"prediction": 1}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)