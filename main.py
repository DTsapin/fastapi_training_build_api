from fastapi import FastAPI
import uvicorn
from Model import CarSaleModel, CarSaleFactors
import pandas as pd

app = FastAPI()
model = CarSaleModel()

@app.get('/')
def index():
    '''
    Первая строка.
    '''
    return {'message': 'Hello, stranger'}


@app.post('/predict')
def predict_sales(cars: CarSaleFactors):
    
    data_dict = pd.DataFrame(
        {
            'year': [cars.year],
            'km_driven': [cars.km_driven],
            'fuel': [cars.fuel],
            'seller_type': [cars.seller_type],
            'transmission': [cars.transmission],
            'owner': [cars.owner],
        }
    )
 
    prediction = model.predict_sales(data_dict)
    return {
        'prediction': prediction
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)