# Импорт библиотек
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from pydantic import BaseModel
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Инициализация класса, описывающего входные факторы модели
class CarSaleFactors(BaseModel):
    year: float 
    km_driven: float 
    fuel: str
    seller_type: str
    transmission: str
    owner: str

# Инициализация класса обучения и прогнозов модели
class CarSaleModel:
    # Конструктор класса, загружаем датасет и стейт обученной модели,
    # если он имеется. Если нет - вызываем функцию _train_model и только после этого сохраняется стейт модели
    def __init__(self):
        self.df = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')
        self.model_fname_ = 'car_sale_model.pkl'
        try:
            self.model = joblib.load(self.model_fname_)
        except Exception as _:
            self.model = self._train_model()
            joblib.dump(self.model, self.model_fname_)
        
    # Непосредственно, функция подготовки данных и обучения модели CatBoostRegressor, возвращает экземпляр модели
    def _train_model(self):
        self.df.new = self.df.drop(['name'], axis=1)
        target = self.df.new['selling_price']
        features = self.df.new.drop(['selling_price'], axis=1)
        features["fuel"] = features["fuel"].astype("category")
        features["seller_type"] = features["seller_type"].astype("category")
        features["transmission"] = features["transmission"].astype("category")
        features["owner"] = features["owner"].astype("category")
        cat_features = list()
        for i in features.columns:
            if features[i].dtype != 'int64':
                get_cat = features.columns.get_loc(i)
                cat_features.append(get_cat)
        simple_imputer = SimpleImputer(strategy='median')
        pipe_num = Pipeline([('imputer', simple_imputer, ["year", "km_driven"])])
        s_imputer = SimpleImputer(strategy='constant', fill_value='unknown')
        pipe_cat = Pipeline([('imputer', s_imputer, ["fuel", "seller_type", "transmission", "owner"])])
        col_transformer = ColumnTransformer([('num_preproc', pipe_num, [x for x in features.columns if features[x].dtype!='category']),
                                     ('cat_preproc', pipe_cat, [x for x in features.columns if features[x].dtype=='category'])])
        model_1 = CatBoostRegressor(cat_features=cat_features, n_estimators=100)
        final_pipe = Pipeline([('preproc', col_transformer),('model', model_1)]).set_output(transform="dataframe")
        model = final_pipe.fit(features, target)
        return model

    # Непосредственно, функция прогнозов на основе входного массива факторов, возвращает значение прогноза
    def predict_sales(self, diction):
        data_in = diction
        prediction = self.model.predict(data_in)
        return prediction[0]