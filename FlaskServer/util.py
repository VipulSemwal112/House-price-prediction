import json
import pickle
import numpy as np


__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    global __data_columns
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    new = np.zeros(len(__data_columns))
    new[0] = sqft
    new[1] = bath
    new[2] = bhk
    if loc_index >= 0:
        new[loc_index] = 1

    return round(__model.predict([new])[0],2)

def get_location_names():
    return __locations

def load_saved_artifacts():
    print('loading saved artifacts started')
    global __data_columns
    global __locations

    with open('./Artifacts/location_columns.json','r') as f:
       __data_columns= json.load(f)['data_columns']
       __locations=__data_columns[3:]

    global __model
    with open('./Artifacts/Home_price_prediction_model.pickle','rb') as f:
        __model = pickle.load(f)
        print('Loading the artifacts is done')

if __name__ =='__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price("1st phase jp nagar", 100, 3, 3))


