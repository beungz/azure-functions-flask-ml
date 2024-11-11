from flask import Flask, render_template, request

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import joblib
import tempfile
import os


filepath = 'model.joblib'

mapping_dict_swap = {'workclass': {'0': ' ?', '1': ' Federal-gov', '2': ' Local-gov', '3': ' Never-worked', '4': ' Private', '5': ' Self-emp-inc', '6': ' Self-emp-not-inc', '7': ' State-gov', '8': ' Without-pay'}, 
                    'race': {'0': ' Amer-Indian-Eskimo', '1': ' Asian-Pac-Islander', '2': ' Black', '3': ' Other', '4': ' White'}, 
                    'education': {'0': ' 10th', '1': ' 11th', '2': ' 12th', '3': ' 1st-4th', '4': ' 5th-6th', '5': ' 7th-8th', '6': ' 9th', '7': ' Assoc-acdm', '8': ' Assoc-voc', '9': ' Bachelors', '10': ' Doctorate', '11': ' HS-grad', '12': ' Masters', '13': ' Preschool', '14': ' Prof-school', '15': ' Some-college'}, 
                    'marital-status': {'0': ' Divorced', '1': ' Married-AF-spouse', '2': ' Married-civ-spouse', '3': ' Married-spouse-absent', '4': ' Never-married', '5': ' Separated', '6': ' Widowed'}, 
                    'occupation': {'0': ' ?', '1': ' Adm-clerical', '2': ' Armed-Forces', '3': ' Craft-repair', '4': ' Exec-managerial', '5': ' Farming-fishing', '6': ' Handlers-cleaners', '7': ' Machine-op-inspct', '8': ' Other-service', '9': ' Priv-house-serv', '10': ' Prof-specialty', '11': ' Protective-serv', '12': ' Sales', '13': ' Tech-support', '14': ' Transport-moving'}, 
                    'relationship': {'0': ' Husband', '1': ' Not-in-family', '2': ' Other-relative', '3': ' Own-child', '4': ' Unmarried', '5': ' Wife'}, 
                    'gender': {'0': ' Female', '1': ' Male'}, 
                    'native-country': {'0': ' ?', '1': ' Cambodia', '2': ' Canada', '3': ' China', '4': ' Columbia', '5': ' Cuba', '6': ' Dominican-Republic', '7': ' Ecuador', '8': ' El-Salvador', '9': ' England', '10': ' France', '11': ' Germany', '12': ' Greece', '13': ' Guatemala', '14': ' Haiti', '15': ' Holand-Netherlands', '16': ' Honduras', '17': ' Hong', '18': ' Hungary', '19': ' India', '20': ' Iran', '21': ' Ireland', '22': ' Italy', '23': ' Jamaica', '24': ' Japan', '25': ' Laos', '26': ' Mexico', '27': ' Nicaragua', '28': ' Outlying-US(Guam-USVI-etc)', '29': ' Peru', '30': ' Philippines', '31': ' Poland', '32': ' Portugal', '33': ' Puerto-Rico', '34': ' Scotland', '35': ' South', '36': ' Taiwan', '37': ' Thailand', '38': ' Trinadad&Tobago', '39': ' United-States', '40': ' Vietnam', '41': ' Yugoslavia'}, 
                    'income': {'0': ' <=50K', '1': ' >50K'}}

# Create Flask object "flask_app"
flask_app = Flask(__name__)

@flask_app.route('/', methods=['GET', 'POST'])
def index():
    # Render index.html
    if request.method == 'GET':
        # If it is GET, render the initial form, to get input from user
        return render_template("index.html", mapping_dict_swap=mapping_dict_swap)
    
    if (request.method == 'POST') & (request.form['button'] == 'submit'):
        # If it is POST and submit button is pressed, extract the input, execute prediction function, then render index.html with prediction output, while retain the current form input
        age = request.form['age']
        w_class = request.form['w_class']
        edu = request.form['edu']
        martial_stat = request.form['martial_stat']
        occup = request.form['occup']
        relation = request.form['relation']
        race = request.form['race']
        gender = request.form['gender']
        c_gain = request.form['c_gain']
        c_loss = request.form['c_loss']
        hours_per_week = request.form['hours_per_week']
        native_country = request.form['native_country']

        # Make dataFrame for model
        to_predict_list = {'age': age, 'workclass': w_class, 'education': edu, 'marital-status': martial_stat, 'occupation': occup, 'relationship': relation, 'race': race, 'gender': gender, 'capital-gain': c_gain, 'capital-loss': c_loss, 'hours-per-week': hours_per_week, 'native-country': native_country}
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        predicted_result = ValuePredictor(to_predict_list)

        if int(predicted_result)==1:
            prediction='Income more than 50K'
        else:
            prediction='Income less that 50K'
        
        return render_template('index.html',
                                    age=age, w_class=w_class, edu=edu, martial_stat=martial_stat, occup=occup, relation=relation, race=race, gender=gender, c_gain=c_gain, c_loss=c_loss, hours_per_week=hours_per_week, native_country=native_country,
                                    prediction=prediction,
                                    mapping_dict_swap=mapping_dict_swap
                                    )
    
    if (request.method == 'POST') & (request.form['button'] == 'clear'):
        # If it is POST and clear button is pressed, then reset the form
        return render_template("index.html", mapping_dict_swap=mapping_dict_swap)


def ValuePredictor(to_predict_list):
    """Prediction function"""
    to_predict = np.array(to_predict_list).reshape(1,12)
    # Load the pre-trained model
    loaded_model = joblib.load(filepath)
    result = loaded_model.predict(to_predict)
    return result[0]


if __name__ == "__main__":
	flask_app.run()
