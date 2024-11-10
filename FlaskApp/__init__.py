from flask import Flask, render_template, request

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import joblib
import tempfile
import os


"""SECTION 1: Data Preprocessing and Model Training"""

# Load dataset
url = "adult.csv"
df = pd.read_csv(url)

# Fill missing values
col_names = df.columns
for c in col_names:
    df[c] = df[c].replace("?", np.nan)
df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))

# Discretisation
df.replace(['Divorced', 'Married-AF-spouse', 
              'Married-civ-spouse', 'Married-spouse-absent', 
              'Never-married','Separated','Widowed'],
             ['divorced','married','married','married',
              'not married','not married','not married'], inplace = True)

# Label Encoder
category_col =['workclass', 'race', 'education','marital-status', 'occupation','relationship', 'gender', 'native-country', 'income'] 
labelEncoder = preprocessing.LabelEncoder()

# Create a map of all the numerical values of each categorical labels.
mapping_dict={}
for col in category_col:
    df[col] = labelEncoder.fit_transform(df[col])
    le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col]=le_name_mapping

mapping_dict_swap={}
for key, value in mapping_dict.items():
    mapping_dict_swap[key] = {str(v): k for k, v in mapping_dict[key].items()}

# Drop redundant columns
df=df.drop(['fnlwgt','educational-num'], axis=1)

X = df.values[:, 0:12]
Y = df.values[:,12]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
dt_clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=5, min_samples_leaf=5)
dt_clf_gini.fit(X_train, y_train)

# Save the model to model.joblib
local_path = tempfile.gettempdir()
filepath = os.path.join(local_path, 'model.joblib')
joblib.dump(dt_clf_gini, filepath)



"""SECTION 2: Flask App and Inference"""
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
