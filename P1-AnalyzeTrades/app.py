# flask app file
# type "flask run" in app.py's directory to run

import pandas as pd
from flask import Flask, jsonify, request, render_template
import pickle

### INPUT ###
runid = 'f71fb9cb2001496ba5cde6ce7a553bd3'
### ####

#load model
model = pickle.load(open(f'mlruns/1/{runid}/artifacts/model/model.pkl','rb'))
# scaler = pickle.load(open('scaler.pkl','rb'))

# app
app = Flask(__name__, template_folder='templates')

# routes
@app.route('/',methods=['GET','POST'])
def main():
    if request.method == 'GET':
        return(render_template('main.html'))
    if request.method == 'POST':
        # should match from the main.html form
        pclass = request.form['Pclass']
        age = request.form['Age']
        sibSp = request.form['SibSp']
        fare = request.form['Fare']        
        sex = request.form['Sex']
        
        if (sex == 'F'):
            sex1 = 0 
            sex0 = 1
        else :
            sex1 = 1 # 1 is male
            sex0 = 0
        
        
        input_variables = pd.DataFrame([[pclass, age, sibSp, fare, sex0, sex1]],
                columns=['Pclass', 'Age', 'SibSp','Fare', 'Sex-0','Sex-1'],
                dtype=int)        
        
        # input_variables = scaler.transform(input_variables) 
        
        prediction = model.predict(input_variables)[0]        
        
        return render_template('main.html',
                original_input={'Pclass': pclass,
                                'Age':age,
                                'Sibsp':sibSp,
                                'Fare': fare ,
                                'Sex:': sex},
                result=str(prediction)
        )

@app.route('/doc',methods=['GET'])
def doc():
    if request.method == 'GET':
        return(render_template('titanic-logistic.html')) # need to update this with every version update

# def predict():
#     # get data
#     data = request.get_json(force = True)
    
#     # convert to dataframe
#     data.update((x,[y]) for x , y in data.items())  
#     data_df = pd.DataFrame.from_dict(data)
    
#     # predictions
#     result = model.predict(data_df)
    
#     # send back to browser
#     output  = {'results': int(result[0])}
    
#     # return data
#     return jsonify(result=output)

if __name__ == '__main__':
    app.run()