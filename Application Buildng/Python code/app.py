import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
 
app = Flask(__name__)  # initializing the app
model = pickle.load(open('CKD.pkl','wb')) #loading the model

@app.route('/') # route to display the home page
def home():
    return render_template('Home.html') # rendering the home page

@app.route('/prediction',methods=['POSt','GET'])
def prediction(): # route to display prediction page
    return render_template('indexnew.html')

@app.route('/Home',methods=['POST','GET'])
def my_home():
    return render_template('Home.html') # rendering the home page

@app.route('/predict',methods=['POST']) # route to show the prediction in web UI     
def predict():
    # reading the inputs given by the user
    inputs_features = [float(x) for x in request.form.values()] 
    features_value = [np.array(inputs_features)]

    features_name = ['pus_cell','blood glouse random','blood urea','red_blood_cells','diabetesmellitus','coronary_artery_disease','pedal_edema','anemia']
     
    df = pd.DataFrame(features_value, columns=features_name)

    output = model.predict(df) # prediction using the loaded model file
    b=[1]
    # showing the prediction results in a UI
    if(output==b):
        return render_template("Result2.html")
    else:
        return render_template("Result.html")

    # showing the prediction results in a UI
    

if __name__ == "__main__":
    app.run(debug=True) # running the app



