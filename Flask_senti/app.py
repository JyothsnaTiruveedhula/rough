from flask import Flask,render_template,request
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# New import for TensorFlow 2.x
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
# If you need to inverse transform, you can use scaler.inverse_transform(scaled_data)


# loading the label encoder 
#le=pickle.load(open('label_encoder.pkl','rb'))

# loading my mlr model
model=pickle.load(open('modelk.pkl','rb'))

#loading Scaler
#scalar=pickle.load(open('scaler.pkl','rb'))


# Flask is used for creating your application
# render template is use for rendering the html page

app= Flask(__name__)  # your application


@app.route('/')  # default route 
def home():
    return render_template('home.html') # rendering if your home page.

@app.route('/pred',methods=['POST']) # prediction route
def predict1():
    '''
    For rendering results on HTML 
    '''
    
    c = request.form["Comment"]
    #
   # t = [[float(rd),float(ad),float(ms),float(s),float(p)]]
   # x=scalar.fit_transform(t)
    output =model.predict(c)
    print(output)
    return render_template("home.html", result = "The predicted Emotion is  "+str(output))
    # running your application
if __name__ == "__main__":
    app.run()
#http://localhost:5000/ or localhost:5000
