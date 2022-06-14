from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np


app = Flask(__name__)
model= pickle.load(open('model.pkl','rb'))

@app.route("/")
def Home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    
    a=request.form['Pclass']
    b=request.form['Age']
    c=request.form['SibSp']
    d=request.form['Parch']
    e=request.form['Fare']
    f=request.form['Sex_n']
    g=request.form['Embarked_n']
    arr=np.array([[a,b,c,d,e,f,g]])
    prediction=model.predict(arr)
    return render_template("index.html",data=prediction[0],)

if __name__ == "__main__":
    app.run(debug=False)