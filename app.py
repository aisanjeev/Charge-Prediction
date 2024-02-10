from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open('Insurance.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_insurance():
    age = int(request.form.get('age'))
    sex = int(request.form.get('sex'))
    bmi = int(request.form.get('bmi'))
    children = int(request.form.get('children'))
    smoker = int(request.form.get('smoker'))


    # prediction
    result = model.predict(np.array([age,sex,bmi,children,smoker]).reshape(1,5))
    fin_result = float(round(result[0],2))

    return render_template('index.html',fin_result=fin_result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
