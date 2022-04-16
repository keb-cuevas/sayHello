from flask import Flask, render_template, request
import pandas as pd
from numpy import float16

#do not show warnings
import warnings
warnings.filterwarnings("ignore")

#import machine learning related libraries
import xgboost as xgb

from sklearn.model_selection import train_test_split
import joblib

path = 'seeds_dataset.txt'

df = pd.read_csv(path, sep= '\t', header= None,
                names=['area','perimeter','compactness','lengthOfKernel',
                       'widthOfKernel','asymmetryCoefficient',
                      'lengthOfKernelGroove','seedType'])

df = df.dropna()
df.info()

X= df.drop('seedType', axis = 1)
y= df['seedType']



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.30, random_state=42)

xgb_model = xgb.XGBClassifier().fit(X_train, y_train)


joblib.dump(xgb_model, "xgb.pkl") #export ML model to pkl file

app = Flask(__name__)
app.secret_key = "ASDFGHJ"

@app.route('/', methods=['GET', 'POST'])

def text():
  if request.method == 'POST':
    
    xgb = joblib.load("xgb.pkl")
    # Get values through input bars
    area = request.form.get("area")
    perimeter = request.form.get("perimeter")
    compactness = request.form.get("compactness")
    lengthOfKernel = request.form.get("lengthOfKernel")
    widthOfKernel = request.form.get("widthOfKernel")
    asymmetryCoefficient = request.form.get("asymmetryCoefficient")
    lengthOfKernelGroove = request.form.get("lengthOfKernelGroove")

    # Put inputs to dataframe
    X = pd.DataFrame([[area, perimeter, compactness, lengthOfKernel, 
                       widthOfKernel, asymmetryCoefficient, lengthOfKernelGroove]], 
                     columns = ["area", "perimeter", "compactness", "lengthOfKernel", 
                                "widthOfKernel", "asymmetryCoefficient", "lengthOfKernelGroove"])
    X = X.astype(float16)

    # Get prediction
    predict = xgb.predict(X)[0]

    if predict == 1.0:
      prediction = "Kama"
    
    elif predict == 2.0:
      prediction = "Rosa"

    elif predict == 3.0:
      prediction = "Canadian"

    else:
      prediction = "Error"

  else:
    prediction = 'Unknown'

  return render_template('text.html', output = prediction)

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app.
    app.run(host='127.0.0.1', port=8080, debug=True)
