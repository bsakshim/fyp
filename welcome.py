#!python.exe
#!C:/xampp/htdocs/trial/venv/Scripts/python.exe
import cgi, cgitb
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge


def weekend_or_weekday(year, month, day):
    #print(year,month,day)
    d = datetime(year, month, day)
    if d.weekday() > 4:
        return 1
    else:
        return 0


def which_day(year, month, day):
    #print(year,month,day)
    d = datetime(year, month, day)
    return d.weekday()


os.environ['HOME'] = 'C:/xampp/htdocs/trial/venv/'
print("Content-type:text/html")
print("")
print("<html><head><title>First Page</title></head>")
#print("<body bgcolor='khaki'><h1>Welcome to Python.....OK...@</h1>")4
print("<body bgcolor='Thistle'><h1>")
print("<br><h1 align='center'><a href='train_test_details.htm'>View Training Results</a></h1>")
df = pd.read_csv('StoreDemand_tmp.csv')
#x = df.describe()
#print("<h2>Hello...<br>", x)
form = cgi.FieldStorage()
icode = int(form.getvalue('item'))
monthnumber = int(form.getvalue('mo'))
#monthnumber=6
dayno = 5
curryear = 2024
storecode = 1
print("<h2 align='center'>Icode: ", icode, "<br>Month Number:", monthnumber,"<br>Year:", curryear, "<br>Storecode:", storecode,"<br>Dayno:",dayno)
weekend = weekend_or_weekday(curryear, monthnumber, dayno)
weekdayno = which_day(curryear, monthnumber, dayno)
print("<br>Weekend:", weekend)
print("<br>Weekdayno:", weekdayno)

p1 = 0
p2 = 0
#
arry = [storecode, icode, curryear, monthnumber, weekdayno, weekend, p1, p2]
parts = df["date"].str.split("-", n=3, expand=True)
df["year"] = parts[0].astype('int')
df["month"] = parts[1].astype('int')
df["day"] = parts[2].astype('int')
df.head()
df['weekend'] = df.apply(lambda x: weekend_or_weekday(x['year'], x['month'], x['day']), axis=1)
df.head()
df['m1'] = np.sin(df['month'] * (2 * np.pi / 12))
df['m2'] = np.cos(df['month'] * (2 * np.pi / 12))
df.head()
df['weekday'] = df.apply(lambda x: which_day(x['year'],
                                             x['month'],
                                             x['day']),
                         axis=1)
df.head()
df.drop('date', axis=1, inplace=True)
df['store'].nunique(), df['item'].nunique()
features = ['store', 'year', 'month',
            'weekday', 'weekend', ]
df = df[df['sales'] < 140]
features = df.drop(['sales', 'year'], axis=1)
target = df['sales'].values

X_train, X_val, Y_train, Y_val = train_test_split(features, target,
                                                  test_size=0.05,
                                                  random_state=22)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
models = [LinearRegression(), XGBRegressor(), Lasso(), Ridge()]
X, Y = make_regression(n_samples=1000, n_features=8, noise=0.1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Train the model
model = XGBRegressor()
#model = LinearRegression()
model.fit(X_train, Y_train)

print("<br><h2 align='center'>Arry: ", arry)
predictions = model.predict([arry])
print("<h1 align='center' style='color:red'>Predicted Value is: ", predictions)
print("</body></html?")

print("</body></html?")
