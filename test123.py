#!python.exe

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from datetime import datetime
from displayfunction import display
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae

import warnings
import cgi
import cgitb
warnings.filterwarnings('ignore')
import os
# In[3]:
print(os.path.expanduser('~'))
from pathlib import Path
print(Path.home())
os.environ['HOME'] = 'C:/xampp/htdocs/trial/'
print(Path.home())
print(os.environ['HOME'])

df = pd.read_csv('StoreDemand_tmp.csv')
x = df.describe()
print("<h2>", x, "</h2>")

# import cgi
# import cgitb
# import mysql.connector
# form=cgi.FieldStorage()
# firstname=form.getvalue('firstname')
# lastname=form.getvalue('lastname')
# pwd=form.getvalue('pwd')
# gender=form.getvalue('gender')
# day=form.getvalue('day')
# print("Content-type:text/html")
# print("")
# print("<html><head><title>First Page</title></head>")
# print("<body bgcolor='khaki'><h1>Welcome to Python..</h1><br>")
# print(firstname,lastname,pwd,gender,day)
# sql = "INSERT INTO login(firstname,lastname) values('"+firstname+"','"+lastname+"')"
# print(sql)
# mydb = mysql.connector.connect(host="localhost",user="root",password="",database="test")
# print("<h1>Connected...</h1>")
# mycursor = mydb.cursor()
# mycursor.execute(sql)
# mydb.commit()
# print("sql executed")
# print(mycursor.rowcount, "row inserted successfully")
#
# print("</body></html>")
