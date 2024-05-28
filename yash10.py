#!C:/Users/kedar/AppData/Local/Microsoft/WindowsApps/python3.9.exe
print("Content-type:text/html")
print("")
print("")
import mysql.connector
import cgi
import cgitb
form=cgi.FieldStorage()

firstname=form.getvalue('firstname')
lastname=form.getvalue('lastname')
pwd=form.getvalue('pwd')
gender=form.getvalue('gender')
day=form.getvalue('day')
sql="INSERT INTO login(firstname,lastname) values('"+firstname+"','"+lastname+"')"
print("")
print("")
print(sql)

mydb=mysql.connector.connect(host="localhost",user="root",password="",database="test")
mycursor=mydb.cursor()
mycursor.execute(sql)
mydb.commit()
print("sql executed")
print(mycursor.rowcount,"row inserted successfully")
