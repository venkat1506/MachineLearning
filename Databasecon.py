import csv
import MySQLdb

mydb = MySQLdb.connect(host='127.0.0.1',
    user='root',
    passwd='1234',
    db='webserices',
    autocommit=True)
cursor = mydb.cursor()

csv_data = csv.reader(('F:\CustInputData.csv'))
for row in csv_data:
    cursor.execute('INSERT INTO Emp(id,deptname,gender,age, jdate ) VALUES("%s", "%s", "%s","%s", "%s")',row)
#close the connection to the database.
mydb.commit()
cursor.close()
print("Done")
