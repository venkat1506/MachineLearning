import pymysql
import pickle


db = pymysql.connect("127.0.0.1","root","root","Project_db" )
cursor = db.cursor()
cursor.execute("SELECT distinct * from EmpRecords")
myresult = cursor.fetchall()


print(myresult)

filename = 'Project_Emp.pkl'
outfile = open(filename,'wb')
pickle.dump(myresult,outfile)
outfile.close()
db.close()
