import time
import pandas as pd
import numpy as np

prasenttime=time.time()
print(prasenttime)


prasenttime=time.gmtime()
print(prasenttime)

print (time.asctime(prasenttime))
print (time.ctime())
time.sleep(2)
print (time.ctime())



import datetime

dti = pd.to_datetime(['JAN/1/2018']),datetime.datetime(2018, 1 , 1)

print(dti)

dti = pd.date_range('2018-JAN-01', periods=3, freq='D')  #S,H,D,M,Y
print(dti)


#localize time
dti = dti.tz_localize('UTC')
print(dti)

#conversion
convortion=dti.tz_convert('US/Pacific')
print(convortion)



#Day name
day_name = pd.Timestamp('06/15/1995')  #2000/15-JAN,1995-10-10,06/15/1998,.....All types are except

name=day_name.day_name()
print(name)
b=pd.date_range('95/JAN/15','DEC/29/95', freq='BM')
print(b)


s = "02202012102826"
s_datetime = datetime.datetime.strptime(s, '%m%Y%d%H%M%S')
print(s_datetime)