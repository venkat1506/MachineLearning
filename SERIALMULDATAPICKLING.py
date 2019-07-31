import Serializemultipul,pickle
f= open("multiple.pkl","wb")
n=int(input("enter no of Employees"))
for i in range(n):
    eno=int(input("enter employee no"))
    ename=input("enther employee name")
    esal=float(input("enter employee salary"))
    eloc=input("enter employee location")
    e=Serializemultipul.Employee(eno,ename,esal,eloc)
    pickle.dump(e,f)