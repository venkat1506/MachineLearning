import Serializemultipul,pickle

f=open("multiple.pkl","rb")
print("Employee details")

while True:
    try:
        obj=pickle.load(f)
        obj.display()
        #obj3.dispaly()    --> you get EOF Error(there is no 3rd obj3 values) so you handle error using try ,except
        #if obj.eid==25:       # onely you want pirticular eid values
           # obj.display()
    except EOFError:
        print("all employees Compleated")
        break
f.close()