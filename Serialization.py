import pickle
class Employee:
    def __init__(self,eid,ename,esal,eloc):
        self.eid=eid
        self.ename=ename
        self.esal=esal
        self.eloc=eloc
    def display(self):
        print(self.eid ,"\t",self.ename,"\t",self.esal,"\t",self.eloc)

with open("pickleingExample.pkl","wb") as f:
    e=Employee(100,'venky',10000,'hyderabad')
    pickle.dump(e,f)
    print("pickling is compleated ")

