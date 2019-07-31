class Employee:
    def __init__(self,eid,ename,esal,eloc):
        self.eid=eid
        self.ename=ename
        self.esal=esal
        self.eloc=eloc

    def display(self):
        print(self.eid,"\t",self.ename,"\t",self.esal,"\t",self.eloc)