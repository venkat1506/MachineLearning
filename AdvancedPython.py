class Employee:
    def __init__(self,id,name,sal):
        self.id=id
        self.name=name
        self.sal=sal

    def Empdetyails(self):
        print('Employee id:' ,self.id)
        print('Employee name:' ,self.name)
        print("Employee Salary:" ,self.sal)

e1=Employee(100,'venky',45.000)
e2=Employee(101,'sree',54.588)
e1.Empdetyails()
e2.Empdetyails()