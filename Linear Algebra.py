import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
a=np.mat('[1 2;3 4]')
print(a)
b=np.mat('[5 6]')
print(b)
print(b.T)
print(a*b.T)


print("2D matrix")
from scipy import linalg
AA=np.array([[1,2],[3,4]])
BB=np.mat([[5,8]])
print(linalg.inv(AA))

print("i am slow")
print(linalg.inv(AA).dot(AA))

print("\n")
print("i am check")
print(AA.dot(linalg.inv(AA).dot(AA))-BB)
print("\n")
print(BB.T)
print(AA.dot(BB.T))
print("\n")


print("new matrix example")
data=np.array([[1,2,3],[4,5,6],[8,5,6]])
print(data)
print(linalg.inv(data))


print("\n")
print(" i am double checking")
print(data.dot(linalg.inv(data)))
print("inverse Example")


print("\n")
newdata=np.array([[1,3,5],[2,5,1],[2,3,8]])
newdata1=np.array([[10],[8],[3]])
print(linalg.inv(newdata).dot(newdata1))
print(newdata.dot(linalg.inv(newdata).dot(newdata1))-newdata1)
print(np.linalg.solve(newdata,newdata1))

print("find det values")
one=np.array([[1,3],[4,5]])
print(one)
print(linalg.det(one))
newdatadet=np.array([[1,3,5],[2,5,1],[2,3,8]])
print(linalg.det(newdatadet))

print("computing norms")
print("\n")
comput=np.mat([[1,2],[4,8]])
print(linalg.norm(comput))
print(linalg.norm(comput,'fro'))  #default frobenius matrix
print(linalg.norm(comput,1))   # max column sum
print(linalg.norm(comput,-1))
print(linalg.norm(comput,np.inf))    # max row sum


print("Eigenvalues and eigenvectors")
print("\n")
Eig=np.array([[1,3],
              [4,6]])

la, v = linalg.eig(Eig)
l1, l2 = la
print(l1,l2)

print(v[:, 0])
print(v[:, 1])

print(np.sum(abs(v**2), axis=0))
v1 = np.array(v[:, 0]).T

print(linalg.norm(Eig.dot(v1) - l1*v1))
eig2=np.array([[1,5,2],[2,4,1],[3,6,2]])
w,new=linalg.eig(eig2)
l1,l2,l3=w
print(l1,l3,l2)
print(new[:,0])

print("Singular value decomposition")
print("\n")
svdex =np.array([[8,7,9],[2,1,3]])

M,N = svdex.shape

U,s,Vh = linalg.svd(svdex)

Sig = linalg.diagsvd(s,M,N)
U, Vh = U, Vh
print(U)
print(Sig)
print(Vh)
print(U.dot(Sig.dot(Vh)))

print("Schur decomposition")
print("\n")
AAA = np.mat('[1 3 2; 1 4 5; 2 3 6]')
T, Z = linalg.schur(AAA)
T1, Z1 = linalg.schur(AAA, 'complex')
T2, Z2 = linalg.rsf2csf(T, Z)
print(T)
print(T2)
print("\n")
print(abs(T1 - T2)) # different
print("\n")
print(abs(Z1 - Z2)) # different

T, Z, T1, Z1, T2, Z2 = map(np.mat,(T,Z,T1,Z1,T2,Z2))

print(abs(AAA - Z*T*Z.H))  # same
print("\n")
print(abs(AAA - Z1*T1*Z1.H))  # same
print(abs(AAA - Z2*T2*Z2.H))
print("\n")
hank=np.array([1,2,3])
print(linalg.hankel(hank))


print("vener input function")
print("\n")
x = np.array([1, 2, 3, 5])
N = 3
ex=np.vander(x, N)
print(ex)


"""print("Solving linear least-squares problems and pseudo-inverses")
print("\n")
c1,c2=5.0,2.0
i=np.r_[1:11]
xi=0.1*i
yi = c1*np.exp(-xi) + c2*xi
zi = yi + 0.05 * np.max(yi) * np.random.randn(len(yi))

A = np.c_[np.exp(-xi)[:, np.newaxis],xi[:, np.newaxis]]
c, resid, rank, sigma = linalg.lstsq(A, zi)

xi2 = np.r_[0.1:1.0:100j]
yi2 = c[0]*np.exp(-xi2) + c[1]*xi2

plt.plot(xi,zi,'x',xi2,yi2)
plt.axis([0,1.1,3.0,5.5])
plt.xlabel("$x_i$")
plt.title("Data fitting with linalg.lstsq")
plt.show()  """

