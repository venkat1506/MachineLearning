import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial as p
from numpy.polynomial.polynomial import polyval
poly = p([1,2,3])
print(poly)
print(poly.coef)
print(poly.window)

print(poly.domain)
print(poly+poly)
print(poly-poly)
print(poly*poly)
print(poly**3)


a=poly // p([-1, 1])
print(a)
b=poly % p([-1,1])
print(b)

print("\n")
quo, rem = divmod(poly, p([-1, 1]))
print(quo)
print(rem)
print("\n")


print("Elevation")
x=np.arange(5)
print(poly(x))

y=np.arange(6).reshape(3,2)
print(y)
print(poly(y))

print(p(poly))
print(poly.roots())

from numpy.polynomial import Chebyshev as T
#print(p + poly([1],domain=[0,1]))# you gor error because Traceback(most recent call last) and different dommine name
print("\n ")
print("Chebyshev ")
print(poly(T([0, 1])))
print("calculus")
print("\n")
newdata =p([4,6])
print(newdata)
print(newdata.integ())
print(newdata.integ(1))
print(newdata.integ(lbnd=-1))
print(newdata.integ(lbnd=-1, k=1))
print("\n")
exa=p([1,2,3])
print(exa.deriv(1))

import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev as T
x = np.linspace(-1, 1, 100)
#x= np.linspace(-2,2,100)
for i in range(5):
    ax = plt.plot(x, T.basis(i)(x), lw=2, label="$T_%d$"%i)
plt.legend()
plt.show()



np.random.seed(11)
x = np.linspace(0, 2*np.pi, 20)
y = np.sin(x) + np.random.normal(scale=.1, size=x.shape)
p = T.fit(x, y, 5)
plt.plot(x, y, 'o')
xx, yy = p.linspace()
plt.plot(xx, yy, lw=2)
p.domain
p.window
plt.show()
print("polyvalues finding ")
polyval(1,[1,2,3])

from scipy.stats import alpha
a = 3.57
mean, var, skew, kurt = alpha.stats(a, moments='mvsk')
