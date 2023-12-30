import numpy as np
import scipy.integrate as integrate
import math
import matplotlib.pyplot as plt
# In[1]:
# We investigate the orthogonality of the Legendre polynomials in the interval [-1,1]
n=10
coeff= np.zeros(n+1)
coeff[n] = 1
n2=9
coeff2= np.zeros(n2+1)
coeff2[n2] = 1
x=np.linspace(-0.999999, 0.99999, 1000)
L1=np.polynomial.legendre.Legendre(coeff, domain=[-1,1])
L2=np.polynomial.legendre.Legendre(coeff2, domain=[-1,1])
integrand = lambda x: L1(x)*L2(x)
# Integrate the product of the two Legendre polynomials over the interval [-1, 1]
result, error = integrate.quad(integrand, -1, 1)
print(result)
#The result is 0, so the Legendre polynomials are orthogonal in the interval [-1,1]


# In[2]:
# We investigate the orthonormality of the Legendre polynomials in the interval [-1,1]
x=np.linspace(-0.999999, 0.99999, 1000)
L=np.polynomial.legendre.Legendre(coeff, domain=[-1,1])
integrand = lambda x: L(x)*L(x)
# Integrate the square of the Legendre polynomial over the interval [-1, 1]
result, error = integrate.quad(integrand, -1, 1)
print(result)
#The result is 0.09523809523809523, so the Legendre polynomials seems to be not orthonormal in the interval [-1,1]

# In[3]:
# We investigate the orthonormality of the Legendre polynomials in the interval [0,1]
x=np.linspace(0.00001, 0.99999, 1000)
L=np.polynomial.legendre.Legendre(coeff, domain=[0,1])
# We try to normalize the Legendre polynomial
L_normalized=np.polynomial.legendre.Legendre(coeff, domain=[0,1])*np.sqrt(2*n+1)
L_normalized2=np.polynomial.legendre.Legendre(coeff2, domain=[0,1])*np.sqrt(2*n2+1)
# First we check if they remain orthogonal
integrand = lambda x: L_normalized(x)*L_normalized2(x)
result, error = integrate.quad(integrand, 0, 1)
print("Integral=", result)
#The result is in the order of 1e-16, so they are orthogonal in the interval [0,1]

# In[4]:
# then we check the orthonormality
integrand = lambda x: L(x)*L(x)
integrand2 = lambda x: L_normalized(x)*L_normalized(x)
result, error = integrate.quad(integrand, 0, 1)
result2, error2 = integrate.quad(integrand2, 0, 1)
print("Integral=", result)
print("Integral with normalization=", result2)
#The result of the integral2 is 1, so the normalization seems to work



