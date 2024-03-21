# This fiel have I Write the commands needed to solve the system Ay = b using Gaussian Elimination (PA=LU).
import numpy as np
from scipy.sparse import diags
import math 
from numpy import log
import matplotlib.pyplot as pyp

def w(x):#true solution
  L = 120
  S = 1000
  q = 100/12
  E = 3.0e7
  I = 625
  c = -(q*E*I/(S**2))
  a = (S/(E*I))**(1/2)
  b = -(q/(2*S))
  c1 = c*(1-(math.exp(-a*L)))/(math.exp(-a*L)-math.exp(a*L))
  c2 = c*((math.exp(a*L))-1)/(math.exp(-a*L)-math.exp(a*L))
  ts = c1*math.exp(a*x)+ c2*math.exp(-a*x) +b*x*(x-L)+c #ts = true sol

  return ts


def setup(n):
    L = 120
    S = 1000
    q = 100/12
    E = 3.0e7
    I = 625
    Q = S / (E * I)
    R = q / (2 * E * I)
    h = L / n
    d = 2 + h**2 * Q
    n = n - 1
    e = np.ones(n)
    offsets = [-1, 0, 1]
    diagonals = [-e, d * e, -e]
    A = diags(diagonals, offsets, shape=(n, n)).toarray()
    r = lambda x: R * x * (x - L)
    x = np.linspace(h, L - h, n)
    b = -h * h * r(x)
    return A, b

def plu(A):
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n, dtype=np.double)
    P = np.eye(n, dtype=np.double)

    for i in range(n):
        for k in range(i, n):
            if ~np.isclose(U[i, i], 0.0):
                break
            U[[k, k+1]] = U[[k+1, k]]
            P[[k, k+1]] = P[[k+1, k]]

        factor = U[i+1:, i] / U[i, i]
        L[i+1:, i] = factor
        U[i+1:] -= factor[:, np.newaxis] * U[i]

    return L, U, P

def back_substitution(U, y):
    n = U.shape[0]
    x = np.zeros_like(y, dtype=np.double)

    for i in range(n-1, -1, -1):
        if U[i, i] == 0:
            x[i] = 0
            continue
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]

    return x

def forward_substitution(L, b):
    m = len(b)
    x = np.empty(m)

    for v in range(m):
        if L[v][v] == 0:
            x[v] = 0
            continue
        value = b[v]
        for i in range(v):
            value -= L[v][i] * x[i]
        value /= L[v][v]
        x[v] = value

    return x

def plu_solve(A, b):
    (L, U, P) = plu(A)
    b = np.matmul(P, b)
    y = forward_substitution(L, b)
    x = back_substitution(U, y)
    return x


# Main

errors = []
for k in range(1,2):
  n = 2**(1+k)
  A, b = setup(n)
  print("A:")
  print(A)
  print("b:")
  print(b)
  
  L, U, P = plu(A)
  print("\nL:\n", L)
  print("\nU:\n", U)
  print("\nP:\n", P)
  
  # x = np.dot(np.linalg.inv(A), b)
  # print("x (numpy solve):", x)
  
  x = plu_solve(A, b)
  print("x (PLU solve):", x)

  num_sol = np.linspace(0, L, n+1).reshape(-1, 1) # make sure the "'"  didnt need to be converted to python
  
  
  err = abs(num_sol - w)
  errors.append(err)

#n = 2^(k+1); 
#x4 = linspace(0,L,n+1)'
#% solve the system Ay=b using PA=LU factorization.
#% You have to write the correct sequence of commands here!
#y4 = [0;y;0];
#w4 = w(x4); 
#err4 = abs(y4-w4)
#w = w(x)

x = log(errors[:-1])
y = log(errors[1:])
dx = x[1:]-x[:-1]
dy = y[1:]-y[:-1]
slopes = [dy[i]/dx[i] for i in range(len(dx))]
print("slopes = ",slopes)
pyp.plot(x,y,"bo-")
pyp.xlabel("log(e_i)")
pyp.ylabel("log(e_{i+1})")
pyp.grid(True)
# This saves to a file
pyp.savefig("./LogErrorsPlot.png")
# This shows it on your screen
# pyp.show()
