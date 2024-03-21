import numpy as np
from scipy.sparse import diags
import scipy
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix


def setup(n):
    # Constants
    L = 120
    S = 1000
    q = 100 / 12
    E = 3.0e7
    I = 625
    Q = S / (E * I)
    R = q / (2 * E * I)
    
    # Discretization
    h = L / n
    d = 2 + h**2 * Q
    n = n - 1
    
    # Diagonal elements
    e = np.ones(n)
    A = diags([-e, d*e, -e], offsets=[-1, 0, 1], shape=(n, n)).tocsc()
    
    # Load vector
    def r(x):
        return R * x * (x - L)
    
    x = np.linspace(h, L - h, n)
    b = -h**2 * r(x)
    
    return A, b

# Example usage
n = 10 # Example number of intervals
A, b = setup(n)
np.set_printoptions(precision=20, suppress=False, threshold=200, linewidth=400, formatter={'float': '{: 0.20f}'.format})
# print("A:", A.toarray())  # Convert sparse matrix A to a dense matrix for printing
# print("b:", b)

# The SuperLU object does not directly expose P, L, and U as separate attributes
# But you can obtain L and U:

# Note: There's no direct P matrix returned by splu. The permutation matrices can be accessed
# through the `perm_c` and `perm_r` attributes for column and row permutations respectively
P, L, U = scipy.linalg.lu(A.toarray())
y = scipy.linalg.solve_triangular(L, np.dot(P, b), lower=True)
x = scipy.linalg.solve_triangular(U, y)

print("Solution x:", x)

## True solution ##

L = 120
q = 100/12
E = 3e7
S = 1000
I = 625

def TrueSolution(x):
    return TrueSolutionWithParams(x, L, q, E, S, I)

def TrueSolutionWithParams(x, L, q, E, S, I):
    c = (-1*q*E*I) / (S**2)
    a = math.sqrt(S / (E*I))
    b = (-1*q)/(2*S)
    c1 = c * (1 - math.exp((-1*a*L))) / (math.exp((-1*a*L)) - math.exp(a*L))
    c2 = c * (math.exp(a*L) - 1) / (math.exp(-1*a*L) - math.exp(a*L))

    return c1 * math.exp(a*x) + c2 * math.exp(-1*a*x) + b*x*(x-L) + c

# for x in range(13):
#     print(TrueSolution(x*10))

    # Create a NumPy vector
true_x = np.vectorize(TrueSolution)((np.arange(1, n) * (L / n)))
print("True Solution true_x: ", true_x)

# print (list(zip(x, true_x)))

num = np.arange(1, n, 1)

# Plot the first vector
plt.plot(num, x, label='x', color='blue')

# Plot the second vector
plt.plot(num, true_x, label='true_x', color='red')

plt.show()