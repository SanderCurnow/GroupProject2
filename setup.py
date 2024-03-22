import numpy as np
from scipy.sparse import diags
import scipy
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
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

# Example usage
for k in range(1,3):
    n = 2 ** (k + 1) # Example number of intervals
    A, b = setup(n)
    np.set_printoptions(precision=20, suppress=False, threshold=200, linewidth=400, formatter={'float': '{: 0.20f}'.format})

    x = spsolve(A, b)

    true_x = np.vectorize(TrueSolution)((np.arange(1, n) * (L / n)))
    # print("True Solution true_x: ", true_x)

    num = np.linspace(0, L, n+1)

    ext_x = np.insert(x, [0, len(x)], 0.0)
    ext_true_x = np.insert(true_x, [0, len(true_x)], 0.0)

    if False:
        # Plot the first vector
        plt.plot(num, ext_x, label='w(x)', color='blue')
        # Plot the second vector
        plt.plot(num, ext_true_x, label='true w(x)', color='red')

        # Add annotations
        # for i in range(len(x)):
            # plt.text(x[i], y1[i], f'({x[i]}, {y1[i]:.2f})', fontsize=8, color='blue', ha='right')
            # plt.text(x[i], y2[i], f'({x[i]}, {y2[i]:.2f})', fontsize=8, color='red', ha='right')

        # Add legend
        plt.legend()

        # Add labels and title
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Plot of true and approximate deflection of beam (k = ' + str(k) + ", n = " + str(n) + ")")

        plt.show()
    else: 
        plt.clf()
        plt.plot(num, np.abs(ext_true_x - ext_x), label='error, |w(x) - true w(x)|', color='red')

        # Add legend
        plt.legend()

        # Add labels and title
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Plot of error of approximation (k = ' + str(k) + ", n = " + str(n) + ")")
        # print(len('Plot of error between true and approximate deflection of beam'))
        
    # plt.show()
        plt.savefig(f'error_{k}.png')






# # Example usage
# k = 2
# n = 2 ** (k + 1) # Example number of intervals
# A, b = setup(n)
# # print(type(A))
# np.set_printoptions(precision=20, suppress=False, threshold=200, linewidth=400, formatter={'float': '{: 0.20f}'.format})
# # print("A:", A.toarray())  # Convert sparse matrix A to a dense matrix for printing
# # print("b:", b)

# # The SuperLU object does not directly expose P, L, and U as separate attributes
# # But you can obtain L and U:

# # Note: There's no direct P matrix returned by splu. The permutation matrices can be accessed
# # through the `perm_c` and `perm_r` attributes for column and row permutations respectively
# # P, L, U = scipy.linalg.lu(A.toarray())
# # y = scipy.linalg.solve_triangular(L, np.dot(P, b), lower=True)
# # x = scipy.linalg.solve_triangular(U, y)

# x = spsolve(A, b)

# print("Solution x:", x)

# ## True solution ##

# L = 120
# q = 100/12
# E = 3e7
# S = 1000
# I = 625

# def TrueSolution(x):
#     return TrueSolutionWithParams(x, L, q, E, S, I)

# def TrueSolutionWithParams(x, L, q, E, S, I):
#     c = (-1*q*E*I) / (S**2)
#     a = math.sqrt(S / (E*I))
#     b = (-1*q)/(2*S)
#     c1 = c * (1 - math.exp((-1*a*L))) / (math.exp((-1*a*L)) - math.exp(a*L))
#     c2 = c * (math.exp(a*L) - 1) / (math.exp(-1*a*L) - math.exp(a*L))

#     return c1 * math.exp(a*x) + c2 * math.exp(-1*a*x) + b*x*(x-L) + c

# # for x in range(13):
# #     print(TrueSolution(x*10))

#     # Create a NumPy vector
# true_x = np.vectorize(TrueSolution)((np.arange(1, n) * (L / n)))
# print("True Solution true_x: ", true_x)

# # print (list(zip(x, true_x)))

# # num = np.arange(1, n, 1)

# num = np.linspace(0, L, n+1)
# # num = num / n

# ext_x = np.insert(x, [0, len(x)], 0.0)
# ext_true_x = np.insert(true_x, [0, len(true_x)], 0.0)

# if False:
#     # Plot the first vector
#     plt.plot(num, ext_x, label='w(x)', color='blue')
#     # Plot the second vector
#     plt.plot(num, ext_true_x, label='true w(x)', color='red')

#     # Add annotations
#     # for i in range(len(x)):
#         # plt.text(x[i], y1[i], f'({x[i]}, {y1[i]:.2f})', fontsize=8, color='blue', ha='right')
#         # plt.text(x[i], y2[i], f'({x[i]}, {y2[i]:.2f})', fontsize=8, color='red', ha='right')

#     # Add legend
#     plt.legend()

#     # Add labels and title
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('Plot of true and approximate deflection of beam (k = ' + str(k) + ", n = " + str(n) + ")")
# else: 
#     plt.plot(num, np.abs(ext_true_x - ext_x), label='error', color='red')

#     # Add legend
#     plt.legend()

#     # Add labels and title
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('Plot of error between true and approximate deflection of beam (k = ' + str(k) + ", n = " + str(n) + ")")

# plt.show()