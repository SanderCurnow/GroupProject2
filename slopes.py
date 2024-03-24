import numpy as np
from scipy.sparse import diags
import scipy
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv

L = 120

E = np.array([5.9990014942022446e-05, 1.4997574020342644e-05, 3.7493963047784303e-06, 9.37347658046565e-07, 2.343352327445567e-07, 5.8582110175862956e-08, 1.4643828459733055e-08, 3.659254411200047e-09, 9.131406217104121e-10, 2.2655087658293716e-10, 5.5138092225695545e-11, 1.1314027189082698e-11, 7.844334340076364e-12])

KN = np.array([7.999424047228128, 31.997504197264455, 127.98982479690575, 511.9591071953506, 2047.8362367890943, 8191.344755165675, 32765.378828670782, 131061.51512229322, 524246.06031008903, 2096984.2409514687, 8387936.965198493, 33551747.834371936, 134206992.16914405])

k_list = np.arange(1,14,1)

n_list = np.power(2, k_list + 1)

h_list = L / n_list.astype(float)

print(k_list)
print(n_list)
print(h_list)

h_log = np.log(h_list)
E_log = np.log(E)
KN_log = np.log(KN)

dh = h_log[1:] - h_log[:-1]
dE = E_log[1:] - E_log[:-1]
dKN = KN_log[1:] - KN_log[:-1]
print(dE)
print(dh)

E_slopes = dE / dh
KN_slopes = dKN / dh

print(E_slopes)
print(KN_slopes)

plt.scatter(h_log, KN_log, color="red")

plt.xlabel('log(h)')
plt.ylabel('log(KN)')
plt.title('Log-Log Plot of KN vs h')
plt.grid(True)

plt.show()
