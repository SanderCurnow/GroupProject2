import argparse
import math
import numpy as np

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

for x in range(13):
    print(TrueSolution(x*10))

