
"""
Created on Wed Sep 15 20:37:29 2021
master in finance
@author: federicoromero
subject: financial engineering 
"""

# Valuación de una opción europea con el metodo implicito de diferencias finitas


import numpy as np
import scipy.linalg as linalg

S = 40
Smax = 120
K = 50
r = 0.02
T = 1
sigma = 0.3
es_call = True
M = 100
N = 100

def valor_opcion_europea(S, K, r, T, sigma, Smax, M, N, es_call=True):
    '''Calcula el valor de la opcion usando el
    metodo de diferencias finitas
    
    Parametros
    ----------
    S: Precio subyacente
    K: Precio de ejercicio
    T: Tiempo al vencimiento
    r: Tasa libre de riesgo 
    sigma: volatilidad 
    Smax: maximo Precio subyacente
    M: numero de particiones del activo
    N: numero de particiones del tiempo
    es_call: define si es opcion call o put
    
    Resultado
    -------
    V : Valor de la opcion
    '''
    M, N = int(M), int(N)  

    dS = Smax / float(M)
    dt = T / float(N)

    grid = np.zeros((M+1, N+1))
    iValues = np.arange(M)
    jValues = np.arange(N)
    SValues = np.linspace(0, Smax, M+1)

    if es_call == True:
        '''
        Condiciones iniciales opcion Call:
        C(0,t) = 0
        C(Smax,t) = Smax - K e^(-r(T-t))
        C(S,T) = Smax - K
        '''
        grid[:,-1] = np.maximum(SValues - K, 0)
        grid[-1,:-1] = Smax - K * np.exp(-r * dt * (N - jValues))
    else:
        '''
        Condiciones iniciales:
        P(0,t) = K e^(-r(T-t))
        P(Smax,t) =  0
        P(S,T) = K - Smax
        '''
        grid[:, -1] = np.maximum(K - SValues, 0)
        grid[0, :-1] = K * np.exp(-r * dt * (N - jValues))

    alpha =  0.5 * dt * iValues * (r  - sigma**2 * iValues)
    beta  =  1 + dt * (r + sigma**2 * iValues**2)
    gamma = -0.5 * dt * iValues * (r  + sigma**2 * iValues)

    coeffs = np.diag(alpha[2:M], -1) + np.diag(beta[1:M]) + np.diag(gamma[1:M-1], 1)

    P, L, U = linalg.lu(coeffs)
    aux = np.zeros(M-1)

    for j in reversed(range(N)):
        aux[0] = np.dot(alpha[1], grid[0, j])
        aux[-1] = np.dot(-gamma[M-1], grid[M, j])
        x1 = linalg.solve(L, grid[1:M, j+1]+aux)
        x2 = linalg.solve(U, x1)
        grid[1:M, j] = x2
    
    V = np.interp(S, SValues, grid[:, 0])
    return V

call_value = valor_opcion_europea(S, K, r, T, sigma, Smax, M, N, es_call=True)
put_value = valor_opcion_europea(S, K, r, T, sigma, Smax, M, N, es_call=False)
print(' valor del call :', call_value)
print(' valor del put :', put_value)