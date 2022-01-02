
"""
Created on Wed Sep 15 20:37:29 2021
Master in finance 
@author: federicoromero
subject: financial engineering 


"""

# Valuación de una opción europea con el metodo Black Scholes y MonteCarlo

import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm


def valor_call_europeo(S0, K, r, T, sigma, *args):
    """
    Precio de un Call europeo por Black-Scholes 
    
    Parámetros:
       S: Precio subyacente
       K: Precio de ejercicio
       T: Tiempo al vencimiento
       r: Tasa libre de riesgo 
    sigma : Volatilidad
    *args : Argumentos extras (extra arguments can be passed, but are not used)

    """
    
    discountedStrike = np.exp(-r * T) * K
    totalVolatility = sigma * np.sqrt(T)
    
    d_minus = np.log(S0 / discountedStrike) / totalVolatility - .5 * totalVolatility
    d_plus = d_minus + totalVolatility
    
    return S0 * norm.cdf(d_plus) - discountedStrike * norm.cdf(d_minus), 0.

import time
inicio = time.time()

# Código a medir
time.sleep(1)
def Valor_call_europeo_MC(S0,K,r,T,sigma,M):
    """
    Precio de un Call europeo por el método de MonteCarlo
    
    Parámetros:
       S: Precio subyacente
       K: Precio de ejercicio
       T: Tiempo al vencimiento
       r: Tasa libre de riesgo 
    sigma : Volatilidad
        M : Número de simulaciones
 

     precio_MC : estimación a partir de MC del precio de la opción en el modelo Black Scholes
     desvest_MC : estimación a partir de MC del desvío estándar
    """  
    
    # genera M muestras con una distribución N(0,1)
    X = np.random.randn(M)
    
    
    # simula M trayectorias en un paso
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * X)
    # define el payoff
    payoff = np.where(ST < K, 0, ST - K) # funciona como max(ST-K, 0) 
    
    # estimación MC
    discountFactor = np.exp(-r*T)
    
    precio_MC = discountFactor*np.mean(payoff)
    desvest_MC = discountFactor*np.std(payoff)/np.sqrt(M)
    return precio_MC, desvest_MC

precio_exacto = valor_call_europeo(40, 50, 0.02, 1, .3)[0]
print("precio exacto:", precio_exacto, "\n")
for M in [100, 10000000, 100000000]:
    precio_MC, std_mc = Valor_call_europeo_MC(40, 50, 0.02, 1, .3, M)
    print("M = ", M)
    print("Precio estimado: ", precio_MC)
    print("Error absoluto en el precio: ", np.abs(precio_MC - precio_exacto), "\n")
fin = time.time()
print("Tiempo de ejecución: ", fin-inicio)

def grafico(pricingMethod, *args):
    S0 = 40
    S = np.linspace(150, 1e-15, num=500)
    T = np.array([1])
    K = 50
    sigma = 0.3
    r = 0.02
    
    price = dict((t, np.array([pricingMethod(s, K, r, t, sigma, *args)[0] for s in S])) for t in T)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    for t in T:
        ax.plot(S, price[t], 'bo', label='T = '+str(t))
    ax.set_xlabel('Precio subyacente')
    ax.set_ylabel('valor del Call')
    ax.set_title('Precio del call europeo como función del subyacente')
    plt.legend(loc='best')
    plt.show()

grafico(valor_call_europeo, 100)
