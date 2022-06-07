# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 23:01:24 2022

@author: Fujitsu-A556
"""

import numpy as np
import scipy.signal as sig

def schroeverb(x, fs, CF, cfgain, AP, apgain, fc=2500):
    """
    Filtra una señal con un reverberador de Schroeder.

    Parameters
    ----------
    x : numpy array
        señal a filtrar.
    fs : int
        frecuencia de muestreo.
    CF : int
        cantidad de veces que se aplica el comb filter.
    cfgain : float
        ganancia de los comb filter. Debe ser menor a 1 para que el sistema 
        sea estable.
    AP : int
        cantidad de veces que se aplica el all pass filter.
    apgain : float
        ganancia de los pasa todo. Debe ser menor a 1 para que el sistema
        sea estable.
    fc : float, optional
        Frecuencia de corte del filtro pasa bajos. The default is 2500.

    Returns
    -------
    y : numpy array.
        señal filtrada. es una señal 100% wet.

    """
    combsum = []    # Lista en la cual se añaden los filtros peine
    B, A = sig.butter(1, fc, fs=fs)    # define coeficientes del filtro pasa bajos
    
    for i in range(CF):
        cfdel = int(fs * np.random.uniform(35e-3, 50e-3))    # Define la demora de forma aleatoria
        y = iircomb(x, cfdel, cfgain)    # Filtra la señal
        y = sig.lfilter(B, A, y)    # Filtro PB a la salida de cada filtro peine
        combsum.append(y)    # Añade la señal filtrada a la lista
    y = np.sum(combsum, axis=0)/CF    # Output del bloque de combfilter
#     y = sig.lfilter(B,A,y)    # Filtro PB una sola vez luego de sumar los filtros peine
    for i in range(AP):
        apdel = int(fs * np.random.uniform(1.7e-3, 5e-3))    # Define la demora de forma aleatoria
        y = allpass(y, apdel, apgain)    # Filtra la señal
    return y

def iircomb(x, delay, gain):
    """
    Filtra una señal con un filtro comb de tipo IIR.

    Parameters
    ----------
    x : numpy array
        señal a filtrar.
    delay : int
        cantidad de muestras a retrasar.
    gain : float
        ganancia con que se atenúa. Debe ser gain < 1 para que el sistema
        sea estable.

    Returns
    -------
    numpy array
        señal filtrada.

    """
    b = np.zeros(delay)
    b[-1] = 1
    a = np.zeros(delay)
    a[0] = 1
    a[-1] = -gain
    return sig.lfilter(b, a, x)

def allpass(x, delay, gain):
    """
    Filtra una señal con un filtro pasa todo.

    Parameters
    ----------
    x : numpy array
        señal a filtrar.
    delay : int
        cantidad de muestras a retrasar.
    gain : float
        ganancia con que se atenúa. Debe ser gain < 1 para que el sistema
        sea estable.

    Returns
    -------
    numpy array
        señal filtrada.

    """
    b = np.zeros(delay)
    b[0] = gain
    b[-1] = 1
    a = np.zeros(delay)
    a[0] = 1
    a[-1] = gain
    return sig.lfilter(b, a, x)