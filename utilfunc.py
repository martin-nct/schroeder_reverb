#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Funciones útiles

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def plot_sig(x, fs=1, title='Señal temporal', tipo='plot', l=16, a=7):
    """
    Grafica una señal en el dominio temporal. 

    Parameters
    ----------
    x : 1darray
        array de una dimensión que se desea graficar.
    fs : int, opcional
        frecuencia de muestreo de x. Por defecto es 1.
    title : str, opcional
        el título del plot. Por defecto es 'Señal temporal'.

    Returns
    -------
    Figure : 
        Gráfico 2D de x.
    """
    
    t = np.linspace(0, x.size/fs, x.size)
    plt.figure(figsize=(l,a))
    if tipo == 'plot':
        plt.plot(t, x)
    elif tipo == 'stem':
        plt.stem(t,x)
    plt.title(title)
    plt.xlabel('Tiempo[s]')
    plt.ylabel('Amplitud')
    plt.grid()
    plt.show()
    
def stft(x,fs,largo,solap):
    '''
    Calcula la STFT de una señal.
    
    Subdivide la señal y la multiplica por ventanas de tipo Hann, 
    aplica la FFT a cada ventana y devuelve información tiempo-frecuencia
    de la señal original.
    
    Parameters
    ----------
    x: 1darray
        señal temporal
    fs: int
        frecuencia de sampleo
    largo: float
        duración de cada ventana en segundos
    solap: float
        superposicion entre ventanas en segundos
    
    Returns
    -------
    X: list
        STFT de la señal. 
    '''
    N = int(largo*fs)     # Largo de ventana en muestras
    P = int(solap*fs)      # Paso en muestras
    n = np.arange(N)
    hann = 0.5 - 0.5*np.cos(2*np.pi*n/N)    # Ventana Hann
    X = []
    for i in range(0,x.size-N, P):
        y = x[i:i+N]*hann
        Y = np.fft.rfft(y)
        X.append(Y)
    return X

def plot_stft(STFT, fs, T, title='STFT'):

    '''
    Grafica una STFT.
    
    Devuelve un plot donde el eje horizontal es el tiempo y
    el eje vertical es la frecuencia. La amplitud se representa
    con pseudocolor cmap = 'magma'.
    
    Parameters
    ----------
    STFT: list
        Lista con la STFT a graficar
    fs: int
        Frecuencia de muestreo de la señal
    T: float
        Duración en segundos de la señal
    title: str, opcional
        Título del gráfico. Por defecto "'STFT'"
    '''

    STFT_MAG = np.asarray(np.abs(STFT))
    f = np.linspace(0, fs/2, STFT_MAG.shape[1])
    t = np.linspace(0, T, STFT_MAG.shape[0])
    plt.figure(1,figsize=(20,10))
    plt.pcolormesh(t, f, STFT_MAG.T, cmap='magma')
    plt.ylabel('Frecuencia (Hz)')
    plt.xlabel('Tiempo (s)')
    plt.title(title)
    plt.show()
    
def plot_stft2(STFT, f, t, title='STFT'):
    
    '''
    Grafica la STFT. Utilizar para STFT calculada con la función de scipy.
    
    Devuelve un plot donde el eje horizontal es el tiempo y
    el eje vertical es la frecuencia. La amplitud se representa
    con pseudocolor cmap = 'magma'.
    
    Parameters
    ----------
    STFT: 2Darray
        Array con la STFT a graficar
    f: array
        Eje frecuencial
    t: array
        Eje temporal
    title: str
        Título del gráfico. Por defecto 'STFT'
    '''
    plt.figure(1,figsize=(20,10))
    plt.pcolormesh(t, f, np.abs(STFT), cmap='magma')
    plt.ylabel('Frecuencia (Hz)')
    plt.xlabel('Tiempo (s)')
    plt.title(title)
    plt.show()

def ssf(x, fs, largo=0.01, solap=None, silencio=0.3, b=1):
    
    '''
    Reducción de ruido por substracción espectral.
    
    Aplica reducción de ruido por el método de substracción espectral a una señal.
    Requiere que la señal no tenga información útil en el inicio. Procesa la señal
    con reducción de ruido residual.
    
    Parameters
    ----------
    x: 1darray 
        Señal a filtrar
    fs: int 
        Frecuencia de muestreo
    largo: float, opcional
        Longitud en segundos de la ventana para realizar la STFT. Por defecto largo = 0.01 seg
    solap: float, opcional
        El solapamiento en segundos entre ventana y ventana en la STFT. Por defecto es la mitad del largo. Debe ser menor
        al largo.
    silencio: float, opcional
        Tiempo en segundos al inicio de la señal donde se estimará el ruido. Se requiere que no haya información útil.
    b: float, opcional
        Parámetro de sobre-sustracción. Usualmente 1<b<2. Por defecto b=1.
    
    Returns
    -------
    tiempo: array
        Vector de tiempo de la señal
    salida: array
        Señal filtrada
    '''
    
    if solap != None:
        solap = int(solap*fs)    # Pone el paso en muestras
    f, t, X = sig.stft(x, fs, nperseg=int(largo*fs), noverlap=solap)    # Calcula STFT
    X = X.T    # Trasponer la matriz
    
    ventanas = []
    fases = []
    for i in range(len(X)):
        ventanas.append(np.abs(X[i]))
        fases.append(np.angle(X[i]))
    
    ruido = np.array(ventanas[0])
    muestras = int(silencio * fs)    # Cantidad de muestras sin señal útil
    M = int(muestras/len(ventanas[0]))
    
    for i in range(1,M):
        ruido += ventanas[i]
    ruido /= M    # Estimador de ruido
    
    ruido *= b    # Oversubstract
    
    for i in range(len(ventanas)):
        for k in range(len(ventanas[i])):
            if ventanas[i][k] > ruido[k]:
                ventanas[i][k] -= ruido[k]
            else: ventanas[i][k] = 0
    
    # Reducción de ruido residual:
    
    maxres = ventanas[0]            
    for i in range(1,M):
        for k in range(len(ventanas[i])):
            if ventanas[i][k] > maxres[k]:
                maxres[k] = ventanas[i][k]    # Selecciona las magnitudes máximas en el silencio inicial luego de la substracción
    
    for i in range(1,len(ventanas)-1):
        for k in range(len(ventanas[i])):
            if ventanas[i][k] < maxres[k]:
                ventanas[i][k] = min(ventanas[i+1][k], ventanas[i][k], ventanas[i-1][k])
                # Elige el valor mínimo en ventanas adyacentes para esa frecuencia
    
    
    out = []
    for i in range(len(ventanas)):
        out.append(ventanas[i] * np.exp(1j*fases[i]))    # Agregamos la magnitud y la fase
    out = np.asarray(out)
    
    tiempo, salida = sig.istft(out.T,fs, noverlap=solap)     # Antitransformamos
    return tiempo, salida

def SNR(x, sigma=None, inicio=0, fin=100):
    '''
    Calcula la relación señal a ruido (SNR).
    
    Calcula la SNR de una señal con media 0según una porción
    de señal donde se estima la desviación estandar. Se define como:
    SNR = RD{|x|}/sigma donde RD es el rango dinámico.
    
    Parameters
    ----------
    x : 1darray 
        Señal a calcular SNR.
    sigma : float
        Desviación estandar del ruido. Por defecto es None
    inicio : int, opcional
        index desde el cual se calcula la desviación estándar. Por defecto es 0.
    fin : int, opcional
        index hasta el cual se calcula la desviación estándar. Por defecto es 100.
    
    Returns
    -------
    SNR: float
        Relación señal a ruido de x.    
    '''
    if sigma==None:
        sigma = np.std(x[inicio:fin])
    return np.around((np.max(abs(x)) - np.min(abs(x))) / sigma, 3)


def polosyceros(b,a,s=7,lim=1.1):
    '''
    Grafica polos y ceros.
    
    Devuelve un gráfico de polos y ceros y sus valores a partir de los
    coeficientes de la función de transferencia. Los coeficientes deben
    estar ordenados en potencias decrecientes de z.
    
    Parameters
    ----------
    b : array-like, list 
        Coeficientes del numerador de la función de transferencia H(z) ordenados
        en potencias decrecientes de z.
    a : array-like, list
        Coeficientes del denominador de la función de transferencia H(z) ordenados
        en potencias decrecientes de z.
    lim : float, opcional
        límites del gráfico. Por defecto es 1.1. El gráfico tendrá las mismas
        proporciones en ambos ejes.
    
    Returns
    -------
    Figure:
        Gráfico 2D
    Print:
        Polos y ceros.
    '''
    z, p, k = sig.tf2zpk(b, a)
    print('Ceros:', np.around(z,5))
    print('Polos:', np.around(p,5))
    tita = np.linspace(0,2*np.pi, 100)
    x = np.cos(tita)
    y = np.sin(tita)

    plt.figure(figsize=(s,s))
    plt.plot(x, y, '--k')
    for i in z:
        plt.scatter(i.real,i.imag, c='b', marker='o')
    for i in p:
        plt.scatter(i.real, i.imag, c='r', marker='x')
    plt.grid(linestyle='--')
    plt.xlabel('Re{z}')
    plt.ylabel('Im{z}')
    plt.title('Diagrama de Polos y Ceros')
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])
    plt.show()
    
def plot_magyfas(w, H, l=15, a=5, g_d=False):
    H_mag = np.abs(H)
    H_fase = np.angle(H)

    plt.figure(1, figsize=(l,a))
    plt.plot(w, H_mag, 'r')
    plt.title('Magnitud de H(w)')
    plt.xlabel('Frecuencia (rad/s)')
    plt.ylabel('Magnitud')
    plt.grid()
    plt.show()

    plt.figure(1, figsize=(l,a))
    plt.plot(w, H_fase, 'g')
    plt.title('Fase de H(w)')
    plt.xlabel('Frecuencia (rad/s)')
    plt.ylabel('Fase (rad/s)')
    plt.grid()
    plt.show()
    
    if g_d==True:
        w, GD = sig.group_delay((b,a))
        plt.figure(figsize=(l,a))
        plt.plot(w, GD, 'k')
        plt.title('Retardo de Grupo de H(w)')
        plt.xlabel('Frecuencia normalizada (rad/s)')
        plt.ylabel('Retardo (muestras)')
        plt.grid()
        plt.show()