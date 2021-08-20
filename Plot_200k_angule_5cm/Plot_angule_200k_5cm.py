import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b, c,d,e):
    return a*x**4+b*x**3+c*x**2+d*x+e

def gauss(x,A,mu,sigma):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def ang(x,a):
    return a*(np.cos(x))**4.2
#def func(x,a,b,c):
    #return a*x**2 +b*x+c

f = 1e3
hscale = 5e-4*1e3

data_CH1 = pd.read_csv('F0028CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0028CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks(np.arange(-10,6,1))
plt.title('Vout receiver module (Angule=0°) @5cm')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
data_CH1 = pd.read_csv('F0041CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0041CH2.CSV', usecols=[4])
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks(np.arange(-10,6,1))
plt.title('Vout receiver module (Angule=+5°) @5cm')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
data_CH1 = pd.read_csv('F0042CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0042CH2.CSV', usecols=[4])
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-1.75,-1,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule=+10°) @5cm')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
data_CH1 = pd.read_csv('F0044CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0044CH2.CSV', usecols=[4])
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.72,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule=+15°) @5cm')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
data_CH1 = pd.read_csv('F0045CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0045CH2.CSV', usecols=[4])
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks(np.arange(-10,6,1))
plt.title('Vout receiver module (Angule=-5°) @5cm')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
data_CH1 = pd.read_csv('F0046CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0046CH2.CSV', usecols=[4])
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks(np.arange(-10,6,1))
plt.title('Vout receiver module (Angule=-10°) @5cm')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
data_CH1 = pd.read_csv('F0047CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0047CH2.CSV', usecols=[4])
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-2.67,-2,-1,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule=-15°) @5cm')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
data_CH1 = pd.read_csv('F0048CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0048CH2.CSV', usecols=[4])
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-1.59,-1,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule=-20°) @5cm')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
angule = [-15,-10,-5,0,5,10,15]
signal = [-2000,-2400,-10000,-10000,-10200,-1760,-720]
popt, pcov = curve_fit(gauss, angule,signal)
xnew = np.linspace(-15,15,100)
plt.plot(angule,signal,'o',label='Experimental Data')
plt.plot(xnew,gauss(xnew,*popt),'r--',label='fit: A=%5.3f, mu=%5.3f, sigma=%5.3f' % tuple(popt))
#plt.plot(xnew,func(xnew,*popt),'r--',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.xticks([-15,-10,-5,0,5,10,15],['-15°','-10°','-5°','0°','5°','10°','15°'])
plt.ylabel('Signal(mV)')
plt.xlabel('Angule(°)')
plt.title('Angule Variation @5cm')
plt.legend()
plt.show()
