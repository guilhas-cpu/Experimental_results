import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b, c,d,e):
    return a*x**4+b*x**3+c*x**2+d*x+e

def gauss(x,A,mu,sigma):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


f = 1e3
hscale = 5e-4*1e3

data_CH1 = pd.read_csv('F0026CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0026CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks(np.arange(-10,6,1))
plt.title('Vout receiver module (Angule=0°) @15cm')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')


plt.figure()
data_CH1 = pd.read_csv('F0029CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0029CH2.CSV', usecols=[4])
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-2.6,-2,-1,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule=+5°) @15cm')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
data_CH1 = pd.read_csv('F0030CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0030CH2.CSV', usecols=[4])
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.663,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule=+10°) @15cm')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
data_CH1 = pd.read_csv('F0031CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0031CH2.CSV', usecols=[4])
plt.plot(data_CH1*1/10, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.134,-0.038,0,0.1,0.2,0.3,0.4,0.5],[-0.134,0.038,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule=+15°) @15cm')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
data_CH1 = pd.read_csv('F0032CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0032CH2.CSV', usecols=[4])
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-2.56,-2,-1,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule=-5°) @15cm')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
data_CH1 = pd.read_csv('F0033CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0033CH2.CSV', usecols=[4])
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.61,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule=-10°) @15cm')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
data_CH1 = pd.read_csv('F0034CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0034CH2.CSV', usecols=[4])
plt.plot(data_CH1*1/10, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.111,0,0.1,0.2,0.3,0.4,0.5],[-0.111,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule=-15°) @15cm')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
angule = [-15,-10,-5,0,5,10,15]
signal = [-92,-580,-2520,-2800,-2600,-660,-105]
popt, pcov = curve_fit(gauss, angule,signal)
xnew = np.linspace(-15,15,100)
plt.plot(angule,signal,'o',label='Experimental Data')
plt.plot(xnew,gauss(xnew,*popt),'r--',label='fit: a=%5.3f, mu=%5.3f, sigma=%5.3f' % tuple(popt))
#plt.plot(xnew,func(xnew,*popt),'r--',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.xticks([-15,-10,-5,0,5,10,15],['-15°','-10°','-5°','0°','5°','10°','15°'])
plt.ylabel('Tensão de saída(V)')
plt.xlabel('Angule(°)')
plt.title('Angule Variation @15cm')
plt.legend()
plt.show()

