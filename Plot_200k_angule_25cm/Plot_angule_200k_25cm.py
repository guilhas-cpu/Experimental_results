import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b, c,d,e):
    return a*x**4+b*x**3+c*x**2+d*x+e

#def func(x,a,b,c):
    #return a*x**2 +b*x+c
def gauss(x,A,mu,sigma):
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

f = 1e3
hscale = 5e-4*1e3

data_CH1 = pd.read_csv('F0024CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0024CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-1.1,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule=0°)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
data_CH1 = pd.read_csv('F0035CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0035CH2.CSV', usecols=[4])
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.511,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule=+5°)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
data_CH1 = pd.read_csv('F0036CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0036CH2.CSV', usecols=[4])
plt.plot(data_CH1*1/10, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.092,0,0.1,0.2,0.3,0.4,0.5],[-0.092,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule=+10°)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
data_CH1 = pd.read_csv('F0037CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0037CH2.CSV', usecols=[4])
plt.plot(data_CH1*1/10, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.064,-0.0325,0,0.1,0.2,0.3,0.4,0.5],[-0.064,-0.0325,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule=+15°)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
data_CH1 = pd.read_csv('F0038CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0038CH2.CSV', usecols=[4])
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.57,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule=-5°)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
data_CH1 = pd.read_csv('F0039CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0039CH2.CSV', usecols=[4])
plt.plot(data_CH1*1/10, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.570,0,0.1,0.2,0.3,0.4,0.5],[-0.570,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule=-10°)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
data_CH1 = pd.read_csv('F0040CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0040CH2.CSV', usecols=[4])
plt.plot(data_CH1*1/10, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.126,0,0.1,0.2,0.3,0.4,0.5],[-0.126,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule=-15°)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')


plt.figure()
angule = [-15,-10,-5,0,5,10,15]
signal = [-104,-520,-860,-960,-540,-72,-40]
popt, pcov = curve_fit(gauss, angule,signal)
xnew = np.linspace(-15,15,100)
plt.plot(angule,signal,'o',label='Experimental Data')
plt.plot(xnew,gauss(xnew,*popt),'r--',label='fit: a=%5.3f, mu=%5.3f, sigma=%5.3f' % tuple(popt))
#plt.plot(xnew,func(xnew,*popt),'r--',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.xticks([-15,-10,-5,0,5,10,15],['-15°','-10°','-5°','0°','5°','10°','15°'])
plt.ylabel('Tensão de saída(mV)')
plt.xlabel('Angule(°)')
plt.title('Angule Variation @25cm')
plt.legend()
plt.show()

