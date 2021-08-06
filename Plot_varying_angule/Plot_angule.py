import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * x**2+ b*x+c

f = 1e3
hscale = 2.50e-4*1e3

data_CH1 = pd.read_csv('F0014CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0014CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.460,0,0.1,0.2,0.3,0.4,0.5],[-0.460,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule = 0°)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

data_CH1 = pd.read_csv('F0015CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0015CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.352,0,0.1,0.2,0.3,0.4,0.5],[-0.352,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule = +5°)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

data_CH1 = pd.read_csv('F0016CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0016CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.303,0,0.1,0.2,0.3,0.4,0.5],[-0.303,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule = +10°)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

data_CH1 = pd.read_csv('F0017CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0017CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.073,0,0.1,0.2,0.3,0.4,0.5],[-0.073,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule = +15°)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')


data_CH1 = pd.read_csv('F0018CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0018CH2.CSV', usecols=[4])
plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.336,0,0.1,0.2,0.3,0.4,0.5],[-0.336,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule = -5°)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

data_CH1 = pd.read_csv('F0019CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0019CH2.CSV', usecols=[4])
plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.224,0,0.1,0.2,0.3,0.4,0.5],[-0.224,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule = -10°)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

data_CH1 = pd.read_csv('F0020CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0020CH2.CSV', usecols=[4])
plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.046,0,0.1,0.2,0.3,0.4,0.5],[-0.046,0,1,2,3,4,5])
plt.title('Vout receiver module (Angule = -15°)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
angule = [-15,-10,-5,0,5,10,15]
signal = [-46,-224,-336,-460,-352,-303,-73]

plt.plot(angule,signal,'o',label='Experimental Data')

xnew = np.linspace(-15,15,100)
popt, pcov = curve_fit(func, angule,signal)
plt.plot(xnew,func(xnew,*popt),'r--',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.xticks([-15,-10,-5,0,5,10,15],['-15°','-10°','-5°','0°','5°','10°','15°'])
plt.ylabel('Signal(mV)')
plt.xlabel('Angule(°)')
plt.title('Angule Variation')
plt.legend()
plt.show()
