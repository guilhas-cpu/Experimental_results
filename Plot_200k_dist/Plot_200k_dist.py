import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * np.exp(b * x) + c

f = 1e3
hscale = 2.50e-4*1e3

data_CH1 = pd.read_csv('F0021CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0021CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.685,0,0.1,0.2,0.3,0.4,0.5],[-0.685,0,1,2,3,4,5])
plt.title('Vout receiver module (Distance=36.5cm)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')


data_CH1 = pd.read_csv('F0022CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0022CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.568,0,1,2,3,4,5])
plt.title('Vout receiver module (Distance=35cm)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

data_CH1 = pd.read_csv('F0023CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0023CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.835,0,1,2,3,4,5])
plt.title('Vout receiver module (Distance=30cm)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')


data_CH1 = pd.read_csv('F0024CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0024CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-1.1,0,1,2,3,4,5])
plt.title('Vout receiver module (Distance=25cm)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

data_CH1 = pd.read_csv('F0025CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0025CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-1.792,-1,0,1,2,3,4,5])
plt.title('Vout receiver module (Distance=20cm)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

data_CH1 = pd.read_csv('F0026CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0026CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-2.919,-2,-1,0,1,2,3,4,5])
plt.title('Vout receiver module (Distance=15cm)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

data_CH1 = pd.read_csv('F0027CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0027CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-6.52,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5])
plt.title('Vout receiver module (Distance=10cm)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')


data_CH1 = pd.read_csv('F0028CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0028CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks(np.arange(-10,6,1))
plt.title('Vout receiver module (Distance=5cm)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
Distance = np.arange(5,36,5)
DeltaV = [0.568,0.820,1.08,1.7,2.8,6.4,10.4]
plt.plot(Distance,DeltaV,'o',label='Experimental Data')
xnew = np.linspace(5,35,100)
popt, pcov = curve_fit(func, Distance, DeltaV)
plt.plot(xnew,func(xnew,*popt),'r--',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.title('Vout vs Distance')
plt.ylabel('Signal(mV)')
plt.xlabel('Distance(cm)')
plt.legend()


plt.show()
