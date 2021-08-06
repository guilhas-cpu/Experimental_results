import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * np.exp(-b * x) + c


f = 1e3
hscale = 2.50e-4*1e3

data_CH1 = pd.read_csv('F0009CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0009CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-1.585,0,1,2,3,4,5])
plt.title('Vout receiver module (Distance=2cm)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

data_CH1 = pd.read_csv('F0010CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0010CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.600,0,1,2,3,4,5])
plt.title('Vout receiver module (Distance=4cm)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')


data_CH1 = pd.read_csv('F0011CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0011CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.320,0,1,2,3,4,5])
plt.title('Vout receiver module (Distance=6cm)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')



data_CH1 = pd.read_csv('F0012CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0012CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.182,0,1,2,3,4,5])
plt.title('Vout receiver module (Distance=8cm)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')


data_CH1 = pd.read_csv('F0013CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0013CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.111,0,1,2,3,4,5])
plt.title('Vout receiver module (Distance=10cm)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
in_current = [47.5,14,8.6,5.1,3.3]
Vout = [-1550,-576,-336,-176,-100]
plt.plot(in_current,Vout,'-*')
plt.title('Vout vs Input Current')
plt.ylabel('Signal(mV)')
plt.xlabel('Current(uA)')

plt.figure()
Distance = np.arange(2,11,2)
DeltaV = [1550,576,336,176,100]

xnew = np.linspace(2,10,100)
popt, pcov = curve_fit(func, Distance,DeltaV)
plt.plot(xnew,func(xnew,*popt),'r--',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.plot(Distance,DeltaV,'o',label='Experimental Data')
plt.title('Vout vs Distance')
plt.ylabel('Signal(mV)')
plt.xlabel('Distance(cm)')
plt.legend()

plt.show()
