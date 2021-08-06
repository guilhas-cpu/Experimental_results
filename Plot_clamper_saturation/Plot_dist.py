import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

f = 1e3
hscale = 2.50e-4*1e3

data_CH1 = pd.read_csv('F0059CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0059CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.title('Vout receiver module (Distance=25cm)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')


data_CH1 = pd.read_csv('F0060CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0060CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.title('Vout receiver module (Distance=15cm)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

data_CH1 = pd.read_csv('F0061CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0061CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.title('Vout receiver module (Distance=5cm)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')


data_CH1 = pd.read_csv('F0062CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0062CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.title('Vout receiver module (Distance=30cm)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')



plt.show()
