import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

f = 250
hscale = 2.50e-3*1e3

data_CH1 = pd.read_csv('F0000CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0000CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.36,0,1,2,3,4,5])
plt.title('Vout receiver module (f=250Hz)')
plt.ylabel('Signal(V)'
plt.xlabel('time(ms)')

f = 500
hscale = 1e-3*1e3

data_CH1 = pd.read_csv('F0001CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0001CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*10+1e-3,hscale))
plt.yticks([-0.36,0,1,2,3,4,5])
plt.title('Vout receiver module (f=500Hz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

f = 750
hscale = 1e-3*1e3

data_CH1 = pd.read_csv('F0002CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0002CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*10+0.001,hscale))
plt.yticks([-0.36,0,1,2,3,4,5])
plt.title('Vout receiver module (f=750Hz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

f = 1e3
hscale = 1e-3*1e3

data_CH1 = pd.read_csv('F0003CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0003CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.36,0,1,2,3,4,5])
plt.title('Vout receiver module (f=1kHz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

f = 1.25e3
hscale = 5e-4*1e3

data_CH1 = pd.read_csv('F0004CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0004CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.36,0,1,2,3,4,5])
plt.title('Vout receiver module (f=1.25kHz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

f = 1.5e3
hscale = 5e-4*1e3

data_CH1 = pd.read_csv('F0005CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0005CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.36,0,1,2,3,4,5])
plt.title('Vout receiver module (f=1.5kHz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

f = 1.75e3
hscale = 2.5e-4*1e3

data_CH1 = pd.read_csv('F0006CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0006CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.36,0,1,2,3,4,5])
plt.title('Vout receiver module (f=1.75kHz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

f = 2e3
hscale = 2.5e-4*1e3

data_CH1 = pd.read_csv('F0007CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0007CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.36,0,1,2,3,4,5])
plt.title('Vout receiver module (f=2kHz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

f =10e3
hscale = 2.5e-5*1e3

data_CH1 = pd.read_csv('F0008CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0008CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-0.36,0,1,2,3,4,5])
plt.title('Vout receiver module (f=10kHz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')
plt.show()
 
