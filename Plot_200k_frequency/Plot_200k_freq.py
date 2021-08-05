import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

f = 250
hscale = 2.50e-3*1e3

data_CH1 = pd.read_csv('F0026CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0026CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-2.919,-2,-1,0,1,2,3,4,5])
plt.title('Vout receiver module (f=1kHz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

f = 2e3
hscale = 2.5e-4*1e3

data_CH1 = pd.read_csv('F0049CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0049CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-2.63,-2,-1,0,1,2,3,4,5])
plt.title('Vout receiver module (f=2kHz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

f = 3e3
hscale = 1e-4*1e3

data_CH1 = pd.read_csv('F0050CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0050CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
#plt.yticks([-2.77,-2,-1,0,1,2,3,4,5])
plt.title('Vout receiver module (f=3kHz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

f = 4e3
hscale = 1e-4*1e3

data_CH1 = pd.read_csv('F0051CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0051CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-2.73,-2,-1,0,1,2,3,4,5])
plt.title('Vout receiver module (f=4kHz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')


f = 5e3
hscale = 1e-4*1e3

data_CH1 = pd.read_csv('F0052CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0052CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-2.66,-2,-1,0,1,2,3,4,5])
plt.title('Vout receiver module (f=5kHz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

f = 6e3
hscale = 1e-4*1e3

data_CH1 = pd.read_csv('F0053CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0053CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-2.65,-2,-1,0,1,2,3,4,5])
plt.title('Vout receiver module (f=6kHz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

f = 7e3
hscale = 2.5e-4*1e3

data_CH1 = pd.read_csv('F0054CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0054CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
#plt.yticks([-2.64,-2,-1,0,1,2,3,4,5]
plt.title('Vout receiver module (f=7kHz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

f = 8e3
hscale = 1e-4*1e3

data_CH1 = pd.read_csv('F0055CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0055CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
#plt.yticks([-2.64,-2,-1,0,1,2,3,4,5])
plt.title('Vout receiver module (f=8kHz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

f = 9e3
hscale = 1e-4*1e3

data_CH1 = pd.read_csv('F0056CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0056CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-2.56,-2,-1,0,1,2,3,4,5])
plt.title('Vout receiver module (f=9kHz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

f = 10e3
hscale = 1e-4*1e3

data_CH1 = pd.read_csv('F0057CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0057CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
#plt.yticks([-2.53,-2,-1,0,1,2,3,4,5])
plt.title('Vout receiver module (f=10kHz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

f = 500
hscale = 5e-4*1e3

data_CH1 = pd.read_csv('F0058CH1.CSV', usecols=[4])
data_CH2 = pd.read_csv('F0058CH2.CSV', usecols=[4])

plt.figure()
plt.plot(data_CH1, label='Input')
plt.plot(data_CH2, label='Output')
plt.legend()
plt.xticks(np.arange(0,2501,250),np.arange(0,hscale*11,hscale))
plt.yticks([-2.73,-2,-1,0,1,2,3,4,5])
plt.title('Vout receiver module (f=500Hz)')
plt.ylabel('Signal(V)')
plt.xlabel('time(ms)')

plt.figure()
frequency = [500,1e3,2e3,3e3,4e3,5e3,6e3,7e3,8e3,9e3,10e3]
Vout = [2.7,2.72,2.7,2.7,2.6,2.56,2.56,2.4,2.3,2.25,2.1]
plt.plot(frequency,Vout,'*-')
plt.xticks(frequency,['500Hz','1kHz','2kHz','3kHz','4kHz','5kHz','6kHz','7kHz','8kHz','9kHz','10kHz'])
plt.ylabel('Signal(V)')
plt.xlabel('Frequency')
plt.title('Vout x frequency')
plt.show()
 
