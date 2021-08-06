import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b):
    return a*x+b


current = [2.01,4,6.06,8.03,10.14,12.03]

Tension = [1.025, 1.407,1.798,2.17,2.57,2.98]

popt,pcov=curve_fit(func,Tension,current)
xnew = np.linspace(1.025,2.98,100)

plt.plot(xnew,func(xnew,*popt),'r--',label='fit: a=%5.3f, b=%5.3f' % tuple(popt))

plt.plot(Tension,current,'o',label='Experimental Data')
plt.title('Tension vs Current (DC Sweep)')
plt.yticks(current)
plt.xticks(Tension)
plt.xlabel('Current(mA)')
plt.ylabel('Tension(mV)')
plt.legend()

plt.show()
