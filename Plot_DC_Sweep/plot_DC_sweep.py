import matplotlib.pyplot as plt


current = [2.01,4,6.06,8.03,10.14,12.03]

Tension = [1.025, 1.407,1.798,2.17,2.57,2.98]

plt.plot(Tension,current,'-*')
plt.title('Tensios vs Current (DC Sweep)')
plt.yticks(current)
plt.xticks(Tension)
plt.xlabel('Current(mA)')
plt.ylabel('Tension(mV)')

plt.show()
