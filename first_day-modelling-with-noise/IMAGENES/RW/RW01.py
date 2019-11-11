import numpy as np
import matplotlib.pyplot as plt 
N=1000
T=1.0
delta = T/np.float(N)
h=1.0/np.sqrt(np.float(N))
t=np.linspace(0,T,N+1)
b= np.random.binomial(1,.5, N)	# bernulli 0,1
omega=2.0*b-1					# bernulli -1,1
Xn=h*(omega.cumsum())			# bernulli -h,h
Xn=np.concatenate(([0], Xn))
M = np.abs(Xn).max()+h
mu=Xn.mean()*np.ones(Xn.shape)
#
plt.plot(t,Xn,'k-o',alpha=0.8,lw=1,ms=4,mfc='green',label=r'$RW$')
plt.plot(t,mu,'r-',label=r'$E[X_n]$')
plt.xlabel(r'$transiciones$')
plt.ylabel(r'$X_n$')
plt.title(r'Construccion toerema Kuo')
#plt.title(r'Caminata del borracho')
plt.grid(True)
plt.legend(shadow=True,loc=0)
plt.show()
