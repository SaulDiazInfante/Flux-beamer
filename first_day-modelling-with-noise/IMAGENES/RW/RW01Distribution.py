import numpy as np
import matplotlib.pyplot as plt 
def rw_path(N=1000, T=1.0):
	delta = T/np.float(N)
	h=1.0/np.sqrt(np.float(N))
	b= np.random.binomial(1,.5, N)	# bernulli 0,1
	omega=2.0*b-1					# bernulli -1,1
	Xn=h*(omega.cumsum())			# bernulli -h,h
	Xn=np.concatenate(([0], Xn))
	return Xn

N = 1000
N_paths = 100
T = 1.0
h = 1.0/np.sqrt(np.float(N))
t = np.linspace(0,T,N+1)
Xn = rw_path()
mu = np.zeros(Xn.shape[0])
for i in np.arange(N_paths-1):
	plt.plot(t,Xn,'g-',alpha=0.3,lw=1,ms=4,mfc='green')
	Xn=rw_path()
	mu = mu + Xn
mu = 1.0 / np.float(N_paths) * mu
plt.plot(t,Xn,'g-',alpha=.8,lw=1,ms=4,mfc='green',label=r'$RW$')
plt.plot(t,mu,'r-',label=r'$E[X_n]$')
'''
plt.plot(t,np.sqrt(t),'-o',ms=2, color='yellow',mec='none',label=r'$\sigma$')
plt.plot(t, 2 * np.sqrt(t),'-o',ms=2,mfc='yellow',mec='none')
plt.plot(t, 3 * np.sqrt(t),'-o',ms=2,mfc='yellow',mec='none')
plt.plot(t, -1 * np.sqrt(t),'-o',ms=2,mfc='yellow',mec='none')
plt.plot(t, -2 * np.sqrt(t),'-o',ms=2,mfc='yellow',mec='none')
plt.plot(t, -3 * np.sqrt(t),'-o',ms=2,mfc='yellow',mec='none')
'''
M = np.abs(Xn).max()+h
#
plt.xlabel(r'$t_n$')
plt.ylabel(r'$X_n$')
plt.title(r'Construccion toerema Kuo')

plt.grid(True)
plt.legend(shadow=True,loc=0)
plt.show()
