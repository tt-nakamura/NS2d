import numpy as np
import matplotlib.pyplot as plt
from NavierStokes2D import NavierStokes2D

CFL = 0.1
Re = 1000
Pe = 1000
LX = 2
LY = 1
NX = 128
NY = 64

U0 = 1
Pj = 20
Rj = LY/4
Lj = LX/4
A0 = 0.5

x,y = np.meshgrid(np.linspace(0,LX,NX,endpoint=False),
                  np.linspace(0,LY,NY,endpoint=False),
                  indexing='ij')

#initial condition for Kelvin-Helmholtz instability
u = (U0/2*(1 + np.tanh(Pj/2*(1 - np.abs(LY/2-y)/Rj)))
     *(1 + A0*np.sin(2*np.pi*x/Lj)))
v = np.zeros_like(u)

#initial condition for passive tracer
chi = np.where(np.logical_and(y>LY/4, y<3*LY/4), 1., 0.)

ns = NavierStokes2D(Re,u,v,Pe,chi,LX,LY,CFL)

plt.figure(figsize=(7.4, 3.2))
plt.subplots_adjust(left=0.08, right=0.98,
                    bottom=0.14, top=0.92,
                    wspace=0.05, hspace=0.24)

t = [0, 0.15, 0.45, 0.85]
for i,dt in enumerate(np.diff(t)):
    ns.run(dt)
    for j in range(2):
        plt.subplot(2,3,i+3*j+1)
        plt.axis('equal')
        plt.axis([x[0,0],x[-1,0],y[0,0],y[0,-1]])
        if j==0: plt.title(r'vorticity ($t={:.2f})$'.format(t[i+1]))
        else:    plt.title(r'tracer ($t={:.2f})$'.format(t[i+1]))
        if j==1: plt.xlabel(r'$x$')
        else:    plt.xticks([])
        if i==0: plt.ylabel(r'$y$')
        else:    plt.yticks([])
        if j==0: z,cm = ns.vorticity(), 'RdBu'
        else:    z,cm = ns.tracer(), 'jet'
        plt.contourf(x,y,z,20,cmap=cm)

plt.savefig('fig2.eps')
plt.show()
