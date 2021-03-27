import numpy as np
import matplotlib.pyplot as plt
from NavierStokes2D import NavierStokes2D

CFL = 0.4
Re = 1000
Pe = 1000
LX = 1
LY = 1
NX = 64
NY = 64

# vortex parameters
psi0 = [0.1, -0.1, -0.1, 0.1]
x0 = [LX/4, LX/4, 3*LX/4, 3*LX/4]
y0 = [LY/2+0.05, LY/2-0.05, LY/2+0.05, LY/2-0.05]
l0 = [0.1*LX]*4

x,y = np.meshgrid(np.linspace(0,LX,NX,endpoint=False),
                  np.linspace(0,LY,NY,endpoint=False),
                  indexing='ij')

# initial condition for vortex dipole
u = np.zeros_like(x)
v = np.zeros_like(y)
for i in range(len(psi0)):
    psi = psi0[i]*np.exp(-((x-x0[i])**2 + (y-y0[i])**2)/2/l0[i]**2)
    u += -(y-y0[i])/l0[i]**2*psi
    v +=  (x-x0[i])/l0[i]**2*psi

#initial condition for passive tracer
chi = np.where(np.logical_and(x>LX/3, x<2*LX/3), 1., 0.)

ns = NavierStokes2D(Re,u,v,Pe,chi,LX,LY,CFL)

plt.figure(figsize=(4, 6.4))
plt.subplots_adjust(left=0.14, right=0.98,
                    bottom=0.08, top=0.95,
                    wspace=0.05, hspace=0.2)

t = [0, 0.07, 1.35, 2.05]
for i,dt in enumerate(np.diff(t)):
    ns.run(dt)
    for j in range(2):
        plt.subplot(3,2,2*i+j+1)
        plt.axis('equal')
        plt.axis([x[0,0],x[-1,0],y[0,0],y[0,-1]])
        if j==0: plt.title(r'vorticity ($t={:.2f})$'.format(t[i+1]))
        else:    plt.title(r'tracer ($t={:.2f})$'.format(t[i+1]))
        if i==2: plt.xlabel(r'$x$')
        else:    plt.xticks([])
        if j==0: plt.ylabel(r'$y$')
        else:    plt.yticks([])
        if j==0: z,cm = ns.vorticity(), 'RdBu'
        else:    z,cm = ns.tracer(), 'jet'
        plt.contourf(x,y,z,20,cmap=cm)

plt.savefig('fig4.eps')
plt.show()
