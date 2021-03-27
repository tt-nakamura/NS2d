# reference:
#   I. Danaila, et al.
#    "An Introduction to Scientific Computing" chapter 12

import numpy as np
from numpy.fft import rfft2,irfft2,fftfreq,rfftfreq

class NavierStokes2D:
    def __init__(self,Re,u0,v0,Pe,chi0,LX=1,LY=1,CFL=0.2):
        """ incompressible Navier-Stokes equation in two dimension
        periodic boundary condition is assumed for both x and y
        Re = Reynolds number
        u0,v0 = 2D arrays of initial velocity field
        Pe = Peclet number for diffusion of passive tracer
        chi0 = 2D arryas of initial tracer density
        LX,LY = lengths of period in x,y direction
        CFL = Courant-Friedrichs-Levy condition < 1
        """
        global R,dt,dx,dy,u,v,Hu,Hv,i,j,ip,im,jp,jm,L,L1
        global P,chi

        NX,NY = u0.shape
        dx = LX/NX
        dy = LY/NY

        i,j = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
        ip = np.roll(i,-1,axis=0)
        im = np.roll(i, 1,axis=0)
        jp = np.roll(j,-1,axis=1)
        jm = np.roll(j, 1,axis=1)

        u = u0.copy()
        v = v0.copy()
        Hu,Hv = convec()
        dt = CFL/np.max(np.abs(u/dx) + np.abs(v/dy))
        R,P = Re,Pe
        chi = chi0.copy()

        Lx = (2*np.sin(np.pi* fftfreq(NX))/dx)**2
        Ly = (2*np.sin(np.pi*rfftfreq(NY))/dy)**2
        L = Lx[:,np.newaxis] + Ly
        L1 = L.copy()
        L1[0,0] = 1

    def velocity(self): return u,v
    def tracer(self): return chi

    def vorticity(self):
        return (v[ip,j] - v)/dx - (u[i,jp] - u)/dy

    def run(self,t):
        """ run simulation for time t """
        while t>0:
            update(min(t,dt))
            t -= dt


def laplacian(f):
    return ((f[im,j] - 2*f + f[ip,j])/dx**2 +
            (f[i,jm] - 2*f + f[i,jp])/dy**2)

def convec():
    """ convection terms """
    ue = (u + u[ip,j])/2
    uw = (u + u[im,j])/2
    du2_dx = (ue**2 - uw**2)/dx

    vn = (v + v[i,jp])/2
    vs = (v + v[i,jm])/2
    dv2_dy = (vn**2 - vs**2)/dy

    un = (u + u[i,jp])/2
    us = (u + u[i,jm])/2
    vn = (v[i,jp] + v[im,jp])/2
    vs = (v + v[im,j])/2
    duv_dy = (un*vn - us*vs)/dy

    ue = (u[ip,j] + u[ip,jm])/2
    ve = (v + v[ip,j])/2
    duv_dx = (ue*ve - us*vs)/dx

    Hu = -du2_dx - duv_dy
    Hv = -duv_dx - dv2_dy
    return Hu,Hv

def ConvecChi():
    """ convection terms for passive tracer """
    ce = (chi + chi[ip,j])/2
    cw = (chi + chi[im,j])/2
    cn = (chi + chi[i,jp])/2
    cs = (chi + chi[i,jm])/2
    dcu_dx = (ce*u[ip,j] - cw*u)/dx
    dcv_dy = (cn*v[i,jp] - cs*v)/dy
    return -dcu_dx - dcv_dy

def update(dt):
    global u,v,Hu,Hv,chi
    Su = -Hu/2
    Sv = -Hv/2
    Hu,Hv = convec()
    Su += 1.5*Hu + laplacian(u)/R
    Sv += 1.5*Hv + laplacian(v)/R
    H = np.ones_like(L) + L*dt/R/2
    u += irfft2(rfft2(Su*dt)/H)
    v += irfft2(rfft2(Sv*dt)/H)
    div_q = (u[ip,j] - u)/dx + (v[i,jp] - v)/dy
    phi = irfft2(rfft2(div_q)/L1)
    u += (phi - phi[im,j])/dx
    v += (phi - phi[i,jm])/dy

    # update density of passive tracer
    chi += (ConvecChi() + laplacian(chi)/P)*dt
