#!/usr/bin/env python
#
# This script demonstrates the periodic nature of the wavefunction produced by time propagation using the Fourier transform
#
# by Samuel Manzer (samuelmanzer.com)

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import time

zeta = 0.05
npoints = 500
xdist = 50
xpoints,spacing = np.linspace(-xdist,xdist,npoints,True,True)
pdist = 10
ppoints,pspacing = np.linspace(-pdist,pdist,npoints,True,True)

# Gaussian wavepacket
x_norm = np.sqrt(1/np.sqrt(np.pi/(2*zeta)))
psi_x = x_norm*np.exp(-zeta*np.multiply(xpoints,xpoints)) 
psi_p_analytic = np.sqrt(1/np.sqrt(2*np.pi*zeta))*np.exp(-(1.0/(4*zeta))*np.multiply(ppoints,ppoints))
psi_p_initial = np.fft.fftshift(np.fft.fft(psi_x))
freqs = np.fft.fftshift(np.fft.fftfreq(npoints,spacing))
# Manually normalize what comes out of the FFT, though we know what analytic form should be
mag = np.trapz(np.multiply(np.absolute(psi_p_initial),np.absolute(psi_p_initial)),2*np.pi*freqs)

plt.ion()
# Position-space plot
fig = plt.figure()
ax0 = plt.subplot2grid((2,3),(0,0))
ax0.set_title("Position-Space |Wavefunction|^2",fontsize=8)
ax0.set_xlim([-xdist,xdist])
psi_x_line, = ax0.plot(xpoints,np.square(np.absolute(psi_x)),color="blue")

# Momentum-space plot
ax1 = plt.subplot2grid((2,3),(0,1))
ax1.set_title("Momentum-Space |Wavefunction|^2 (FFT)",fontsize=8)
ax1.set_xlim([-10,10])
psi_p_line, = ax1.plot(2*np.pi*freqs,(1/mag)*np.square(np.absolute(psi_p_initial)),color="red")

# Analytic momentum space plot
ax2 = plt.subplot2grid((2,3),(0,2))
ax2.set_title("Momentum-Space |Wavefunction|^2 (Exact)",fontsize=8)
psi_p_analytic_line, = ax2.plot(ppoints,np.square(np.absolute(psi_p_analytic)),color="green")

# Fourier representation periodicity plot 
# Six periodic instances of the Gaussian
# A bit tricky because origin is at the center of our interval, so need to add appropriate phase
ax3 = plt.subplot2grid((2,3),(1,0),colspan=3)
ax3.set_title("Fourier Transform Periodicity",fontsize=10)
periodic_interval =  np.linspace(-6.5*xdist,6.5*xdist,7*npoints,True)
phase = np.array([xpoints[0]] * len(periodic_interval)) 
all_waves_at_all_points = np.exp(2*np.pi*1j*np.outer(freqs,periodic_interval - phase))
ax3.set_xlim([-6.5*xdist,6.5*xdist])
psi_x_periodic = 1.0/(len(psi_p_initial))*np.dot(psi_p_initial,all_waves_at_all_points)
psi_x_periodic_line, = ax3.plot(periodic_interval,np.square(np.absolute(psi_x_periodic)),color="purple")


plt.tight_layout()

# Propagate - set particle mass magnitue equal to hbar^2/2 for simplicity
# Time is in increments scaled by hbar
for time_step in np.arange(0,40,0.2):
    psi_p = np.multiply(np.exp(-1j*np.square(2*np.pi*freqs)*time_step),psi_p_initial)

    psi_x = np.fft.ifft(np.fft.ifftshift(psi_p))
    psi_x_line.set_ydata(np.square(np.absolute(psi_x)))

    # Update momentum-space representation - for free particle will not change
    mag = np.trapz(np.multiply(np.absolute(psi_p),np.absolute(psi_p)),2*np.pi*freqs)
    psi_p_line.set_ydata((1/mag)*np.square(np.absolute(psi_p)))

    # Update periodic repr
    all_waves_at_all_points = np.exp(2*np.pi*1j*np.outer(freqs,periodic_interval - phase))
    psi_x_periodic = 1.0/(len(psi_p))*np.dot(psi_p,all_waves_at_all_points)
    psi_x_periodic_line.set_ydata(np.square(np.absolute(psi_x_periodic)))

    plt.draw()
plt.ioff()
plt.show()
