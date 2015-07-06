#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time

zeta = 0.3 
npoints = 1000
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
fig0 = plt.figure()
ax0 = fig0.add_subplot(131)
ax0.set_xlim([-xdist,xdist])
psi_x_line, = ax0.plot(xpoints,np.square(psi_x),color="blue")

# Momentum-space plot
ax1 = fig0.add_subplot(132)
ax1.set_xlim([-10,10])
psi_p_line, = ax1.plot(2*np.pi*freqs,(1/mag)*np.square(np.absolute(psi_p_initial)),color="red")

# Analytic momentum space plot
ax2 = fig0.add_subplot(133)
psi_p_analytic_line, = ax2.plot(ppoints,np.square(np.absolute(psi_p_analytic)),color="green")
psi_p_analytic_line.set_ydata(np.square(np.absolute(psi_p_analytic)))


# Fourier representation periodicity plot 
# Six periodic instances of the Gaussian
ax3 = fig0.add_subplot(211)
periodic_interval =  np.linspace(-6*xdist,6*xdist,6*npoints,True)
all_waves_at_all_points = np.exp(2*np.pi*1j*np.outer(freqs,periodic_interval))
psi_x_periodic = 1.0/(len(psi_p_initial))*np.dot(psi_p_initial,all_waves_at_all_points)
psi_x_periodic_line, = ax3.plot(periodic_interval,np.square(np.absolute(psi_x_periodic)),color="purple")

#psi_x_periodic = x_norm* 
#fig0.tight_layout()
fig0.subplots_adjust(wspace=0.1,hspace=0.15)
# Propagate - set particle mass magnitue equal to 2*hbar for simplicity
for time_step in np.arange(0,1000,1):
    psi_p = np.multiply(np.exp(-1j*np.square(freqs)*time_step),psi_p_initial)

    psi_x = np.fft.ifft(psi_p)
    psi_x_line.set_ydata(np.square(np.absolute(psi_x)))

    # Update momentum-space representation  
    #psi_p_analytic_line.set_ydata(np.square(np.absolute(psi_p_analytic)))
    mag = np.trapz(np.multiply(np.absolute(psi_p),np.absolute(psi_p)),2*np.pi*freqs)
    psi_p_line.set_ydata((1/mag)*np.square(np.absolute(psi_p)))

    fig0.canvas.draw()
    time.sleep(0.01)
