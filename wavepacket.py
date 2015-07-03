#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time

zeta = 0.3 
npoints = 1000
xdist = 50
xpoints,spacing = np.linspace(-xdist,xdist,npoints,True,True)

# Gaussian wavepacket
psi_x = np.exp(-zeta*np.multiply(xpoints,xpoints)) 
psi_p_initial = np.fft.fft(psi_x)
freqs = np.fft.fftfreq(npoints,spacing)

# Propagate - set particle mass magnitue equal to 2*hbar for simplicity
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
psi_x_line, = ax.plot(xpoints,np.square(psi_x),color="blue")
for time_step in np.arange(0,1000,1):
    psi_p = np.multiply(np.exp(-1j*np.square(freqs)*time_step),psi_p_initial)
    psi_x = np.fft.ifft(psi_p)
    psi_x_line.set_ydata(np.square(np.absolute(psi_x)))
    fig.canvas.draw()
    time.sleep(0.01)
