#!/usr/bin/env python
#
#
#
# by Samuel Manzer (samuelmanzer.com)

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import time

# Need a much larger interval than what we plot to contain fourier noise
sigma2 = 100
npoints = 10000
xdist = 1000
xpoints,spacing = np.linspace(-xdist,xdist,npoints,True,True)

# Gaussian wavepacket with initial momentum
k_init = 1
x_norm = np.power(np.pi*sigma2,-0.25)
x_init = 200
psi_x = x_norm*np.exp(1j*k_init*(xpoints+x_init))*np.exp(-np.multiply((xpoints+x_init),(xpoints+x_init))/(2*sigma2))
freqs = np.fft.fftshift(np.fft.fftfreq(npoints,spacing))

# Heaviside potential barrier
v_x = .8*np.minimum(np.sign(xpoints)+1,np.repeat(1,xpoints.size))
kappa_init = np.sqrt(k_init*k_init - np.amax(v_x))

plt.ion()

# Position-space plot
fig,ax0 = plt.subplots()
psi_x_line, = ax0.plot(xpoints,np.square(np.absolute(psi_x)),color="blue")
ax0.set_title("Position-Space |Wavefunction|^2",fontsize=16)
v_x_line, = ax0.plot(xpoints,v_x,color="red")
ax0.set_xlim([-300,300])
ax0.set_ylim([0,0.1])

R_t1 = np.power((k_init - kappa_init)/(k_init + kappa_init),2)
R_t2 = ((2*k_init/np.power(kappa_init,3)) + 8/np.power(kappa_init,2))*R_t1*1.0/(sigma2)
T_t1 = (4*k_init*kappa_init)/np.power(k_init+kappa_init,2) 
print "R (plane wave): ",R_t1
print "T (plane wave): ",T_t1
print "R (wave packet): ",R_t1 + R_t2
print "T (wave packet): ",T_t1 - R_t2

plt.tight_layout()

# Propagate - set particle mass magnitue equal to hbar^2/2 for simplicity
# Time is in increments scaled by hbar
time_incr = 0.04
redraw_interval = 10
for time_step in np.arange(0,5000):

    p_0 = np.fft.fftshift(np.fft.fft(psi_x))
    p_1 = np.exp((-1j/2.0)*np.square(2*np.pi*freqs)*time_incr)*p_0
    psi_x_0 = np.exp(-1j*v_x*time_incr)*np.fft.ifft(np.fft.ifftshift(p_1))
    p_2 = np.fft.fftshift(np.fft.fft(psi_x_0))
    p_3 = np.exp((-1j/2.0)*np.square(2*np.pi*freqs)*time_incr)*p_2
    psi_x = np.fft.ifft(np.fft.ifftshift(p_3))

    if time_step % redraw_interval == 0:
        psi_x_line.set_ydata(np.square(np.absolute(psi_x)))
        plt.draw()

print "R (actual): ",np.trapz(np.square(np.absolute(psi_x))[0:len(xpoints)/2],xpoints[0:len(xpoints)/2])
print "T (actual): ",np.trapz(np.square(np.absolute(psi_x))[len(xpoints)/2:-1],xpoints[len(xpoints)/2:-1])
plt.ioff()
plt.show()
