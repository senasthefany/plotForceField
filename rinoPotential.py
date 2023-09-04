import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import math
import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)
matplotlib.rc('axes', linewidth=2)
matplotlib.rc('lines', linewidth=3)


def rino(i):
    potential = []

    for r in radius:
        if r<rcut:
            V = (i['H']/(r**(i['eta']))) + ((i['Zi']*i['Zj'])/r)*math.exp(-r/i['lambda1']) - (i['D']/r**4)*math.exp(-r/i['lambda4']) - (i['W']/r**6)
        else:
            V = 0
        potential.append(V)
    rinoPotential2body.append(potential)

def rino2(i):
    potAngles = []
    potRadius = []

    for angle in angles:
        V = ((math.cos(angle)-i['cosTheta'])**2)/(1 + i['C']*(math.cos(angle)-i['cosTheta'])**2)
        potAngles.append(V)
    rinoPot3bodyAngle.append(potAngles)



SiSiSi = {
    'H': 0.82023,
    'eta': 11.0,
    'Zi': 1.6,
    'Zj': 1.6,
    'lambda1': 999,
    'D': 0.0,
    'lambda4': 4.43,
    'W': 0.0,
    'B': 0.0,
    'cosTheta': 0.0,
    'C': 0.0,
    'gamma':0.0,
    'r0': 0.0,
    }

OOO = {
    'H': 743.848,
    'eta': 7.0,
    'Zi': -0.8,
    'Zj': -0.8,
    'lambda1': 999,
    'D': 22.1179,
    'lambda4': 4.43,
    'W': 0.0,
    'B': 0.0,
    'cosTheta': 0.0,
    'C': 0.0,
    'gamma': 0.0,
    'r0': 0.0,
    }

OSiSi = {
    'H': 163.859,
    'eta': 9.0,
    'Zi': -0.8,
    'Zj': 1.6,
    'lambda1': 999,
    'D': 44.2357,
    'lambda4': 4.43,
    'W': 0.0,
    'B': 20.146,
    'cosTheta': -0.77714596,
    'C': 0.0,
    'gamma': 1.0,
    'r0': 2.6,
    }

SiOO = {
    'H': 163.859,
    'eta': 9.0,
    'Zi': 1.6,
    'Zj': -0.8,
    'lambda1': 999,
    'D': 44.2357,
    'lambda4': 4.43,
    'W': 0.0,
    'B': 5.0365,
    'cosTheta': -0.333333333333,
    'C': 0.0,
    'gamma': 1.0,
    'r0': 2.6,
    }

rcut = 10.0
pot2 = [SiSiSi, OOO, SiOO]
pot3 = [SiSiSi, OOO, OSiSi, SiOO]
radius = np.arange(0.9,10.00,0.05)
radiusAngle1 = np.arange(0.5,3.0,0.1)
radiusAngle2 = np.arange(0.5,3.0,0.1)
angles = np.linspace(0,np.pi,180)

rinoPotential2body = []
rinoPotential3body = []

rinoPot3bodyAngle = []
rinoPot3bodyRadius = []

for i in pot2:
    rino(i)
#np.savetxt('rinoPotential2body.out', np.c_[radius,rinoPotential2body[0],rinoPotential2body[1],rinoPotential2body[2]])


for i in pot3:
    rino2(i)

#2-body
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(radius, rinoPotential2body[0], 'm', label='Si-Si')
ax.plot(radius, rinoPotential2body[1], 'r', label='O-O')
ax.plot(radius, rinoPotential2body[2], 'k', label='Si-O')
ax.legend()
plt.xlabel(r'r ($\AA$)', fontsize=30)
plt.ylabel(r'V (eV)', fontsize=30)
plt.xlim([0, 10])
plt.ylim([-3.5, 5])
plt.tight_layout()
plt.savefig('rino2.png',dpi=100)
plt.close()


#3-body angle
fig, bx = plt.subplots(figsize=(10,8))
# bx.plot(angles, rinoPot3bodyAngle[0], 'g', label='Si-Si-Si')
bx.plot(angles, rinoPot3bodyAngle[1], 'm',label='O-O-O /\nSi-Si-Si')
bx.plot(angles, rinoPot3bodyAngle[2], 'k', label='O-Si-Si')
bx.plot(angles, rinoPot3bodyAngle[3], 'r', label='Si-O-O')
plt.xlim([0,np.pi])
tick_pos= [0, np.pi/4, np.pi/2, 3*np.pi/4 , np.pi]
labels = ['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']
plt.xticks(tick_pos, labels)
plt.xlabel(r'$\theta_{ijk}$', fontsize=30, labelpad=2)
plt.ylabel(r'$\Theta(\theta_{ijk})$', fontsize=30)
bx.legend()
plt.tight_layout()
plt.savefig('rino3Angle.png',dpi=100)
plt.close()


#3-body Radius
x = np.arange(0.1,2.6,0.1)
y = np.arange(0.1,2.6,0.1)

plt.subplots(figsize=(10,8))
X, Y = np.meshgrid(x,y)
Z = np.zeros((len(x),len(x)))

# for i in range(len(x)):
#    for j in range(len(x)):
#        Z[i,j] = f(X[i,j],Y[i,j])
i = OSiSi
Z = np.exp(i['gamma']/(X-i['r0'])) * np.exp(i['gamma']/(Y-i['r0']))
# plt.xlim([0, 3])
# plt.ylim([0, 3])
plt.imshow(Z, interpolation='catrom', cmap='gist_heat', extent=[0.01,3,0,3], origin='lower',vmin=0, vmax=0.5)
plt.xlabel(r'$r_{ij} \;(\AA)$', fontsize=30)
plt.ylabel(r'$r_{ik} \;(\AA)$', fontsize=30)
cbar = plt.colorbar()
cbar.ax.locator_params(nbins=5)
cbar.set_label(r'$R(r_{ij},r_{ik})$', fontsize=30)
plt.tight_layout()
plt.savefig('rino3Radius.png',dpi=100)
plt.close()
# plt.show()
