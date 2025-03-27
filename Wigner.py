import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

def Gaussian(v,sx,stx,sy,sty):
    x, tx, y, ty = v
    return np.exp(-(x**2)/2/(sx**2)-(tx**2)/2/(stx**2)-(y**2)/2/(sy**2)-(ty**2)/2/(sty**2))/(2*np.pi)**2/sx/stx/sy/sty

x = np.linspace(-1.3/2,1.3/2,1000)
tx = np.linspace(-0.02,0.02,1000)
x, tx = np.meshgrid(x,tx)
y = np.zeros(x.shape); ty = np.zeros(tx.shape)
W0 = Gaussian([x,tx,y,ty],1.3/4,0.01,1.3/4,0.01)

prop1 = np.matrix([[1,1000],[0,1]])
prop1 = block_diag(prop1,prop1,np.eye(2))
lens1 = np.matrix([[1,0],[-1/100,1]])
lens1 = block_diag(lens1,lens1,np.eye(2))
prop2 = np.matrix([[1,100],[0,1]])
prop2 = block_diag(prop2,prop2,np.eye(2))

dev = np.linalg.inv(prop2)@np.linalg.inv(lens1)@np.linalg.inv(prop1)
dev = prop2.T@lens1.T@prop1.T
#z = np.stack([x, tx, y, ty, np.zeros(x.shape), np.zeros(x.shape)], axis=-1)
#x, tx, y, ty, _, _ = dev@z
x = x*(dev[0,0]+dev[1,0]+dev[2,0]+dev[3,0])
tx = tx*(dev[0,1]+dev[1,1]+dev[2,1]+dev[3,1])
y = y*(dev[0,2]+dev[1,2]+dev[2,2]+dev[3,2])
ty = ty*(dev[0,3]+dev[1,3]+dev[2,3]+dev[3,3])

Wf = Gaussian([x,tx,y,ty],1.3/4,0.01,1.3/4,0.01)

cov = np.diag([(1.3/4)**2,0.01**2,(1.3/4)**2,0.01**2,1,1])
cov = np.linalg.inv(prop2)@np.linalg.inv(lens1)@np.linalg.inv(prop1)@cov@prop1@lens1@prop2
print(cov)

plt.subplot(121)
plt.imshow(W0)
plt.subplot(122)
plt.imshow(Wf)
plt.show()