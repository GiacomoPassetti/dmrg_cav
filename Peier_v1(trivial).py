# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 09:43:05 2020

@author: giaco
"""

import sys
import numpy as np
from tenpy import models
from tenpy.networks.site import SpinSite
from tenpy.networks.site import FermionSite
from tenpy.networks.site import BosonSite
from tenpy.models.model import CouplingModel
from tenpy.models.model import CouplingMPOModel
from tenpy.models.spins import SpinModel
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.lattice import Lattice
from tenpy.tools.params import get_parameter
import tenpy.linalg.np_conserved as npc
from tenpy.networks.mpo import MPO, MPOEnvironment
import tenpy.linalg.charges as charges
from tenpy.models.lattice import Chain


def sites(L):
 FSite=FermionSite(None, filling=0.5)
 BSite=BosonSite(Nmax=2000,conserve=None, filling=0 )
 sites=[]
 sites.append(BSite)
 for i in range(L):
     sites.append(FSite)
 return sites

def product_state(L):
    ps=['vac']
    for i in range(int(L/2)):
        ps.append('empty')
        ps.append('full')
    return ps

def psi(sites,ps):
    psi=MPS.from_product_state(sites, ps)
    return psi

Nmax, L, g, U, Omega, t, Nfer =100, 6, 1, 0.5, 1, 1, 10


class Boson_operator:
    def __init__(self, dim):
        B = np.zeros([dim+1, dim+1], dtype=np.float)  # destruction/annihilation operator
        for n in range(1, dim+1):
          B[n - 1, n] = np.sqrt(n)
        self.Id=np.identity(dim+1)
        self.B=B
        self.Bd = np.transpose(B)  # .conj() wouldn't do anything
        # Note: np.dot(Bd, B) has numerical roundoff errors of eps~=4.4e-16.
        Ndiag=np.arange(dim+1, dtype=np.float)
        self.Ndiag = Ndiag
        self.N = np.diag(Ndiag)
        self.NN = np.diag(Ndiag**2)
        return None

class Fermion_operator:
    def __init__(self, h):
        JW = np.array([[1., 0.], [0., -1.]])
        self.JW = JW
        C = np.array([[0., 1.], [0., 0.]])
        self.C = C
        
        self.Cd = np.array([[0., 0.], [1., 0.]])
        self.N = np.array([[0., 0.], [0., 1.]])
        self.Id=np.identity(2)
        return None

bos=Boson_operator(Nmax)
fer=Fermion_operator(1)
W0=np.zeros((1,7,Nmax+1,Nmax+1),dtype=complex)
Wb=np.zeros((7,7,2,2),dtype=complex)
WL=np.zeros((7,1,2,2),dtype=complex)
fer=Fermion_operator(1)
bos=Boson_operator(Nmax)
W0[0,0,:]=Omega*bos.N
W0[0,2,:]=-t*(1-1j*g)*(bos.B+bos.Bd)
W0[0,4,:]=t*(1+1j*g)*(bos.B+bos.Bd)
W0[0,6,:]=bos.Id
Wb[0,0,:]=fer.Id
Wb[1,0,:]=fer.C
Wb[3,0,:]=fer.Cd
Wb[5,0,:]=fer.N
Wb[2,1,:]=fer.Cd
Wb[2,2,:]=fer.Id
Wb[4,3,:]=fer.C
Wb[4,4,:]=fer.Id
Wb[6,5,:]=U*fer.N
Wb[6,6,:]=fer.Id
WL=Wb[:,0,:,:]
WL=WL.reshape(7,1,2,2)
Ws=[npc.zeros([v_legs[0], v_legs[1].conj(), Bp_leg],dtype=np.float64,qtotal=[0],labels=['vL', 'vR', 'p'])]
W0=npc.Array.from_ndarray_trivial(W0,dtype=None, labels=['wL','wR','p','p*'])
Wb=npc.Array.from_ndarray_trivial(Wb,dtype=None, labels=['wL','wR','p','p*'])
WL=npc.Array.from_ndarray_trivial(WL,dtype=None, labels=['wL','wR','p','p*'])


Ws.append(W0)
for i in range(L-1):
    Ws.append(Wb)
Ws.append(WL)
#Let's creates first the MPS chain with all the Fermion sites:
chinfo = npc.ChargeInfo([1], ['N'])  # This is the information about the global conserved charge of the system: The total number of fermions
Bqflat=[[0]]*Nmax #Since the B_site doesnt contribute to the global number of Fermions, all of his physical legs have charge [0]
# create LegCharges on physical leg and even/odd bonds for the Fermion site
p_leg = npc.LegCharge.from_qflat(chinfo, [[1], [0]])  # charges for site occupied/unoccupied
Bp_leg=npc.LegCharge.from_qflat(chinfo, Bqflat)
v_legs=[]#create a list with all the legs for the chain (boson site included)
for i in range(int((L+2)/2)):
    v_legs.append(npc.LegCharge.from_qflat(chinfo, [[i]]))
    v_legs.append(npc.LegCharge.from_qflat(chinfo, [[i]]))


Bs=[npc.zeros([v_legs[0], v_legs[1].conj(), Bp_leg],dtype=np.float64,qtotal=[0],labels=['vL', 'vR', 'p'])]
for i in range(L):
    Bs.append(npc.zeros([v_legs[i+1], v_legs[i+2].conj(), p_leg],dtype=np.float64,qtotal=[0],labels=['vL', 'vR', 'p']))
    
Bs[0][0,0,0]=1
for i in range(int(L/2)):
     Bs[2*i+1][0,0,0]=1
for i in range(int(L/2)):
     Bs[2*i+2][0,0,1]=1
print(Bs[1],Bs[2])











