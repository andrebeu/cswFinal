import sys
import itertools
import numpy as np

"""
controls the gridsearch parameter space
given an index, return parameter string
""" 

## given parameter index, returns parameter string
param_set_idx = int(sys.argv[1])
##

c = np.arange(10,100,5) # concentration 
stwi = np.arange(10,120,10) # stickiness_wi
stbt = np.arange(10,100,10) # stickiness_bt
# sp = np.arange(0.85,1.21,0.1) # sparsity
# pvar = [0] # pvar
# lrate = [1] # lrate
# lratep = [1] # lratep
# decay = [1] # decay_rate

itrprod = itertools.product(
    c,stwi,stbt,
    # sp,pvar,lrate,lratep,decay
)

# gsize = len(c)*len(stwi)*len(stbt)*len(sp)*\
# len(pvar)*len(lrate)*len(lratep)*len(decay)
gsize = len(c)*len(stwi)*len(stbt)

# print('grid size',gsize,'COMMENT OUT BEFORE RUNNING')

for idx,paramL in enumerate(itrprod):
  if idx == param_set_idx:
    print(" ".join([str(i) for i in paramL]))
    break


