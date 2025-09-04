"""
wtf_compile.py

class to instantiate object that can perform a forwards pass through a network as defined by an input dictionary (which has been created using wtf.tf_to_dict)

"""

import numpy as np
import jax.numpy as jnp

class wtf_compile:
    def __init__(
        self, 
        wtf_dict,
        test=False, 
        jaxxed=False
        ):
        
        self.wtf_dict = wtf_dict
        self.jaxxed = jaxxed
        self.test = test

    if jaxxed:
        pass

    else:
        
        def forward(self, x):
            for layer in self.wtf_dict['structure']['stem']:
                print(layer)
            









        

