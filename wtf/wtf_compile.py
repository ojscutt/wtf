"""
wtf_compile.py

class to instantiate object that can perform a forwards pass through a network as defined by an input dictionary (which has been created using wtf.tf_to_dict)

"""

import numpy as np
import jax.numpy as jnp

class wtf_compile:
    def __init__(self, wtf_dict, jaxxed=False):
        self.wtf_dict = wtf_dict
        self.jaxxed = jaxxed

        self.n_branches = len([key for key in list(self.wtf_dict['structure'].keys())[1:] if 'branch' in key]) 
    
    def elu(self, x):
        if self.jaxxed:
            return jnp.where(x >= 0, x, jnp.exp(x)-1)

        else:
            return np.where(x >= 0, x, np.exp(x)-1)

    def stem_pass(self, x):
        if self.jaxxed:
            for layer_name in list(self.wtf_dict['structure']['stem'].values()):
                layer = self.wtf_dict['layers'][layer_name]
                layer_type = layer['type']
                
                if layer_type == 'InputLayer':
                    input_shape = jnp.shape(x)
                    batch_shape = layer['batch_shape']
                    assert input_shape[1] == batch_shape[1],(
                        f'wtf!\r \tinput shape = (n, {input_shape[1]}), but expected input layer batch_shape = (n, {batch_shape[1]})'
                    )
                    pass

                elif layer_type == 'Dense':
                    activation = layer['activation']
                    
                    weights = jnp.array(layer['weights'])
                    biases = jnp.array(layer['biases'])
                    
                    x = jnp.dot(x, weights) + biases

                    if activation == 'elu':
                        x = self.elu(x)
            return x
            
        else:
            for layer_name in list(self.wtf_dict['structure']['stem'].values()):
                layer = self.wtf_dict['layers'][layer_name]
                layer_type = layer['type']
                
                if layer_type == 'InputLayer':
                    input_shape = np.shape(x)
                    batch_shape = layer['batch_shape']
                    assert input_shape[1] == batch_shape[1],(
                        f'wtf!\r \tinput shape = (n, {input_shape[1]}), but expected input layer batch_shape = (n, {batch_shape[1]})'
                    )
                    pass

                elif layer_type == 'Dense':
                    activation = layer['activation']
                    
                    weights = np.array(layer['weights'])
                    biases = np.array(layer['biases'])
                    
                    x = np.dot(x, weights) + biases

                    if activation == 'elu':
                        x = self.elu(x)
            return x

    def branch_pass(self, stem_out):
        if self.jaxxed:
            branch_outs = []
            
            for branch_idx in range(self.n_branches):
                x = stem_out
                
                for layer_name in list(self.wtf_dict['structure'][f'branch_{branch_idx}'].values())[1:]: #[1:] because first is final layer of stem
                    layer = self.wtf_dict['layers'][layer_name]
                    layer_type = layer['type']
    
                    if layer_type == 'Dense':
                        activation = layer['activation']
                        
                        weights = jnp.array(layer['weights'])
                        biases = jnp.array(layer['biases'])
                        
                        x = jnp.dot(x, weights) + biases
    
                        if activation == 'elu':
                            x = self.elu(x)

                branch_outs.append(x)

            return branch_outs

        else:
            branch_outs = []
            
            for branch_idx in range(self.n_branches):
                x = stem_out
                
                for layer_name in list(self.wtf_dict['structure'][f'branch_{branch_idx}'].values())[1:]: #[1:] because first is final layer of stem
                    layer = self.wtf_dict['layers'][layer_name]
                    layer_type = layer['type']
                    
    
                    if layer_type == 'Dense':
                        activation = layer['activation']
                        
                        weights = np.array(layer['weights'])
                        biases = np.array(layer['biases'])
                        
                        x = np.dot(x, weights) + biases
    
                        if activation == 'elu':
                            x = self.elu(x)

                branch_outs.append(x)

            return branch_outs

    def forward(self, x):

        stem_out = self.stem_pass(x)
        
        if self.n_branches > 0:
            branch_outs = self.branch_pass(stem_out)
            return branch_outs

        else:
            return stem_out









        

