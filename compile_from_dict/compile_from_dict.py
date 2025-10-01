"""
wtf_compile.py

class to instantiate object that can perform a forwards pass through a network as defined by an input dictionary (which has been created using wtf.tf_to_dict)

"""
class wtf_base:
    def __init__(self, wtf_dict, to_array_func, to_array_dtype):
        self.wtf_dict = wtf_dict
        self.to_array_func = to_array_func
        self.to_array_dtype = to_array_dtype
        
        self.n_branches = len([key for key in list(self.wtf_dict['structure'].keys())[1:] if 'branch' in key])

        self.stem_layers, self.branches = self.build_model_from_dict(to_array_func, to_array_dtype)

    def build_layers_from_dict(self, layer_list, layer_names, to_array_func, to_array_dtype):
        for layer_name in layer_names:
            layer = self.wtf_dict['layers'][layer_name]
            layer_type = layer['type']

            if layer_type == 'Dense':
                weights = to_array_func(layer['weights'], dtype=to_array_dtype)
                biases = to_array_func(layer['biases'], dtype=to_array_dtype)
                activation = layer['activation']
                
                layer_list.append(('Dense', weights, biases, activation))

            elif layer_type == 'InputLayer':
                
                layer_list.append(('InputLayer', layer['batch_shape']))
    
    def build_model_from_dict(self, to_array_func, to_array_dtype):
        stem_layers = []
        stem_layer_names = list(self.wtf_dict['structure']['stem'].values())
        self.build_layers_from_dict(stem_layers, stem_layer_names, to_array_func, to_array_dtype)

        branches = []
        for branch_idx in range(self.n_branches):
            branch_layers = []
            branch_layer_names = list(self.wtf_dict['structure'][f'branch_{branch_idx}'].values())[1:]
            self.build_layers_from_dict(branch_layers, branch_layer_names, to_array_func, to_array_dtype)

            branches.append(branch_layers)

        return stem_layers, branches

class numpy_compile(wtf_base):
    import numpy as np
    
    def __init__(self, wtf_dict):
        super().__init__(wtf_dict, to_array_func=self.np.array, to_array_dtype=self.np.float32)

        self.stem_functions = self.make_layer_functions(self.stem_layers)
        self.branches_functions = [self.make_layer_functions(branch_layers) for branch_layers in self.branches]


    def elu(self, x):
        return self.np.where(x >= 0, x, self.np.exp(x)-1)
    
    def make_layer_functions(self, layers):
        layer_functions = []
        for layer in layers:
            if layer[0] == 'Dense':
                _, weights, biases, activation = layer

                def layer_function(x, weights=weights, biases=biases, activation=activation):
                    x = x @ weights + biases
                    
                    if activation == 'elu':
                        x = self.elu(x)
                        
                    return x
                    
                layer_functions.append(layer_function)

            elif layer[0] == 'InputLayer':
                _, batch_shape = layer
                
                def layer_function(x, expected_batch_shape=batch_shape[1]):
                    if x.shape[1] != expected_batch_shape:
                        raise ValueError(f'wtf!\r \tinput shape = (n, {x.shape[1]}), but expected input layer batch_shape = (n, {expected_batch_shape})')
                    
                    return x
                    
                layer_functions.append(layer_function)
                
        return layer_functions   

    def layer_functions_pass(self, layer_functions, x):
        for layer_function in layer_functions:
            x = layer_function(x)

        return x
    
    def stem_pass(self, x):
        return self.layer_functions_pass(self.stem_functions, x)

    def branch_pass(self, stem_outputs):
        return [self.layer_functions_pass(branch_functions, stem_outputs) for branch_functions in self.branches_functions]

    def forward_pass(self, x):
        stem_output = self.stem_pass(x)
        return self.branch_pass(stem_output) if self.n_branches else stem_output
    
class jax_compile(wtf_base):
    import jax
    import jax.numpy as jnp
    
    def __init__(self, wtf_dict):
        super().__init__(wtf_dict, to_array_func=self.jnp.array, to_array_dtype=self.jnp.float32)

        self.stem_functions = self.make_layer_functions(self.stem_layers)
        self.branches_functions = [self.make_layer_functions(branch_layers) for branch_layers in self.branches]

        self.jit_forward_pass = self.jax.jit(self.forward_pass)

    def make_layer_functions(self, layers):
        layer_functions = []
        for layer in layers:
            if layer[0] == 'Dense':
                _, weights, biases, activation = layer

                def layer_function(x, weights=weights, biases=biases, activation=activation):
                    x = x @ weights + biases
                    
                    if activation == 'elu':
                        x = self.jax.nn.elu(x)
                        
                    return x
                    
                layer_functions.append(layer_function)

            elif layer[0] == 'InputLayer':
                _, batch_shape = layer
                
                def layer_function(x):
                    return x
                    
                layer_functions.append(layer_function)
                
        return layer_functions   

    def layer_functions_pass(self, layer_functions, x):
        for layer_function in layer_functions:
            x = layer_function(x)

        return x
    
    def stem_pass(self, x):
        return self.layer_functions_pass(self.stem_functions, x)

    def branch_pass(self, stem_outputs):
        return [self.layer_functions_pass(branch_functions, stem_outputs) for branch_functions in self.branches_functions]

    def forward_pass(self, x):
        stem_output = self.stem_pass(x)
        return self.branch_pass(stem_output) if self.n_branches else stem_output
    

        

        
                
        



            
            
            
        











        

