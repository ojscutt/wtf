# WTF - Without TensorFlow
a couple functions for moving away from TensorFlow dependency after training a model:

`tf_to_json` - goes from a saved keras model to a streamlined dict saved to json containing important architecture information, weights, bases, and layer order.

`json_to_numpy` - loads in a network from a saved json dict and converts forward pass to a set of numpy operations (use this if you don't plan on JIT compiling!)

`json_to_jax` - loads in a network from a saved json dict and converts forward pass to a set of JAX operations (use this if you want to JIT compile the forwards pass!)
