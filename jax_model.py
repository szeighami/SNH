import haiku as hk
import jax

def Phi(features, out_dim, in_dim, width, no_layers, **kwargs):
	def sine_func(x):
  		return jax.numpy.sin(x)

	layer_stack = []
	layer_stack.append(hk.Linear(width, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform')))# first layer
	layer_stack.append(sine_func)
	for j in range(no_layers): # consecutive layers
		layer_stack.append(hk.Linear(width, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform')))
		layer_stack.append(jax.nn.swish)
	layer_stack.append(hk.Linear(out_dim, w_init=hk.initializers.VarianceScaling(scale=2.0, distribution='uniform')))# final layer

	return hk.Sequential(layer_stack)(features)
