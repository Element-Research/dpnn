# dpnn : deep extensions to nn

This package provides many useful features that aren't part of the main nn package. 
These include [sharedClone](#nn.Module.sharedClone), which allows you to clone a module and share 
parameters or gradParameters with the original module, without incuring any memory overhead.
We also redefined [type](#nn.Module.type) such that the type-cast preserves Tensor sharing within a structure of modules. 

The package provides the following Modules:

 * [Decorator](#nn.Decorator) : abstract class to change the behaviour of an encapsulated module ;
 * [DontCast](#nn.DontCast) : prevent encapsulated module from being casted by `Module:type()` ;
 * [Serial](#nn.Serial) : decorate a module makes its serialized output more compact ; 
 * [Inception](#nn.Inception) : implements the Inception module of the GoogleLeNet article ;
 * [Dictionary](#nn.Dictionary) : a LookupTable with sparse updates;
 * [Collapse](#nn.Collapse) : just like `nn.View(-1)`;
 * [Convert](#nn.Convert) : convert between different tensor types or shapes;
 * [ZipTable](#nn.ZipTable) : zip a table of tables into a table of tables;
 * [ReverseTable](#nn.ReverseTable) : reverse the order of elements in a table;
 * [PrintSize](#nn.PrintSize) : prints the size of inputs and gradOutputs (useful for debugging);
 * [WhiteNoise](#nn.WhiteNoise) : Adds isotropic Gaussian noise to the signal when in training mode.

A lot of the functionality implemented here was pulled from 
[dp](https://github.com/nicholas-leonard/dp), which makes heavy use of this package. 
However, dpnn can be used without dp (for e.g. you can use it with optim), 
which is one of the main reasons why we made it.

<a name='nn.Module'></a>
## Module ##

The Module interface has been further extended with methods that facilitate 
stochastic gradient descent like [updateGradParameters](#nn.Module.updageGradParameters) (i.e. momentum learning), 
[weightDecay](#nn.Module.weightDecay), [maxParamNorm](#nn.Module.maxParamNorm) (for regularization), and so on.

<a name='nn.Module.dpnn_parameters'></a>
### Module.dpnn_parameters ###

A table that specifies the name of parameter attributes. 
Defaults to `{'weight', 'bias'}`, which is a static variable (i.e. table exists in class namespace). 
Sub-classes can define their own table statically. 

<a name='nn.Module.dpnn_gradParameters'></a>
### Module.dpnn_gradParameters ###

A table that specifies the name of gradient w.r.t. parameter attributes. 
Defaults to `{'gradWeight', 'gradBias'}`, which is a static variable (i.e. table exists in class namespace). 
Sub-classes can define their own table statically. 

<a name='nn.Module.type'></a>
### [self] Module:type(type_str) ###

This function converts all the parameters of a module to the given `type_str`. 
The `type_str` can be one of the types defined for [torch.Tensor](https://github.com/torch/torch7/blob/master/doc/tensor.md)
like `torch.DoubleTensor`, `torch.FloatTensor` and `torch.CudaTensor`. 
Unlike the [type method](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module.type)
defined in [nn](https://github.com/torch/nn), this one was overriden to 
maintain the sharing of [storage](https://github.com/torch/torch7/blob/master/doc/storage.md#storage)
among Tensors. This is especially useful when cloning modules share `parameters` and `gradParameters`.

<a name='nn.Module.sharedClone'></a>
### [clone] Module:sharedClone([shareParams, shareGradParams]) ###

Similar to [clone](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module.clone).
Yet when `shareParams = true` (the default), the cloned module will share the parameters 
with the original module. 
Furthermore, when `shareGradParams = true` (the default), the clone module will share 
the gradients w.r.t. parameters with the original module.
This is equivalent to :
```lua
clone = mlp:clone()
clone:share(mlp, 'weight', 'bias', 'gradWeight', 'gradBias')
```
yet it is much more efficient, especially for modules with lots of parameters, as these 
Tensors aren't needlessly copied during the `clone`.
This is particularly useful for [Recurrent neural networks](https://github.com/Element-Research/rnn/blob/master/README.md) 
which require efficient copies with shared parameters and gradient w.r.t. parameters for each time-step.

<a name='nn.Module.maxParamNorm'></a>
### Module:maxParamNorm([maxOutNorm, maxInNorm]) ###

This method implements a hard constraint on the upper bound of the norm of output and/or input neuron weights 
[(Hinton et al. 2012, p. 2)](http://arxiv.org/pdf/1207.0580.pdf) .
In a weight matrix, this is a contraint on rows (`maxOutNorm`) and/or columns (`maxInNorm`), respectively. 
Has a regularization effect analogous to [weightDecay](#nn.Module.weightDecay), but with easier to optimize hyper-parameters. 
Assumes that parameters are arranged (`output dim x ... x input dim`). 
Only affects parameters with more than one dimension.
The method should normally be called after [updateParameters](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module.updateParameters). 
It uses the C/CUDA optimized [torch.renorm](https://github.com/torch/torch7/blob/master/doc/maths.md#torch.renorm) function.
Hint : `maxOutNorm = 2` usually does the trick. 

<a name='nn.Module.momentumGradParameters'></a>
### [momGradParams] Module:momentumGradParameters() ###

Returns a table of Tensors (`momGradParams`). For each element in the 
table, a corresponding parameter (`params`) and gradient w.r.t. parameters 
(`gradParams`) is returned by a call to [parameters](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module.parameters).
This method is used internally by [updateGradParameters](#nn.Module.updateGradParameters).

<a name='nn.Module.updateGradParameters'></a>
### Module:updateGradParameters(momFactor [, momDamp, momNesterov]) ###

Applies classic momentum or Nesterov momentum [(Sutskever, Martens et al, 2013)](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf) to parameter gradients. 
Each parameter Tensor (`params`) has a corresponding Tensor of the same size for gradients w.r.t. parameters (`gradParams`).
When using momentum learning, another Tensor is added for each parameter Tensor (`momGradParams`).
This method should be called before [updateParameters](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module.updateParameters)
as it affects the gradients w.r.t. parameters.

Classic momentum is computed as follows :

```lua
momGradParams = momFactor*momGradParams + (1-momDamp)*gradParams
gradParams = momGradParams
```

where `momDamp` has a default value of `momFactor`.

Nesterov momentum (`momNesterov = true`) is computed as follows (the first line is the same as classic momentum):

```lua
momGradParams = momFactor*momGradParams + (1-momDamp)*gradParams
gradParams = gradParams + momFactor*momGradParams
```
The default is to use classic momentum (`momNesterov = false`).

<a name='nn.Module.weightDecay'></a>
### Module:weightDecay(wdFactor [, wdMinDim]) ###

Decays the weight of the parameterized models. 
Implements an L2 norm loss on parameters with dimensions greater or equal to `wdMinDim` (default is 2).
The resulting gradients are stored into the corresponding gradients w.r.t. parameters.
Such that this method should be called before [updateParameters](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module.updateParameters).

<a name='nn.Module.gradParamClip'></a>
### Module:gradParamClip(cutoffNorm [, moduleLocal]) ###

Implements a contrainst on the norm of gradients w.r.t. parameters [(Pascanu et al. 2012)](http://arxiv.org/pdf/1211.5063.pdf).
When `moduleLocal = false` (the default), the norm is calculated globally to Module for which this is called.
So if you call it on an MLP, the norm is computed on the concatenation of all parameter Tensors.
When `moduleLocal = true`, the norm constraint is applied 
to the norm of all parameters in each component (non-container) module.
This method is useful to prevent the exploding gradient in 
[Recurrent neural networks](https://github.com/Element-Research/rnn/blob/master/README.md).

<a name='nn.Decorator'></a>
## Decorator ##

```lua
dmodule = nn.Decorator(module)
```

This module is an abstract class used to decorate a `module`. This means 
that method calls to `dmodule` will call the same method on the encapsulated 
`module`, and return its results.

<a name='nn.DontCast'></a>
## DontCast ##

```lua
dmodule = nn.DontCast(module)
```

This module is a decorator. Use it to decorate a module that you don't
want to be cast when the `type()` method is called.

```lua
module = nn.DontCast(nn.Linear(3,4):float())
module:double()
th> print(module:forward(torch.FloatTensor{1,2,3}))
 1.0927
-1.9380
-1.8158
-0.0805
[torch.FloatTensor of size 4]
```

<a name='nn.Serial'></a>
## Serial ##
```lua
dmodule = nn.Serial(module)
dmodule:[light,medium,heavy]Serial([type])
```
This module is a decorator that can be used to control the serialization/deserialization 
behavior of the encapsulated module. Basically, making the resulting string or 
file heavy (the default), medium or light in terms of size. Furthermore, when 
specified, the `type` attribute (e.g *float*, *double*, *cuda*, *torch.FloatTensor*, *torch.DoubleTensor* and so on.),
determines what type the module will be cast to during serialization. 
Note that this will also be the type of the deserialized object.

The `heavySerial([type])` has the serialization process serialize every attribute in the module graph, 
which is the default behavior of nn. 

The `mediumSerial([type])` has the serialization process serialize 
everything except the attributes specified in each module's `dpnn_mediumEmpty`
table, which has a default value of `{'output', 'gradInput', 'momGradParams', 'dpnn_input'}`.
During serialization, whether they be tables or Tensors, these attributes are emptied (no storage).
Some modules overwrite the default `Module.dpnn_mediumEmpty` static attribute with their own.
The default serialization `type` of the `mediumSerial()` is *float*.

The `lightSerial([type])` has the serialization process empty  
everything a call to `mediumSerial(type)` would (so it uses `dpnn_mediumEmpty`).
But also empties all the parameter gradients specified by the 
attribute `dpnn_gradParameters`, which defaults to `{gradWeight, gradBias}`.

We recomment using `mediumSerial()` for training, and `lightSerial()` for 
production (feed-forward-only models).

<a name='nn.Inception'></a>
## Inception ##

<a name='nn.Dictionary'></a>
## Dictionary ##

<a name='nn.Collapse'></a>
## Collapse ##

```lua
module = nn.Collapse(nInputDim)
```

This module is the equivalent of:
```
view = nn.View(-1)
view:setNumInputDim(nInputDim)
```
It collapses all non-batch dimensions. This is useful for converting 
a spatial feature map to the single dimension required by a dense 
hidden layer like Linear.

<a name='nn.Convert'></a>
## Convert ##

```lua
module = nn.Convert([inputShape, outputShape])
```
Module to convert between different data formats.
For example, we can flatten images by using :
```lua
module = nn.Convert('bchw', 'bf')
``` 
or equivalently
```lua
module = nn.Convert('chw', 'f')
```
Lets try it with an input:
```lua
print(module:forward(torch.randn(3,2,3,1)))
 0.5692 -0.0190  0.5243  0.7530  0.4230  1.2483
-0.9142  0.6013  0.5608 -1.0417 -1.4014  1.0177
-1.5207 -0.1641 -0.4166  1.4810 -1.1725 -1.0037
[torch.DoubleTensor of size 3x6]
```
You could also try:

```lua
module = nn.Convert('chw', 'hwc')
input = torch.randn(1,2,3,2)
input:select(2,1):fill(1)
input:select(2,2):fill(2)
print(input)
(1,1,.,.) = 
  1  1
  1  1
  1  1
(1,2,.,.) = 
  2  2
  2  2
  2  2
[torch.DoubleTensor of size 1x2x3x2]
print(module:forward(input))
(1,1,.,.) = 
  1  2
  1  2

(1,2,.,.) = 
  1  2
  1  2

(1,3,.,.) = 
  1  2
  1  2
[torch.DoubleTensor of size 1x3x2x2]
```


Furthermore, it automatically converts the `input` to have the same type as `self.output`
(i.e. the type of the module).
So you can also just use is for automatic input type converions:
```lua
module = nn.Convert()
print(module.output) -- type of module
[torch.DoubleTensor with no dimension]
input = torch.FloatTensor{1,2,3}
print(module:forward(input))
 1
 2
 3
[torch.DoubleTensor of size 3]
```

<a name='nn.ZipTable'></a>
## ZipTable ##

```lua
module = nn.ZipTable()
```

Zips a table of tables into a table of tables.

Example:
```lua
print(module:forward{ {'a1','a2'}, {'b1','b2'}, {'c1','c2'} })
{ {'a1','b1','c1'}, {'a2','b2','c2'} }
```

<a name='nn.ReverseTable'></a>
## ReverseTable ##

```lua
module = nn.ReverseTable()
```

Reverses the order of elements in a table.

Example:

```lua
print(module:forward{1,2,3,4})
{4,3,2,1}
```

<a name='nn.PrintSize'></a>
## PrintSize ##

<a name='nn.WhiteNoise'></a>
## WhiteNoise ##

```lua
module = nn.WhiteNoise([mean, stdev])
```
Useful in training [Denoising Autoencoders] (http://arxiv.org/pdf/1507.02672v1.pdf). Takes `mean` and `stdev` of the `Gaussian` as input. Default values for mean and standard deviation are 0 and 0.1 respectively. With `module:training()`, noise is added during forward. During `backward` gradients are passed as it is. With `module:evaluate()` the mean is added to the input.
<a name='nn.ModuleCriterion'></a>
## ModuleCriterion ##

```lua
criterion = nn.ModuleCriterion(criterion [, inputModule, targetModule, castTarget])
```

This criterion decorates a `criterion` by allowing the `input` and `target` to be 
fed through optional an `inputModule` and `targetModule` before being passed to the 
`criterion`. The `inputModule` must not contain parameters as these would not be updated. 

When `castTarget = true` (the default), the `targetModule` is cast along with the `inputModule` and 
`criterion`. Otherwise, the `targetModule` isn't.  
