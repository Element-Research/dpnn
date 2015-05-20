# dpnn : extensions to nn Modules and Criterions. 

This package provides many useful features that aren't part of the main nn package. 
These include [sharedClone], which allows you to clone a module and share 
parameters or gradParameters with the original module, without incuring any memory overhead.
Or [sharedType], which preserves Tensor sharing within a structure of modules. 

The Module interface has been further extended with methods that facilitate 
stochastic gradient descent like [updateGradParameters] (i.e. momentum learning), 
[weightDecay], [maxParamNorm] (for regularization), and so on.

The package provides the following Modules:
 * [Decorator](#nn.Decorator) : abstract class to change the behaviour of an encapsulated module ;
 * [DontCast](#nn.DontCast) : prevent encapsulated module from being casted by `Module:type()` ;
 * [Serial](#nn.Serial) : decorate a module makes its serialized output more compact ; 
 * [Inception](#nn.Inception) : implements the Inception module of the GoogleLeNet article ;
 * [Dictionary](#nn.Dictionary) : a LookupTable with sparse updates;
 * [Collapse](#nn.Collapse) : just like `nn.View(-1)`;
 * [Convert](#nn.Convert) : convert between different tensor types or shapes;
 * [ZipTable](#nn.ZipTable) : zip a table of tables into a table of tables;
 * [PrintSize](#nn.PrintSize) : prints the size of inputs and gradOutputs (useful for debugging);

A lot of the functionality implemented here was pulled from 
[dp](https://github.com/nicholas-leonard/dp), which makes heavy use of this package. 
However, dpnn can be used without dp (for e.g. you can use it with optim), 
which is one of the main reasons why we made it.

<a name='nn.Module'></a>
## Module ##

### Module:type(type_str) ###
This function converts all the parameters of a module to the given `type_str`. 
The `type_str` can be one of the types defined for [torch.Tensor](https://github.com/torch/torch7/blob/master/doc/tensor.md)
like `torch.DoubleTensor`, `torch.FloatTensor` and `torch.CudaTensor`. 
Unlike the [type method](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module.type)
defined in [nn](https://github.com/torch/nn), this one was overriden to 
maintain the sharing of [storage](https://github.com/torch/torch7/blob/master/doc/storage.md#storage)
among Tensors. This is especially useful when cloning modules share `parameters` and `gradParameters`.

### Module:sharedClone() ###

### Module:maxParamNorm([maxOutNorm, maxInNorm]) ###

### Module:updateGradParameters(momFactor [, momDamp, momNesterov]) ###

### Module:weightDecay(wdFactor [, wdMinDim]) ###

### Module:gradParamClip(cutoffNorm [, moduleLocal]) ###

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

<a name='nn.ZipTable'></a>
## ZipTable ##

<a name='nn.PrintSize'></a>
## PrintSize ##

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

