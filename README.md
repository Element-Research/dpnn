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
 * [Collapse](#nn.Collapse) : just like nn.View(-1);
 * [Convert](#nn.Convert) : convert between different tensor types or shapes;
 * [ZipTable](#nn.ZipTable) : zip a table of tables into a table of tables;
 * [PrintSize](#nn.PrintSize) : prints the size of inputs and gradOutputs (useful for debugging);

A lot of the functionality implemented here was pulled from 
[dp](https://github.com/nicholas-leonard/dp), which makes heavy use of this package. 
However, dpnn can be used without dp (for e.g. you can use it with optim), 
which is one of the main reasons why we made it.

## Module ##

## Decorator ##
