# Lateral Connections in Denoising Autoencoders Support Supervised Learning

In this tutorial we will understand how to implement ladder network as explained in [[1](http://arxiv.org/pdf/1504.08215.pdf)]. In this paper the authors have shown how unsupervised learning using a denoising autoencoder with lateral connections help improve the classification accuracy in supervised learning.

To produce results as mentioned in the paper please run following command (best test error we got was **`0.6%`**). To run this script you will need following torch packages: [`nn`](https://github.com/torch/nn), [`nngraph`](https://github.com/torch/nngraph), [`dp`](https://github.com/nicholas-leonard/dp), [`dpnn`](https://github.com/Element-Research/dpnn), [`optim`](https://github.com/torch/optim) and [`cunn`](https://github.com/torch/cunn) & [`cutorch`](https://github.com/torch/cutorch) if using cuda (```--useCuda``` flag).
```
   th tutorials/ladder.lua --verbose --eta 500 --epochs 100 --learningRate 0.002 --linearDecay --endLearningRate 0 --startEpoch 50 --useCuda --deviceId 1 --noiseSigma 0.3 --useBatchNorm --batchSize 100 --adam --noValidation --attempts 10
```

The unsupervised learning (denoising) task supplements the supervised learning task (classification in this case). As in autoencoders this network has an encoder and a decoder. The output of encoder is also used for classification. The output of encoder is **`N`** dimensional where **`N`** is number of classes. This **`N`** dimensional vector is used for computing classification cost as well as feeds into the decoder.

## Classification
Encoder/classifier units are defined as
```lua
   Z = nn.BatchNormalization(hidden_units)(nn.Linear(inputDims, hidden_units)(previous_H))
```
where
```lua
   H = nn.ReLU()(nn.CMul()(nn.Add()(Z)))
```
For first layer **`previous_H`** is the corrupted input.
```lua
   input = nn.WhiteNoise(mean, sigma)
```

**`H`** for last encoder unit is defined as
```lua
   H = nn.LogSoftMax()(nn.CMul()(nn.Add()(Z)))
```
Last **`H`** feeds into the negative log likelihood criterion.

## Denoising
Typically in denoising autoencoder the input samples are corrupted using Dropout [```nn.Dropout```](https://github.com/torch/nn/blob/master/Dropout.lua) but in this paper the authors use isotropic Gaussian noise [```nn.WhiteNoise```](https://github.com/Element-Research/dpnn/blob/master/WhiteNoise.lua) with zero mean.

### Lateral Connections in Autoencoder
**`Z`** units in encoder are laterally connected to corresponding unit in the decoder. The output of decoder unit for neuron `i` is defined by
```
   z^_i = a_i1 * z_i + a_i2 * sigmoid(a_i3 + a_i4) + a_i5
```
where 
```
   a_ij = c_ij * u_i + d_ij
```
**`U`** is output of decoder unit's ```nn.Linear()```. For the top most layer  **`U`** is zero. **`Z`** is output of corresponding encoder unit (this is lateral connection, decoder takes output from its previous unit through **`U`** as well as corresponding encoder unit). For the lowest layer of decoder **`Z`** is the corrupted input signal. **`c_j`** and **`d_j`** are trainable weight vectors. This forms the crux of the ladder network. This can be easily implemented using **`nngraph`** as follows

For the topmost layer **`U`**`= 0` and **`Z`** is the batch normalized output from the corresponding (in this case last) encoder/classifier unit. **`Z^`** for topmost layer is defined as
```lua
   z_hat1 = nn.CMul(hiddens[i])(Z)
   z_hat2 = nn.CMul(hiddens[i])(Z)
   z_hat3 = nn.CMul(hiddens[i])(Z)
   z_hat34 = nn.Add(hiddens[i])(z_hat3)
   z_hatSigmoid34 = nn.Sigmoid()(z_hat34)
   z_hat234 = nn.CMulTable()({z_hat2, z_hatSigmoid34})
   z_hat5 = nn.CMul(hiddens_units)(Z)

   -- Z_hat = z^
   Z_hat = nn.CAddTable()({z_hat1, z_hat234, z_hat5})
```

For lower decoder units **`Z^`** is defined as
```lua
   
      u = nn.Linear()(previous_Z_hat)

      cu1 = nn.CMul(hidden_units)(u)
      du1 = nn.Add(hidden_units])(u)
      a1 = nn.CAddTable()({cu1, du1})
      cu2 = nn.CMul(hidden_units)(u)
      du2 = nn.Add(hidden_units)(u)
      a2 = nn.CAddTable()({cu2, du2})
      cu3 = nn.CMul(hidden_units)(u)
      du3 = nn.Add(hidden_units)(u)
      a3 = nn.CAddTable()({cu3, du3})
      cu4 = nn.CMul(hidden_units)(u)
      du4 = nn.Add(hidden_units)(u)
      a4 = nn.CAddTable()({cu4, du4})
      cu5 = nn.CMul(hidden_units)(u)
      du5 = nn.Add(hidden_units)(u)
      a5 = nn.CAddTable()({cu5, du5})

      z_hat1 = nn.CMulTable()({a1, z})
      z_hat2 = nn.CMulTable()({a3, z})
      z_hat3 = nn.Sigmoid()(nn.CAddTable()({z_hat2, a4}))
      z_hat4 = nn.CMulTable()({a2, z_hat3})
      Z_hat = nn.CAddTable()({z_hat1, z_hat4, a5})
```
`Z_hat` is `z^`. Final `Z_hat` is the output of decoder and feeds into the mean squared error criterion.

## Criterions
Negative log likelihood criterion is used for classification task.
```lua
   nll = nn.ClassNLLCriterion()
```
Mean squared error is used for the auxillary task.
```lua
   mse = nn.MSECriterion()
```
These two training criterions are combined using `eta` which determines weight for auxillary task. If `eta` is zero then the model is trained for classification only.
Combined criterion
```lua
   criterions = ParallelCriterion()
   criterions:add(nll)
   criterions:add(mse, eta)
```

## References
[1] Rasmus, Antti, Harri Valpola, and Tapani Raiko. "Lateral Connections in Denoising Autoencoders Support Supervised Learning." arXiv preprint arXiv:1504.08215 (2015).
