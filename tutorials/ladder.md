# Machine learning tutorials for torch7.

## Lateral Connections in Denoising Autoencoders Support Supervised Learning

In this tutorial we will understand how to implement ladder network as explained in [[1](http://arxiv.org/pdf/1504.08215.pdf)]. In this paper the authors have shown how unsupervised learning using a denoising autoencoder with lateral connections help improve the classification accuracy in supervised learning.

To produce results as mentioned in the paper please run following command
```
   th tutorials/ladder.lua --verbose --eta 500 --epochs 100 --learningRate 0.002 --linearDecay --endLearningRate 0 --startEpoch 50 --useCuda --deviceId 1 --noiseSigma 0.3 --useBatchNorm --batchSize 100 --adam --noValidation --attempts 10
```

The unsupervised learning (denoising) task supplements the supervised learning task (classification in this case). As in autoencoders this network has an encoder and a decoder. The output of encoder is also used for classification. The output of encoder is **`N`** dimensional where **`N`** is number of classes. This **`N`** dimensional vector is used for computing classification cost as well as feeds into the decoder.

### Denoising
Typically in denoising autoencoder the input samples are corrupted using Dropout ```nn.Dropout``` but in this paper the authors use isotropic Gaussian noise ```nn.WhiteNoise```.

### Lateral Connections in Autoencoder
Units in encoder are laterally connected to corresponding unit in the decoder. The vertical connection of the decoder is standard fully connected layer. Lateral connection for neuron `i` is defined by
```
   z^_i = a_i1 * z_i + a_i2 * sigmoid(a_i3 + a_i4) + a_i5
```
where 
```
   a_ij = c_ij * u_i + d_ij
```
**`u`** is output of decoder unit. **`z`** is output of corresponding encoder unit (this is lateral connection, decoder takes output from its previous unit as well as corresponding encoder unit). **`c_j`** and **`d_j`** are trainable weight vectors. This forms the crux of the ladder network. This can be easily implemented using **`nngraph`** as follows
```lua
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
`Z_hat` is `z^`. Please check code in *tutorials/ladder.lua*.

### Criterions
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

### References
[1] Rasmus, Antti, Harri Valpola, and Tapani Raiko. "Lateral Connections in Denoising Autoencoders Support Supervised Learning." arXiv preprint arXiv:1504.08215 (2015).
