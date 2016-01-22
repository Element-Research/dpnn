# Machine learning tutorials for torch7.

## Lateral Connections in Denoising Autoencoders Support Supervised Learning

In this tutorial we will understand how to implement ladder network as explained in [[1](http://arxiv.org/pdf/1504.08215.pdf)]. In this paper the authors have shown how unsupervised learning using a denoising autoencoder with lateral connections help improve the classification accuracy in supervised learning. 

The unsupervised learning (denoising) task supplements the supervised learning task (classification in this case). As in autoencoders this network has an encoder and a decoder. The output of encoder is also used for classification. The output of encoder is **`N`** dimensional where **`N`** is number of classes. This **`N`** dimensional vector is used for computing classification cost as well as feeds into the decoder.

### Denoising
Typically in denoising autoencoder the input samples are corrupted using Dropout ```nn.Dropout``` but in this paper the authors use isotropic Gaussian noise ```nn.WhiteNoise```.

### Lateral Connections in Autoencoder





### References
[1] Rasmus, Antti, Harri Valpola, and Tapani Raiko. "Lateral Connections in Denoising Autoencoders Support Supervised Learning." arXiv preprint arXiv:1504.08215 (2015).
