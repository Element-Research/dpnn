require 'torch'
require 'nn'
require 'nnx'
local _ = require 'moses'

-- create global dpnn table
dpnn = {}
dpnn.version = 2

unpack = unpack or table.unpack -- lua 5.2 compat

-- for testing:
require('dpnn.test')

-- extensions to existing modules
require('dpnn.Module')
require('dpnn.Container')
require('dpnn.Sequential')
require('dpnn.ParallelTable')
require('dpnn.LookupTable')
require('dpnn.SpatialBinaryConvolution')
require('dpnn.SimpleColorTransform')
require('dpnn.PCAColorTransform')

-- extensions to existing criterions
require('dpnn.Criterion')

-- extensions to make serialization more efficient
require('dpnn.SpatialMaxPooling')
require('dpnn.SpatialConvolution')
require('dpnn.SpatialConvolutionMM')
require('dpnn.SpatialBatchNormalization')
require('dpnn.BatchNormalization')

-- decorator modules
require('dpnn.Decorator')
require('dpnn.Serial')
require('dpnn.DontCast')
require('dpnn.NaN')

-- modules
require('dpnn.PrintSize')
require('dpnn.Convert')
require('dpnn.Constant')
require('dpnn.Collapse')
require('dpnn.ZipTable')
require('dpnn.ZipTableOneToMany')
require('dpnn.CAddTensorTable')
require('dpnn.ReverseTable')
require('dpnn.Dictionary')
require('dpnn.Inception')
require('dpnn.SoftMaxTree')
require('dpnn.SoftMaxForest')
require('dpnn.Clip')
require('dpnn.SpatialUniformCrop')
require('dpnn.SpatialGlimpse')
require('dpnn.WhiteNoise')
require('dpnn.ArgMax')
require('dpnn.CategoricalEntropy')
require('dpnn.TotalDropout')
require('dpnn.Kmeans')
require('dpnn.OneHot')
require('dpnn.SpatialRegionDropout')
require('dpnn.FireModule')
require('dpnn.SpatialFeatNormalization')

-- Noise Contrastive Estimation
require('dpnn.NCEModule')
require('dpnn.NCECriterion')

-- REINFORCE
require('dpnn.Reinforce')
require('dpnn.ReinforceGamma')
require('dpnn.ReinforceBernoulli')
require('dpnn.ReinforceNormal')
require('dpnn.ReinforceCategorical')
require('dpnn.VRClassReward')

-- criterions
require('dpnn.ModuleCriterion')
require('dpnn.BinaryLogisticRegression')
require('dpnn.SpatialBinaryLogisticRegression')

return dpnn
