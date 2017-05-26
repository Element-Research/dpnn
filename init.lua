require 'torch'
require 'nn'
local _ = require 'moses'
-- create global dpnn table
dpnn = {}
dpnn.version = 2

unpack = unpack or table.unpack -- lua 5.2 compat

function dpnn.require(packagename)
   assert(torch.type(packagename) == 'string')
   local success, message = pcall(function() require(packagename) end)
   if not success then
      print("missing package "..packagename..": run 'luarocks install nnx'")
      error(message)
   end
end

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
require('dpnn.Serial')

-- modules
require('dpnn.ReverseTable')
require('dpnn.Inception')
require('dpnn.Clip')
require('dpnn.SpatialUniformCrop')
require('dpnn.SpatialGlimpse')
require('dpnn.ArgMax')
require('dpnn.CategoricalEntropy')
require('dpnn.TotalDropout')
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

-- REINFORCE criterions
require('dpnn.VRClassReward')
require('dpnn.BinaryClassReward')

-- criterions
require('dpnn.BinaryLogisticRegression')
require('dpnn.SpatialBinaryLogisticRegression')

return dpnn
