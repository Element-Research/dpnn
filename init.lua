----------------------------------------------------------------------
--
-- Copyright (c) 2015 Nicholas Leonard
--
-- 
-- Permission is hereby granted, free of charge, to any person obtaining
-- a copy of this software and associated documentation files (the
-- "Software"), to deal in the Software without restriction, including
-- without limitation the rights to use, copy, modify, merge, publish,
-- distribute, sublicense, and/or sell copies of the Software, and to
-- permit persons to whom the Software is furnished to do so, subject to
-- the following conditions:
-- 
-- The above copyright notice and this permission notice shall be
-- included in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
-- EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
-- MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
-- NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
-- LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
-- OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
-- WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
-- 
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'nnx'
_ = require 'moses'

-- create global dpnn table
dpnn = {}
dpnn.version = 2

unpack = unpack or table.unpack -- lua 5.2 compat

-- for testing:
torch.include('dpnn', 'test.lua')

-- extensions to existing modules
torch.include('dpnn', 'Module.lua')
torch.include('dpnn', 'Container.lua')
torch.include('dpnn', 'Sequential.lua')
torch.include('dpnn', 'ParallelTable.lua')
torch.include('dpnn', 'LookupTable.lua')

-- extensions to existing criterions
torch.include('dpnn', 'Criterion.lua')

-- extensions to make serialization more efficient
torch.include('dpnn', 'SpatialMaxPooling.lua')
torch.include('dpnn', 'SpatialConvolution.lua')
torch.include('dpnn', 'SpatialConvolutionMM.lua')
torch.include('dpnn', 'SpatialBatchNormalization.lua')
torch.include('dpnn', 'BatchNormalization.lua')

-- decorator modules
torch.include('dpnn', 'Decorator.lua')
torch.include('dpnn', 'Serial.lua')
torch.include('dpnn', 'DontCast.lua')

-- modules
torch.include('dpnn', 'PrintSize.lua')
torch.include('dpnn', 'Convert.lua')
torch.include('dpnn', 'Constant.lua')
torch.include('dpnn', 'Collapse.lua')
torch.include('dpnn', 'ZipTable.lua')
torch.include('dpnn', 'ZipTableOneToMany.lua')
torch.include('dpnn', 'CAddTensorTable.lua')
torch.include('dpnn', 'ReverseTable.lua')
torch.include('dpnn', 'Dictionary.lua')
torch.include('dpnn', 'Inception.lua')
torch.include('dpnn', 'SoftMaxTree.lua')
torch.include('dpnn', 'SoftMaxForest.lua')
torch.include('dpnn', 'Clip.lua')
torch.include('dpnn', 'SpatialUniformCrop.lua')
torch.include('dpnn', 'SpatialGlimpse.lua')
torch.include('dpnn', 'WhiteNoise.lua')
torch.include('dpnn', 'ArgMax.lua')
torch.include('dpnn', 'CategoricalEntropy.lua')
torch.include('dpnn', 'TotalDropout.lua')
torch.include('dpnn', 'Kmeans.lua')
torch.include('dpnn', 'OneHot.lua')
torch.include('dpnn', 'SpatialRegionDropout.lua')
torch.include('dpnn', 'FireModule.lua')
torch.include('dpnn', 'SpatialFeatNormalization.lua')

-- REINFORCE
torch.include('dpnn', 'Reinforce.lua')
torch.include('dpnn', 'ReinforceBernoulli.lua')
torch.include('dpnn', 'ReinforceNormal.lua')
torch.include('dpnn', 'ReinforceCategorical.lua')
torch.include('dpnn', 'VRClassReward.lua')

-- criterions
torch.include('dpnn', 'ModuleCriterion.lua')
torch.include('dpnn', 'BinaryLogisticRegression.lua')
torch.include('dpnn', 'SpatialBinaryLogisticRegression.lua')
