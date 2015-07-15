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

-- for testing:
torch.include('dpnn', 'test.lua')

-- extensions to existing modules
torch.include('dpnn', 'Module.lua')
torch.include('dpnn', 'Container.lua')
torch.include('dpnn', 'Sequential.lua')
torch.include('dpnn', 'ParallelTable.lua')

-- extensions to make serialization more efficient
torch.include('dpnn', 'SpatialMaxPooling.lua')
torch.include('dpnn', 'SpatialConvolution.lua')
torch.include('dpnn', 'SpatialConvolutionMM.lua')

-- decorator modules
torch.include('dpnn', 'Decorator.lua')
torch.include('dpnn', 'Serial.lua')
torch.include('dpnn', 'DontCast.lua')

-- modules
torch.include('dpnn', 'PrintSize.lua')
torch.include('dpnn', 'Convert.lua')
torch.include('dpnn', 'Collapse.lua')
torch.include('dpnn', 'ZipTable.lua')
torch.include('dpnn', 'ReverseTable.lua')
torch.include('dpnn', 'Dictionary.lua')
torch.include('dpnn', 'Inception.lua')
torch.include('dpnn', 'SoftMaxTree.lua')
torch.include('dpnn', 'SoftMaxForest.lua')
torch.include('dpnn', 'SpatialUniformCrop.lua')

-- criterions
torch.include('dpnn', 'ModuleCriterion.lua')
