-- Online (Hard) Kmeans layer.
local Kmeans, parent = torch.class('nn.Kmeans', 'nn.Module')

function Kmeans:__init(k, dim, scale)
   self.k = k
   self.dim = dim

   -- scale for online kmean update
   self.scale = scale

   assert(k > 0, "Clusters cannot be 0 or negative.")
   assert(dim > 0, "Dimensionality cannot be 0 or negative.")

   -- Kmeans centers -> self.weight
   self.weight = torch.Tensor()
   self.weight:resize(self.k, self.dim)
   self.weight:uniform(-1, 1)

   self.gradWeight = self.weight.new()
   self.gradWeight:resizeAs(self.weight):zero()
   self.loss = 0 -- sum of within cluster error
end

-- Initialize Kmeans weight with random samples from input.
function Kmeans:initRandom(input)
   local inputDim = input:nDimension()
   assert(inputDim == 2, "Incorrect input dimensionality. Expecting 2D.")

   local noOfSamples = input:size(1)
   local dim = input:size(2)
   assert(dim == self.dim, "Dimensionality of input and weight don't match.")
   assert(noOfSamples >= self.k, "Need atleast k samples for initialization.")

   local indices = torch.zeros(self.k)
   indices:random(1, noOfSamples)

   for i=1, self.k do
      self.weight[i]:copy(input[indices[i]])
   end
end


function Kmeans:updateOutput(input)
   local inputDim = input:nDimension()
   assert(inputDim == 2, "Incorrect input dimensionality. Expecting 2D.")

   local noOfSamples = input:size(1)
   local dim = input:size(2)
   assert(dim == self.dim, "Dimensionality of input and weight don't match.")

   assert(input:isContiguous(), "Input is not contiguous.")

   self.output = self.output or self.weight.new()
   self.output:resize(noOfSamples)

   self.gradWeight:zero()
   self._clusterSampleCount = self._clusterSampleCount or self.weight.new()
   self._clusterSampleCount:resize(self.k):zero()

   -- a sample copied k times to compute distance between sample and weight
   self._expandedSample = self._expandedSample or self.weight.new()

   -- distance between a sample and weight
   self._clusterDistances = self._clusterDistances or self.weight.new()

   self._temp = self._temp or input.new()
   self._minScore = self._minScore or self.weight.new()
   self._minIndx = self._minIndx or self.weight.new()
   self.loss = 0
   local clusterIndx
   for i=1, noOfSamples do
      self._temp:expand(input[{{i}}], self.k, self.dim)
      self._expandedSample:resize(self._temp:size()):copy(self._temp)

      -- (x-c)
      self._expandedSample:add(-1, self.weight)
      -- Euclidean distance ||x-c||
      self._clusterDistances:norm(self._expandedSample, 2, 2)
      -- Squared Euclidean distance ||x-c||^2
      self._clusterDistances:pow(2)

      self._minScore, self._minIndx = self._clusterDistances:min(1)
      clusterIndx = self._minIndx[1][1]
      self.output[i] = clusterIndx
      self.loss = self.loss + self._minScore[1][1]

      -- center update rule c -> c + 1/n (x-c)
      -- Saving (x-c). Negate gradWeight in accGradParameters since
      -- we use negative gradient in gradient descent.
      self.gradWeight[clusterIndx]:add(self._expandedSample[clusterIndx])
      self._clusterSampleCount[clusterIndx] = 
                                    self._clusterSampleCount[clusterIndx] + 1
   end
   return self.output
end


function Kmeans:updateOutput_single(input)
   local inputDim = input:nDimension()
   assert(inputDim == 2, "Incorrect input dimensionality. Expecting 2D.")

   local noOfSamples = input:size(1)
   local dim = input:size(2)
   assert(dim == self.dim, "Dimensionality of input and weight don't match.")

   self.output = self.output or self.weight.new()
   self.output:resize(noOfSamples)

   self.gradWeight:zero()
   self._clusterSampleCount = self._clusterSampleCount or self.weight.new()
   self._clusterSampleCount:resize(self.k):zero()

   -- a sample copied k times to compute distance between sample and weight
   self._expandedSample = self._expandedSample or self.weight.new()

   -- distance between a sample and weight
   self._clusterDistances = self._clusterDistances or self.weight.new()

   self._temp = self._temp or input.new()
   self._minScore = self._minScore or self.weight.new()
   self._minIndx = self._minIndx or self.weight.new()
   self.loss = 0
   local clusterIndx
   for i=1, noOfSamples do
      self._temp:expand(input[{{i}}], self.k, self.dim)
      self._expandedSample:resize(self._temp:size()):copy(self._temp)

      -- (x-c)
      self._expandedSample:add(-1, self.weight)
      -- Euclidean distance ||x-c||
      self._clusterDistances:norm(self._expandedSample, 2, 2)
      -- Squared Euclidean distance ||x-c||^2
      self._clusterDistances:pow(2)

      self._minScore, self._minIndx = self._clusterDistances:min(1)
      clusterIndx = self._minIndx[1][1]
      self.output[i] = clusterIndx
      self.loss = self.loss + self._minScore[1][1]

      -- center update rule c -> c + 1/n (x-c)
      -- Saving (x-c). Negate gradWeight in accGradParameters since
      -- we use negative gradient in gradient descent.
      self.gradWeight[clusterIndx]:add(self._expandedSample[clusterIndx])
      self._clusterSampleCount[clusterIndx] = 
                                    self._clusterSampleCount[clusterIndx] + 1
   end
   return self.output
end

-- Kmeans has its own criterion hence gradInput are zeros
function Kmeans:updateGradInput(input, gradOuput)
   self.gradInput = self.weight.new()
   self.gradInput:resize(input:size(1), input:size(2)):zero()
   return self.gradInput
end

-- We define kmeans update rule as c -> c + scale * 1/n * sum_i (x-c).
-- n is no. of x's belonging to c.
-- With this update rule you can do online as well full update.
-- We are saving (x-c) in gradWeight in updateOutput.
-- We will update weight in this function and then set gradWeight to zero so
-- that gradient descent update won't make any change.
function Kmeans:accGradParameters(input, gradOutput, scale)
   local scale = self.scale or scale or 1
   assert(scale > 0 , " Scale has to be positive.")

   -- 1/n * sum_i (x-c)
   for i=1, self.k do
      if self._clusterSampleCount[i] > 0 then
         self.gradWeight[i]:div(self._clusterSampleCount[i])
      end
   end

   -- scale * 1/n * sum_i (x-c)
   if scale ~= 1 then self.gradWeight:mul(scale) end

   -- Update kmeans weight
   self.weight:add(self.gradWeight)

   -- zeroing gradWeight so the gradient descent update won't affect weight.
   self.gradWeight:zero()
end

function Kmeans:type(type, tensorCache)
   if type then
      -- prevent premature memory allocations
      self._temp = nil
      self._minScore = nil
      self._minIndx = nil
      self._expandedSample = nil
      self._clusterDistances = nil
      self._clusterSampleCount = nil
   end
   return parent.type(self, type, tensorCache)
end
