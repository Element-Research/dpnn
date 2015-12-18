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

   self.gradWeight = self.weight.new()
   self.gradWeight:resizeAs(self.weight)
   self.loss = 0 -- sum of within cluster error

   self.clusterSampleCount = self.weight.new()
   self.clusterSampleCount:resize(self.k)

   self:reset()
end

-- Reset
function Kmeans:reset(stdev)
   local stdev = stdev or 1
   self.weight:uniform(-stdev, stdev)
   self:resetNonWeight()
end

function Kmeans:resetNonWeight()
   self.gradWeight:zero()
   self.loss = 0
   self.clusterSampleCount:zero() 
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

-- Kmeans updateOutput (forward)
function Kmeans:updateOutput(input)
   local inputDim = input:nDimension()
   assert(inputDim == 2, "Incorrect input dimensionality. Expecting 2D.")

   local batchSize = input:size(1)
   local dim = input:size(2)
   assert(dim == self.dim, "Dimensionality of input and weight don't match.")

   assert(input:isContiguous(), "Input is not contiguous.")

   self.output = self.output or self.weight.new()
   self.output:resize(batchSize)

   -- a sample copied k times to compute distance between sample and weight
   self._expandedSamples = self._expandedSamples or self.weight.new()

   -- distance between a sample and weight
   self._clusterDistances = self._clusterDistances or self.weight.new()

   self._temp = self._temp or input.new()
   self._tempExpanded = self._tempExpanded or input.new()

   -- Expanding inputs
   self._temp:view(input, 1, batchSize, self.dim)
   self._tempExpanded:expand(self._temp, self.k, batchSize, self.dim)
   self._expandedSamples:resize(self.k, batchSize, self.dim)
                        :copy(self._tempExpanded)

   -- Expanding weights
   self._tempWeight = self._tempWeight or self.weight.new()
   self._tempWeightExp = self._tempWeightExp or self.weight.new()
   self._expandedWeight = self._expanedWeight or self.weight.new()
   self._tempWeight:view(self.weight, k, 1, self.dim)
   self._tempWeightExp:expand(self._tempWeight, self._expandedSamples:size())
   self._expandedWeight:resize(self.k, batchSize, self.dim)
                       :copy(self._tempWeightExp)

   -- x-c
   self._expandedSamples:add(-1, self._expandedWeight)
   -- Squared Euclidean distance
   self._clusterDistances:norm(self._expandedSamples, 2, 3)
   self._clusterDistances:pow(2)
   self._clusterDistances:resize(self.k, batchSize)

   self._minScore = self._minScore or self.weight.new()
   self._minIndx = self._minIndx or self.weight.new()
   self._minScore, self._minIndx = self._clusterDistances:min(1)
   self._minIndx = self._minIndx:view(-1)
   self.output:copy(self._minIndx)
   self.loss = self.loss + self._minScore:sum()
   
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
-- With this update rule and gradient descent will be negative the gradWeights.
function Kmeans:accGradParameters(input, gradOutput, scale)
   local scale = self.scale or scale or 1
   assert(scale > 0 , " Scale has to be positive.")

   -- Update cluster sample count
   local batchSize = input:size(1)
   self._cscAdder = self._cscAdder or self.weight.new()
   self._cscAdder:resize(batchSize):fill(1)
   self.clusterSampleCount:indexAdd(1, self._minIndx, self._cscAdder)

   -- added corresponding x-c to gradWeight
   for i=1, batchSize do
      self.gradWeight[self._minIndx[i]]
                :add(self._expandedSamples[self._minIndx[i]][i])
   end 

   -- 1/n * sum_i (x-c)
   for i=1, self.k do
      if self.clusterSampleCount[i] > 0 then
         self.gradWeight[i]:div(self.clusterSampleCount[i])
      end
   end

   -- scale * 1/n * sum_i (x-c)
   if scale ~= 1 then self.gradWeight:mul(scale) end

   -- Negate gradWeight such gradient descent updates centers correctly
   self.gradWeight:mul(-1)
end

function Kmeans:type(type, tensorCache)
   if type then
      -- prevent premature memory allocations
      self._expandedSamples = nil
      self._clusterDistances = nil
      self._temp = nil
      self._tempExpanded = nil
      self._tempWeight = nil
      self._tempWeightExp = nil
      self._expandedWeight = nil
      self._minScore = nil
      self._minIndx = nil
      self._cscAdder = nil
   end
   return parent.type(self, type, tensorCache)
end
