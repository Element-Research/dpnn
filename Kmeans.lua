-- Online (Hard) Kmeans layer.
local Kmeans, parent = torch.class('nn.Kmeans', 'nn.Module')

function Kmeans:__init(k, dim, mean, std, centers)
   self.k = k
   self.dim = dim
   self.mean = mean or 0
   self.mean = std or 0.01
   assert(k > 0, "Clusters cannot be 0 or negative.")
   assert(dim > 0, "Dimensionality cannot be 0 or negative.")

   if centers ~= nil then
      self.centers = centers
   else
      self.centers = torch.Tensor()
      self.centers:resize(self.k, self.dim)
      self.centers:normal(self.mean, self.std)
   end
   self.gradWeight = self.centers.new()
   self.gradWeight:resizeAs(centers)
end

function Kmeans:updateOutput(input)
   local inputDim = input:nDimension()
   assert(inputDim == 2, "Incorrect input dimensionality. Expecting 2D.")

   local noOfSamples = input:size(1)
   local dim = input:size(2)
   assert(dim == self.dim, "Dimensionality of input and centers don't match.")

   self.output = self.output or self.centers.new()
   self.output:resize(noOfSamples)

   self.gradWeight:zero()
   self._clusterSampleCount = self._clusterSampleCount or self.centers.new()
   self._clusterSampleCount:resize(self.k):zero()

   -- a sample copied k times to compute distance between sample and centers
   self._expandedSample = self._expandedSample or self.centers.new()

   -- distance between a sample and centers
   self._clusterDistances = self._clusterDistances or self.centers.new()

   self._temp = self._temp or input.new()
   self._minScore = self._minScore or self.centers.new()
   self._minIndx = self._minIndx or self.centers.new()
   local clusterIndx
   for i=1, noOfSamples do
      self._temp:expand(input[{{i}}], self.k, self.dim)
      self._expandedSample:resize(self._temp:size()):copy(self._temp)

      -- (x-c)
      self._expandedSample:add(-1, self.centers)
      -- Euclidean distance ||x-c||
      self._clusterDistances:norm(self._expandedSample, 2, 2)
      -- Squared Euclidean distance ||x-c||^2
      self._clusterDistances:pow(2)

      self._minScore, self._minIndx = self._clusterDistances:min(1)
      clusterIndx = self._minIndx[1][1]
      self.output[i] = clusterIndx

      -- center update rule c -> c + 1/n (x-c)
      -- Saving (x-c). Negate gradWeight in accGradParameters since
      -- we use negative gradient in gradient descent.
      self.gradWeight[clusterIndx]:add(self._expandedSample[clusterIndx])
      self._clusterSampleCount[clusterIndx] = 
                                    self._clusterSampleCount[clusterIndx] + 1
   end
   return self.output
end

-- Kmeans has its own criterion hence no gradInput is generated
function Kmeans:updateGradInput(input, gradOuput)
   return
end

-- We define kmeans update rule as c -> c + beta * 1/n * sum_i (x-c).
-- n is no. of x's belonging to c.
-- With this update rule you can do online as well full update.
-- We are saving (x-c) in gradWeight in updateOutput.
-- We will update centers in this function and then set gradWeight to zero so
-- that gradient descent update won't make any change.
function Kmeans:accGradParameters(input, gradOutput, beta)
   local beta = beta or 1
   self.beta = beta
   assert(self.beta <= 1, "Beta greater than 1.")

   self.alpha = 1 - self.beta

   -- 1/n * sum_i (x-c)
   for i=1, self.k do
      if self._clusterSampleCount[i] > 0 then
         self.gradWeight[i]:div(self._clusterSampleCount[i])
      end
   end

   -- beta * 1/n * sum_i (x-c)
   self.gradWeight:mul(beta)

   -- Update kmeans centers
   self.centers:add(self.gradWeight)

   -- zeroing gradWeight so the gradient descent update won't affect centers.
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
