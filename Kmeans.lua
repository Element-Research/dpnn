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
end

function Kmeans:updateOutput(input)
   local inputDim = input:nDimension()
   assert(inputDim == 2, "Incorrect input dimensionality. Expecting 2D.")

   local noOfSamples = input:size(1)
   local dim = input:size(2)
   assert(dim == self.dim, "Dimensionality of input and centers don't match.")

   self.output = self.output or self.centers.new()
   self.output:resize(noOfSamples)

   -- a sample copied k times to compute distance between sample and centers
   self._expandedSample = self._expandedSample or self.centers.new()

   -- distance between a sample and centers
   self._clusterDistances = self._clusterDistances or self.centers.new()

   self._temp = self._temp or input.new()
   self._minScore = self._minScore or self.centers.new()
   self._minIndx = self._minIndx or self.centers.new()
   for i=1, noOfSamples do
      self._temp:expand(input[{{i}}], self.k, self.dim)
      self._expandedSample:resize(self._temp:size()):copy(self._temp)

      -- Squared Euclidean distance
      self._expandedSample:add(-1, self.centers)
      self._clusterDistances:norm(self._expandedSample, 2, 2)
      self._clusterDistances:pow(2)

      self._minScore, self._minIndx = self._clusterDistances:min(1)
      self.output[i] = self._minIndx[1][1]
   end
   return self.output
end

function Kmeans:type(type, tensorCache)
   if type then
      -- prevent premature memory allocations
      self._temp = nil
      self._minScore = nil
      self._minIndx = nil
      self._expandedSample = nil
      self._clusterDistances = nil
   end
   return parent.type(self, type, tensorCache)
end
