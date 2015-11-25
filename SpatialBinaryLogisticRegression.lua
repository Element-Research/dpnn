------------------------------------------------------------------------
--[[ SpatialBinaryLogisticRegression ]]--
-- Takes an image of size batchSize x nChannel x width x height as input.
-- Computes Binary Logistic Regression Cost.
-- Useful for 2 class pixel classification.
------------------------------------------------------------------------

local SpatialBinaryLogisticRegression, parent = torch.class('nn.SpatialBinaryLogisticRegression', 'nn.Criterion')

function SpatialBinaryLogisticRegression:__init()
   parent.__init(self)
   self.sizeAverage = true
end

function SpatialBinaryLogisticRegression:updateOutput(input, target)
   local inputDim = input:nDimension()
   local targetDim = target:nDimension()

   -- Check dimensions of input and target
   assert(inputDim == targetDim, "nDimension of input and target don't match.")
   assert(inputDim == 4 or inputDim == 3, "Expecting image or batch on images")

   for i=1,inputDim do
      assert(input:size(i) == target:size(i),
                                  "Input and target dimensions don't match.")
   end

   -- Check batch or single image
   if inputDim == 4 then
      self._isBatch = true
      assert(input:size(2) == 1, "No. of channels should be 1.")
      self._k = input:size(1)
      self._h = input:size(3)
      self._w = input:size(4)
   else
      self._isBatch = false
      assert(input:size(1) == 1, "No. of channels should be 1.")
      self._k = 1
      self._h = input:size(2)
      self._w = input:size(3)
   end

   self._baseExponents = self._baseExponents or input.new()
   self._coeff = self._coeff or input.new()
   self._logCoeff = self._logCoeff or input.new()

   --Compute exponent = -target*input
   self._baseExponents:resize(input:size()):copy(input)
   self._baseExponents:cmul(target)
   self._baseExponents:mul(-1)
   -- Compute exp(exponent)
   self._baseExponents:exp()

   self._coeff:resize(input:size()):copy(self._baseExponents)
   self._coeff:add(1)

   self._logCoeff:resize(input:size()):copy(self._coeff)
   self._logCoeff:log()

   if self.sizeAverage then
      return self._logCoeff:sum()/(2 * self._k * self._h * self._w)
   else
      return self._logCoeff:sum()/(2 * self._h * self._w)
   end
end

function SpatialBinaryLogisticRegression:updateGradInput(input, target)
   self.gradInput = self.gradInput or input.new()
   local gradInput = self.gradInput
   gradInput:resize(target:size()):copy(target)
   gradInput:mul(-1)
   gradInput:cmul(self._baseExponents)
   gradInput:cdiv(self._coeff)
   if self.sizeAverage then
      gradInput:div(2 * self._k * self._h * self._w)
   else
      gradInput:div(2 * self._h * self._w)
   end
   return gradInput
end
