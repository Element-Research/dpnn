------------------------------------------------------------------------
--[[ BinaryLogisticRegression ]]--
-- Takes an image of size batchSize x 1 or  just batchSize as input.
-- Computes Binary Logistic Regression Cost.
-- Useful for 2 class classification.
------------------------------------------------------------------------

local BinaryLogisticRegression, parent = torch.class('nn.BinaryLogisticRegression', 'nn.Criterion')

function BinaryLogisticRegression:__init(sizeAverage)
   parent.__init(self)
   if sizeAverage ~= nil then
      self.sizeAverage = sizeAverage
   else
      self.sizeAverage = true
   end
end

function BinaryLogisticRegression:updateOutput(input, target)
   local inputDim = input:nDimension()
   local targetDim = target:nDimension()

   -- Check dimensions of input and target
   assert(inputDim == 1 or inputDim == 2,
                                  "Input:Expecting batchSize or batchSize x 1")
   assert(targetDim == 1 or targetDim == 2,
                                 "Target:Expecting batchSize or batchSize x 1")
   if inputDim == 2 then
      assert(input:size(1)==1 or input:size(2)==1, 
                                        "Input: Expecting batchSize x 1.")
   end
   if targetDim == 2 then
      assert(target:size(1)==1 or target:size(2)==1,
                                        "Target: Expecting batchSize x 1.")
   end

   local inputElements = input:nElement()
   local targetElements = target:nElement()

   assert(inputElements == targetElements,
                           "No of input and target elements should be same.")

   self._k = inputElements
   local input = input:view(-1)
   local target = target:view(-1)

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
      return self._logCoeff:sum()/(self._k)
   else
      return self._logCoeff:sum()
   end
end

function BinaryLogisticRegression:updateGradInput(input, target)
   self.gradInput = self.gradInput or input.new()
   local gradInput = self.gradInput
   gradInput:resize(input:size()):copy(target)
   gradInput:mul(-1)
   gradInput:cmul(self._baseExponents)
   gradInput:cdiv(self._coeff)
   if self.sizeAverage then
      gradInput:div(self._k)
   end
   return gradInput
end

function BinaryLogisticRegression:type(type, tensorCache)
   if type then
      self._baseExponents = nil
      self._coeff = nil
      self._logCoeff = nil
   end
   return parent.type(self, type, tensorCache)
end
