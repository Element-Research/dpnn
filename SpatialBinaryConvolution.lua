-- Reference: http://arxiv.org/abs/1603.05279
-- We use floating point Matrix-Matrix multiplication as in SpatialConvolution.
-- Filters are made binary {-1, +1} using Sign.
-- Convolution output is scaled by L1-norm of the filters.

-- Inheriting nn/SpatialConvolution.

local SpatialBinaryConvolution, parent = torch.class('nn.SpatialBinaryConvolution', 'nn.SpatialConvolution')

function SpatialBinaryConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   parent.__init(self, nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   parent.noBias(self)

   self.iwh = self.nInputPlane * self.kW * self.kH 
   self.owh = self.nOutputPlane * self.kW * self.kH 
   self.train = true
end

function SpatialBinaryConvolution:training()
   self.train = true
end

function SpatialBinaryConvolution:evaluate()
   self.train = false
end

-- Function to binarize weights and compute L1 norms
function SpatialBinaryConvolution:binarizeWeight()
   self.tempWeight = self.tempWeight or self.weight.new()

   -- Grad Input alphas
   self.gradInputAlphas = self.gradInputAlphas or self.weight.new()
   self.gradInputAlphas:resize(self.nInputPlane)

   local temp = self.weight:transpose(1,2)
   self.tempWeight:resizeAs(temp):copy(temp)
   self.gradInputAlphas:norm(self.tempWeight:view(self.nInputPlane, -1), 1, 2)
   self.gradInputAlphas:div(self.owh) -- 1/owh

   -- alphas
   self.tempWeight:resizeAs(self.weight):copy(self.weight)
   self.alphas = self.alphas or self.weight.new()
   self.alphas:resize(self.nOutputPlane)
   self.alphas:norm(self.weight:view(self.nOutputPlane, -1), 1, 2)
   self.alphas:div(self.iwh) -- 1/iwh

   -- Binarize weights
   if not self.wmask then
      if torch.type(self.weight) == 'torch.CudaTensor' then
         self.wmask = torch.CudaTensor()
      else
         self.wmask = torch.ByteTensor()
      end
   end

   -- Binarizing weights
   self.weight.ge(self.wmask, self.weight, 0)
   self.weight[self.wmask] = 1
   self.weight.lt(self.wmask, self.weight, 0)
   self.weight[self.wmask] = -1
end

function SpatialBinaryConvolution:updateOutput(input)
   -- Binarize Weights
   self.binarizeWeight(self)

   -- Convolution
   self.output = parent.updateOutput(self, input)

   -- Scale output by alphas
   self._tempAlphas = self._tempAlphas or self.output.new()   
   self._tempAlphasExpanded = self._tempAlphasExpanded or self.output.new() 
   self._tempAlphasSamples = self._tempAlphasSamples or self.output.new()
   if self.output:nDimension() == 4 then
      local batchSize = self.output:size(1)
      local height = self.output:size(3)
      local width = self.output:size(4)

      self._tempAlphas = self.alphas:view(1, self.nOutputPlane, 1, 1)
      self._tempAlphasExpanded:expand(self._tempAlphas, batchSize,
                                      self.nOutputPlane, height, width)
      self._tempAlphasSamples:resizeAs(self._tempAlphasExpanded)
                             :copy(self._tempAlphasExpanded)
      self.output:cmul(self._tempAlphasSamples)
   else
      local height = self.output:size(2)
      local width = self.output:size(3)

      self._tempAlphas = self.alphas:view(self.nOutputPlane, 1, 1)
      self._tempAlphasExpanded:expand(self._tempAlphas, self.nOutputPlane,
                                      height, width)
      self._tempAlphasSamples:resizeAs(self._tempAlphasExpanded)
                             :copy(self._tempAlphasExpanded)
      self.output:cmul(self._tempAlphasSamples)
   end

   -- In evaluate mode.
   if not self.train then self.weight:copy(self.tempWeight) end

   return self.output 
end

function SpatialBinaryConvolution:updateGradInput(input, gradOutput)
   self.gradInput = parent.updateGradInput(self, input, gradOutput)

   -- Scale gradInput by gradAlphas
   self._tempGradAlphas = self._temp or self.gradInput.new()
   self._tempGradAlphasExpanded = self._temp or self.gradInput.new()
   self._tempGradAlphasSamples = self._temp or self.gradInput.new()
   if self.gradInput:nDimension() == 4 then
      local batchSize = self.gradInput:size(1)
      local height = self.gradInput:size(3)
      local width = self.gradInput:size(4)

      self._tempGradAlphas = self.gradInputAlphas:view(1, self.nInputPlane,
                                                       1, 1)
      self._tempGradAlphasExpanded:expand(self._tempGradAlphas,
                                          batchSize, self.nInputPlane,
                                          height, width)
      self._tempGradAlphasSamples:resizeAs(self._tempGradAlphasExpanded)
                                 :copy(self._tempGradAlphasExpanded)

      self.gradInput:cmul(self._tempGradAlphasSamples)
   else
      local height = self.gradInput:size(2)
      local width = self.gradInput:size(3)

      self._tempGradAlphas = self.gradInputAlphas:view(self.nInputPlane,
                                                       1, 1)
      self._tempGradAlphasExpanded:expand(self._tempGradAlphas,
                                          self.nInputPlane,
                                          height, width)
      self._tempGradAlphasSamples:resizeAs(self._tempGradAlphasExpanded)
                                 :copy(self._tempGradAlphasExpanded)

      self.gradInput:cmul(self._tempGradAlphasSamples)
   end
   return self.gradInput
end

function SpatialBinaryConvolution:accGradParameters(input, gradOutput, scale)

   parent.accGradParameters(self, input, gradOutput, scale)

   --[[
   Copy back floating point weights for weight update.
   This could be done individually after forward and backward, but to avoid
   additional copy is done at the end of backward.
   --]]

   self.weight:copy(self.tempWeight)
end

function SpatialBinaryConvolution:type(type, tensorCache)
   self.tempWeight = nil
   self.alphas = nil
   self.gradInputAlphas = nil
   self.wmask = nil

   self._tempAlphas = nil 
   self._tempAlphasExpanded = nil
   self._tempAlphasSamples = nil

   self._tempGradAlphas = nil
   self._tempGradAlphasExpanded = nil
   self._tempGradAlphasSamples = nil

   parent.type(self, type, tensorCache)
end

function SpatialBinaryConvolution:__tostring__()
   return "Binary Convolution: "..parent.__tostring__(self)
end
