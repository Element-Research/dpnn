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
end

-- Function to binarize weights and compute L1 norms
function binarizeWeight(self)
   -- Grad Input alphas
   self.gradInputAlphas = self.gradInputAlphas or self.weight.new()
   self.gradInputAlphas:resize(self.nInputPlane)
   for i=1, self.nInputPlane do
      self.gradInputAlphas[i] = self.weight[{{}, {i}}]:norm(1)
   end
   self.gradInputAlphas:div(self.owh) -- 1/owh

   -- alphas and binarize filters.
   self.tempWeight = self.tempWeight or self.weight.new()
   self.tempWeight:resizeAs(self.weight):copy(self.weight)
   self.alphas = self.alphas or torch.Tensor()
   self.alphas:resize(self.nOutputPlane)
   for i=1, self.nOutputPlane do
      self.alphas[i] = self.weight[i]:norm(1)
      self.weight[i]:apply(function(x) 
                              if x>=0 then return 1 else return -1 end
                          end)
   end
   self.alphas:div(self.iwh) -- 1/iwh
end

function SpatialBinaryConvolution:updateOutput(input)
   -- Binarize Weights
   binarizeWeight(self)

   -- Convolution
   self.output = parent.updateOutput(self, input)

   -- Scale alphas
   if self.output:nDimension() == 4 then
      for i=1, self.nOutputPlane do
         self.output[{{}, {i}}]:mul(self.alphas[i])
      end
   else
      for i=1, self.nOutputPlane do
         self.output[{{i}}]:mul(self.alphas[i])
      end
   end
   return self.output 
end

function SpatialBinaryConvolution:updateGradInput(input, gradOutput)
   self.gradInput = parent.updateGradInput(self, input, gradOutput)

   -- Scale gradInput accordingly
   if self.gradInput:nDimension() == 4 then
      for i=1, self.nInputPlane do
         self.gradInput[{{}, {i}}]:mul(self.gradInputAlphas[i])
      end
   else
      for i=1, self.nInputPlane do
         self.gradInput[{{i}}]:mul(self.gradInputAlphas[i])
      end
   end
   return self.gradInput
end

function SpatialBinaryConvolution:accGradParameters(input, gradOutput, scale)

   assert(self.gradWeight:sum()==0, 
          "Called zeroGradParameters before backward.")
   
   parent.accGradParameters(self, input, gradOutput, scale)

   -- Scale gradWeight by alphas
   for i=1, self.nOutputPlane do
      self.gradWeight[i]:mul(self.alphas[i])
   end

   -- Copy back floating point weights for weight update.
   self.weight:copy(self.tempWeight)
end

function SpatialBinaryConvolution:type(type, tensorCache)
   self.tempWeight = self.tempWeight and torch.Tensor()
   parent.type(self, type, tensorCache)
end

function SpatialBinaryConvolution:__tostring__()
   return "Binary Convolution: "..parent.__tostring__(self)
end
