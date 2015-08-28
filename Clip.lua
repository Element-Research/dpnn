------------------------------------------------------------------------
--[[ Clip ]]--
-- clips values within minval and maxval
------------------------------------------------------------------------
local Clip, parent = torch.class("nn.Clip", "nn.Module")

function Clip:__init(minval, maxval)
   assert(torch.type(minval) == 'number')
   assert(torch.type(maxval) == 'number')
   self.minval = minval
   self.maxval = maxval
   parent.__init(self)
end

function Clip:updateOutput(input)
   -- bound results within height and width
   self._mask = self._mask or input.new()
   self._byte = self._byte or torch.ByteTensor()
   self.output:resizeAs(input):copy(input)
   self._mask:gt(self.output, self.maxval)
   local byte = torch.type(self.output) == 'torch.CudaTensor' and self._mask 
      or self._byte:resize(self._mask:size()):copy(self._mask)
   self.output[byte] = self.maxval
   self._mask:lt(self.output, self.minval)
   byte = torch.type(self.output) == 'torch.CudaTensor' and self._mask 
      or self._byte:resize(self._mask:size()):copy(self._mask)
   self.output[byte] = self.minval
   return self.output
end

function Clip:updateGradInput(input, gradOutput)
   self.gradInput:set(gradOutput)
   return self.gradInput
end

