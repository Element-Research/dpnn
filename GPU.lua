local GPU, parent = nn.GPU, nn.Container

function GPU:sharedClone()
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.zeroGradParameters(self) end)
   else
      parent.zeroGradParameters(self)
   end
end

function GPU:maxParamNorm(...)
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.maxParamNorm(self, ...) end)
   else
      return parent.maxParamNorm(self, ...)
   end
end

function GPU:gradParamClip(...)
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.gradParamClip(self, ...) end)
   else
      return parent.gradParamClip(self, ...)
   end
end

function GPU:weightDecay(...)
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.weightDecay(self, ...) end)
   else
      return parent.weightDecay(self, ...)
   end
end

function GPU:momentumGradParameters(...)
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.momentumGradParameters(self, ...) end)
   else
      return parent.momentumGradParameters(self, ...)
   end
end

function GPU:updateGradParameters(...)
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.updateGradParameters(self, ...) end)
   else
      return parent.updateGradParameters(self, ...)
   end
end

function GPU:checkParameters(...)
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.checkParameters(self, ...) end)
   else
      return parent.checkParameters(self, ...)
   end
end

function GPU:contiguousInput(...)
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.contiguousInput(self, ...) end)
   else
      return parent.contiguousInput(self, ...)
   end
end

function GPU:toBatch(...)
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.toBatch(self, ...) end)
   else
      return parent.toBatch(self, ...)
   end
end

function GPU:fromBatch(...)
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.fromBatch(self, ...) end)
   else
      return parent.fromBatch(self, ...)
   end
end
