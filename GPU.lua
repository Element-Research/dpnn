local GPU, parent = nn.GPU, nn.Container

function GPU:maxParamNorm(...)
   local args = {...}
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.maxParamNorm(self, unpack(args)) end)
   else
      return parent.maxParamNorm(self, unpack(args))
   end
end

function GPU:gradParamClip(...)
   local args = {...}
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.gradParamClip(self, unpack(args)) end)
   else
      return parent.gradParamClip(self, unpack(args))
   end
end

function GPU:weightDecay(...)
   local args = {...}
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.weightDecay(self, unpack(args)) end)
   else
      return parent.weightDecay(self, unpack(args))
   end
end

function GPU:momentumGradParameters(...)
   local args = {...}
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.momentumGradParameters(self, unpack(args)) end)
   else
      return parent.momentumGradParameters(self, unpack(args))
   end
end

function GPU:updateGradParameters(...)
   local args = {...}
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.updateGradParameters(self, unpack(args)) end)
   else
      return parent.updateGradParameters(self, unpack(args))
   end
end

function GPU:checkParameters(...)
   local args = {...}
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.checkParameters(self, unpack(args)) end)
   else
      return parent.checkParameters(self, unpack(args))
   end
end

function GPU:contiguousInput(...)
   local args = {...}
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.contiguousInput(self, unpack(args)) end)
   else
      return parent.contiguousInput(self, unpack(args))
   end
end

function GPU:toBatch(...)
   local args = {...}
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.toBatch(self, unpack(args)) end)
   else
      return parent.toBatch(self, unpack(args))
   end
end

function GPU:fromBatch(...)
   local args = {...}
   if self._type == 'torch.CudaTensor' then
      return cutorch.withDevice(self.device, function() return parent.fromBatch(self, unpack(args)) end)
   else
      return parent.fromBatch(self, unpack(args))
   end
end
