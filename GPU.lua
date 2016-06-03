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

-- set the device of the decorated module
function GPU:setDevice(device)
   self.device = device or self.device
   
   local function recursiveModuleDevice(obj)
      if type(obj) == 'table' and not (torch.isTypeOf(obj, 'nn.GPU') or torch.type(obj) == 'torch.MultiCudaTensor') then
         for k,v in pairs(obj) do
            obj[k] = recursiveModuleDevice(v)
         end
      elseif torch.type(obj):match('torch.Cuda.*Tensor') then
         if obj:getDevice() ~= self.device then
            obj = obj:clone() -- this will reallocate it to self.device
            local newdevice = obj:getDevice()
            -- when nElement() == 0 newdevice is 0
            assert(newdevice == self.device or newdevice == 0)
         end
      end
      assert(obj ~= nil)
      return obj
   end
   
   assert(self.modules[1])
   self.modules[1] = cutorch.withDevice(self.device, function() return recursiveModuleDevice(self.modules[1]) end)
   return self
end
