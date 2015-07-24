local Dictionary, parent = torch.class("nn.Dictionary", "nn.LookupTable")

-- don't use this with optim (useless), use nn.LookupTable instead
function Dictionary:__init(dictSize, embeddingSize, accUpdate)
   parent.__init(self, dictSize, embeddingSize)
   assert(parent.__version >= 4, "Update the nn package to newest version")
   if accUpdate then -- this could save you lots of memory
      self:accUpdateOnly()
   else
      self.gradWeight:zero()
   end
   self.accUpdate = accUpdate
   self.inputs = {}
end

function Dictionary:accUpdateOnly()
   self.accUpdate = false
   return parent.accUpdateOnly(self)
end

function Dictionary:updateOutput(input)
   assert(torch.type(input) ~= 'torch.CudaTensor', "Expecting non-CudaTensor input")
   return parent.updateOutput(self, input)
end

function Dictionary:accGradParameters(input, gradOutput, scale)
   self.nBackward = (self.nBackward or 0) + input:nElement()
   -- this is why we dont want cuda inputs
   input:apply(function(k) self.inputs[k] = (self.inputs[k] or 0) + 1 end)
   return parent.accGradParameters(self, input, gradOutput, scale)
end

function Dictionary:sparseParameters()
   -- only return the parameters affected by the forward/backward
   local params, gradParams, scales = {}, {}, {}
   for k,nBackward in pairs(self.inputs) do
      params[k] = self.weight:select(1, k)
      if self.gradWeight then
         gradParams[k] = self.gradWeight:select(1, k)
      end
      scales[k] = 1
   end
   return params, gradParams, scales, self.weight:size(1)
end

function Dictionary:momentumGradParameters()
   -- get dense view of momGradParams
   if not self.momGradParams or _.isEmpty(self.momGradParams) then
      assert(not self.accUpdate, "cannot use momentum with accUpdate")
      self.momGradParams = {self.gradWeight:clone():zero()}
   end
   local momGradWeight = self.momGradParams[1]
   local momGradParams = {}
   -- only return the parameters affected by the forward/backward
   for k,nBackward in pairs(self.inputs) do
      momGradParams[k] = momGradWeight:select(1, k)
   end
   return momGradParams
end

function Dictionary:maxParamNorm(maxOutNorm, maxInNorm)
   maxOutNorm = self.maxOutNorm or maxOutNorm or self.maxInNorm or maxInNorm
   if not (maxOutNorm or maxInNorm) then
      return
   end
   
   for k,nBackward in pairs(self.inputs) do
      self.weight:narrow(1, k, 1):renorm(2, 2, maxOutNorm)
   end
end

function Dictionary:zeroGradParameters()
   if not self.accUpdate then
      for k,_ in pairs(self.inputs) do
         self.gradWeight:select(1, k):zero()
      end
   end
   for k,v in pairs(self.inputs) do
      self.inputs[k] = nil
   end
   self.nBackward = 0
end

function Dictionary:sharedClone()
   local clone = parent.sharedClone(self)
   clone.inputs = self.inputs
   return clone 
end

   
