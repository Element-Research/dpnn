local Dictionary, parent = torch.class("nn.Dictionary", "nn.LookupTable")

-- don't use this with optim (useless), use nn.LookupTable instead
function Dictionary:__init(dictSize, embeddingSize, accUpdate)
   parent.__init(self, dictSize, embeddingSize)
   if accUpdate then -- this could save you lots of memory
      self.gradWeight = nil
   end
   self.accUpdate = accUpdate
   self.dpnn_sparseParameters = true --disable for use with optim 
end

function Dictionary:parameters()
   if self.dpnn_sparseParameters then
      -- only return the parameters affected by the forward/backward
      local params, gradParams, scales = {}, {}, {}
      for k,nBackward in pairs(self.inputs) do
         local kscale = self:scaleUpdateByKey(k)
         params[k] = self.weight:select(1, k)
         if self.gradWeight then
            gradParams[k] = self.gradWeight:select(1, k)
         end
         scales[k] = self:scaleUpdateByKey(k)
      end
      return params, gradParams, scales, self.weight:size(1)
   end
   return parent.parameters(self)
end

function Dictionary:momentumGradParameters()
   if self.dpnn_sparseParameters then
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
   return parent.momentumGradParameters(self)
end

function Dictionary:maxParamNorm(maxOutNorm, maxInNorm)
   maxOutNorm = self.maxOutNorm or maxOutNorm or self.maxInNorm or maxInNorm
   if not (maxOutNorm or maxInNorm) then
      return
   end
   
   for k,nBackward in pairs(self.inputs) do
      self.weight:narrow(1, k, 1):renorm(1, 2, maxOutNorm)
   end
end

-- just to be safe (for optim)
function Dictionary:getParameters()
   local sp = self.dpnn_sparseParameters
   self.dpnn_sparseParameters = false
   
   local flatParameters, flatGradParameters = parent.getParameters(self)
   
   self.dpnn_sparseParameters = sp
   return flatParameters, flatGradParameters
end
   
