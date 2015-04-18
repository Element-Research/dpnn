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

function Dictionary:updateParameters(learningRate)
   -- sparse params can have different learningRate scales per param
   local params, gradParams, scales = self:parameters()
   if params then
      for i,param in pairs(params) do -- pairs for sparse params
         local scale = scales and scales[i] or 1
         param:add(-learningRate*scale, gradParams[i])
      end
   end
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
      return params, gradParams, scales
   end
   return parent.parameters(self)
end

function Dictionary:momentumGradParameters()
   if self.dpnn_sparseParameters then
      local momGradWeight = parent.momentumGradParameters(self)
      if not momGradWeight then
         return
      end
      momGradWeight = momGradWeight[1]
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
   
