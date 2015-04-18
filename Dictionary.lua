local Dictionary, parent = torch.class("nn.Dictionary", "nn.LookupTable")

function Dictionary:parameters()
   if self.dpnn_sparseParameters then
      -- only return the parameters affected by the forward/backward
      local params, gradParams, scales = {}, {}, {}
      for k,nBackward in pairs(self.inputs) do
         local kscale = self:scaleUpdateByKey(k)
         params[k] = self.weight:select(1, k)
         if not self._acc_update then
            gradParams[k] = self._lookup.gradWeight:select(1, k)
         end
         scales[k] = self:scaleUpdateByKey(k)
      end
      return params, gradParams, scales
   end
   return parent.parameters(self)
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
   
