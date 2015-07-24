local LookupTable, parent = nn.LookupTable, nn.Module

function LookupTable:maxParamNorm(maxOutNorm, maxInNorm)
   maxOutNorm = self.maxOutNorm or maxOutNorm or self.maxInNorm or maxInNorm
   if not (maxOutNorm or maxInNorm) then
      return
   end
   
   if maxOutNorm and maxOutNorm > 0 then
      -- cols feed into output neurons 
      self.weight:renorm(2, 2, maxOutNorm)
   end
   if maxInNorm and maxInNorm > 0 then
      -- rows feed out from input neurons
      self.weight:renorm(2, 1, maxInNorm)
   end
end
