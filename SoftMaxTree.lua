local SoftMaxTree, parent = nn.SoftMaxTree, nn.Module

function SoftMaxTree:momentumGradParameters()
   -- get dense view of momGradParams
   if not self.momGradParams or _.isEmpty(self.momGradParams) then
      assert(not self.accUpdate, "cannot use momentum with accUpdate")
      self.momGradParams = {self.gradWeight:clone():zero(), self.gradBias:clone():zero()}
   end
   local momGradParams = self.momGradParams
   if self.static and not _.isEmpty(self.updates) then      
      local momGradWeight = momGradParams[1]
      local momGradBias = momGradParams[2]
      momGradParams = {}
      -- only return the parameters affected by the forward/backward
      for parentId, scale in pairs(self.updates) do
         local node = self.parentChildren:select(1, parentId)
         local parentIdx = node[1]
         local nChildren = node[2]
         momGradParams[parentId] = momGradWeight:narrow(1, parentIdx, nChildren)
         local biasId = parentId+self.maxParentId
         momGradParams[biasId] = momGradBias:narrow(1, parentIdx, nChildren)
      end
   end
   return momGradParams
end
