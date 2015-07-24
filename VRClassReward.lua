------------------------------------------------------------------------
--[[ VRClassReward ]]--
-- Variance reduced classification reinforcement criterion.
-- Reward is 1 for success, Reward is 0 otherwise.
-- reward = (Reward - baseline) where baseline is exp moving avg of Reward
-- Note : for RNNs with R = 1 for last step in sequence, encapsulate it
-- in nn.ModuleCriterion(VRClassReward, nn.SelectTable(-1))
------------------------------------------------------------------------
local VRClassReward, parent = torch.class("nn.VRClassReward", "nn.Criterion")

function VRClassReward:__init(module, basecoeff, scale)
   self.module = module -- so it can call module:reinforce(reward)
   self.basecoeff = basecoeff or 0.9 -- weight of past baseline
   self.baseline = 0
   self.scale = scale or 1 -- scale of reward
end

function VRClassReward:updateOutput(input, target)
   assert(input:dim() == 2, "only works with batches")
   self._maxVal = self._maxVal or input.new()
   self._maxIdx = self._maxIdx or torch.type(input) == 'torch.CudaTensor' and input.new() or torch.LongTensor()
   
   -- max class value is class prediction
   self._maxIdx:max(self._maxVal, input, 2)
   if torch.type(self._maxIdx) ~= torch.type(target) then
      self._target = self._target or self._maxIdx.new()
      self._target:resize(target:size()):copy(target)
      target = self._target
   end
   
   -- reward = scale when correctly classified
   self._maxIdx:eq(self._maxIdx, target)
   self.reward = self.reward or input.new()
   self.reward:resize(self._maxIdx:size()):copy(self._maxIdx)
   self.reward:mul(self.scale)
   
   -- loss = -sum(reward)
   self.output = -self.reward:sum()
end

function VRClassReward:udpateGradInput(input, target)
   -- update baseline using current and past rewards
   -- baseline is exponentially moving average of reward
   if self.baseline == 0 then
      self.baseline = self.reward:mean()
   else
      self.baseline = self.basecoeff*self.baseline + (1-self.basecoeff)*self.reward:mean()
   end
   
   -- reduce variance of reward using baseline
   self.vrReward = self.vrReward or self.reward.new()
   self.vrReward:copy(self.reward):add(-self.baseline)
   
   -- broadcast reward to modules
   self.module:reinforce(self.vrReward)
   
   -- zero gradInput (this criterion has no gradInput)
   self.gradInput:resiseAs(input):zero()
end

function VRClassReward:type(type)
   self._maxVal = nil
   self._maxIdx = nil
   self._target = nil
   local module = self.module
   self.module = nil
   local ret = parent.type(self, type)
   self.module = module
   return ret
end
