------------------------------------------------------------------------
--[[ BinaryClassReward ]]--
-- Variance reduced binary classification reinforcement criterion.
-- The binary class version of VRClassReward.
-- input : {class prediction, baseline reward}
-- Reward is 1 for success, Reward is 0 otherwise.
-- reward = scale*(Reward - baseline) where baseline is 2nd input element
-- Note : for RNNs with R = 1 for last step in sequence, encapsulate it
-- in nn.ModuleCriterion(BinaryClassReward, nn.SelectTable(-1))
------------------------------------------------------------------------
local BinaryClassReward, parent = torch.class("nn.BinaryClassReward", "nn.Criterion")

function BinaryClassReward:__init(module, scale, criterion)
   parent.__init(self)
   self.module = module -- so it can call module:reinforce(reward)
   self.scale = scale or 1 -- scale of reward
   self.criterion = criterion or nn.MSECriterion() -- baseline criterion
   self.sizeAverage = true
   self.gradInput = {torch.Tensor()}
end

function BinaryClassReward:updateOutput(input, target)
   assert(torch.type(input) == 'table')
   local input = input[1]
   assert(input:dim() == 1)
   assert(target:dim() == 1)
   self._binary = self._binary or input.new()
   self._binary:gt(input, 0.5)
   
   -- max class value is class prediction
   if torch.type(self._binary) ~= torch.type(target) then
      self._target = self._target or self._binary.new()
      self._target:resize(target:size()):copy(target)
      target = self._target
   end
   
   -- reward = scale when correctly classified
   self._reward = self._reward or input.new()
   self._reward:eq(self._binary, target)
   self.reward = self.reward or input.new()
   self.reward:resize(self._reward:size(1)):copy(self._reward)
   self.reward:mul(self.scale)
   
   -- loss = -sum(reward)
   self.output = -self.reward:sum()
   if self.sizeAverage then
      self.output = self.output/input:size(1)
   end
   return self.output
end

function BinaryClassReward:updateGradInput(inputTable, target)
   local input, baseline = unpack(inputTable)
   
   -- reduce variance of reward using baseline
   self.vrReward = self.vrReward or self.reward.new()
   self.vrReward:resizeAs(self.reward):copy(self.reward)
   self.vrReward:add(-1, baseline)
   if self.sizeAverage then
      self.vrReward:div(input:size(1))
   end
   -- broadcast reward to modules
   self.module:reinforce(self.vrReward)  
   
   -- zero gradInput (this criterion has no gradInput for class pred)
   self.gradInput[1]:resizeAs(input):zero()
   
   -- learn the baseline reward
   self.gradInput[2] = self.criterion:backward(baseline, self.reward)
   
   return self.gradInput
end

function BinaryClassReward:type(type)
   self._binary = nil
   self._target = nil
   local module = self.module
   self.module = nil
   local ret = parent.type(self, type)
   self.module = module
   return ret
end
