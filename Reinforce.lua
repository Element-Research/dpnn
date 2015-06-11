local Reinforce, parent = torch.class("nn.AbstractReinforce", "nn.Module")

-- a ReinforceCriterion will call this
function Reinforce:reinforce(reward)
   parent.reinforce(self, reward)
   self.reward = reward
end

function Reinforce:updateOutput(input)
   self.output:set(input)
end

function Reinforce:updateGradInput(input, gradOutput)
   local reward = self:rewardAs(input)
   self.gradInput:resizeAs(reward):copy(reward)
end

-- this can be called by updateGradInput
function Reinforce:rewardAs(input)
   assert(self.reward:dim() == 1)
   if input:isSameSizeAs(reward) then
      return self.reward
   else
      assert(self.reward:size(1) == input:size(1))
      self._reward = self._reward or self.reward.new()
      self.__reward = self.__reward or self.reward.new()
      local size = input:size():fill(1):totable()
      table.remove(size, 1)
      self._reward:view(reward, self.reward:size(1), table.unpack(size))
      self.__reward:expandAs(self._reward, self.input)
      return self.__reward
   end
end
