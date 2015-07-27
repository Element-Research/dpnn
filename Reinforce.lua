local Reinforce, parent = torch.class("nn.Reinforce", "nn.Module")

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
   if input:isSameSizeAs(self.reward) then
      return self.reward
   else
      if self.reward:size(1) ~= input:size(1) then
         -- assume input is in online-mode
         input = self:toBatch(input, input:dim())
         assert(self.reward:size(1) == input:size(1), self.reward:size(1).." ~= "..input:size(1))
      end
      self._reward = self._reward or self.reward.new()
      self.__reward = self.__reward or self.reward.new()
      local size = input:size():fill(1):totable()
      size[1] = self.reward:size(1)
      self._reward:view(self.reward, table.unpack(size))
      self.__reward:expandAs(self._reward, input)
      return self.__reward
   end
end
