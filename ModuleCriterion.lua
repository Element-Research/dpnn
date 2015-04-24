local ModuleCriterion, parent = torch.class("nn.ModuleCriterion", "nn.Criterion")

function ModuleCriterion:__init(criterion, inputModule, targetModule)
   self.inputModule = inputModule
   self.targetModule = targetModule
   self.castTarget = true
   if self.inputModule then
      local params = self.inputModule:parameters()
      if params and #params > 0 then
         print"Warning: nn.ModuleCriterion doesn't support parameter updates"
      end
   end
   self.criterion = criterion
end

function ModuleCriterion:forward(input, target)
   if self.inputModule then
      self.input = self.inputModule:forward(input)
   end
   if self.targetModule then
      self.target = self.outputModule:forward(target)
   end
   self.output = self.criterion:forward(self.input or input, self.target or target)
   return self.output
end

function ModuleCriterion:backward(input, target)
   self.gradInput = self.criterion:backward(self.input or input, self.target or target)
   if self.inputModule then
      self.gradInput = self.inputModule:backward(input, self.gradInput)
   end
   return self.gradInput
end

function ModuleCriterion:type(type)
   if self.inputModule then
      self.inputModule:type(type)
   end
   if self.castTarget and self.targetModule then
      self.targetModule:type(type)
   end
   return parent.type(self, type)
end
