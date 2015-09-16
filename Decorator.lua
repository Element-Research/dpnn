local Decorator, parent = torch.class("nn.Decorator", "nn.Container")

function Decorator:__init(module)
   parent.__init(self)
   self.module = module
   -- so that it can be handled like a Container
   self.modules[1] = module
end

function Decorator:updateOutput(input)
   self.output = self.module:updateOutput(input)
   return self.output
end

function Decorator:updateGradInput(input, gradOutput)
   self.gradInput = self.module:updateGradInput(input, gradOutput)
   return self.gradInput
end

function Decorator:accGradParameters(input, gradOutput, scale) 
   self.module:accGradParameters(input, gradOutput, scale)
end

function Decorator:accUpdateGradParameters(input, gradOutput, lr)
   self.module:accUpdateGradParameters(input, gradOutput, lr)
end

function Decorator:sharedAccUpdateGradParameters(input, gradOutput, lr)
   self.module:sharedAccUpdateGradParameters(input, gradOutput, lr)
end

function Decorator:__tostring__()
   if self.module.__tostring__ then
      return torch.type(self) .. ' @ ' .. self.module:__tostring__()
   else
      return torch.type(self) .. ' @ ' .. torch.type(self.module)
   end
end

-- useful for multiple-inheritance
function Decorator.decorate(class)
   class.updateOutput = nn.Decorator.updateOutput
   class.updateGradInput = nn.Decorator.updateGradInput
   class.accGradParameters = nn.Decorator.accGradParameters
   class.accUpdateGradParameters = nn.Decorator.accUpdateGradParameters
   class.sharedAccUpdateGradParameters = nn.Decorator.sharedAccUpdateGradParameters
   class.__tostring__ =  nn.Decorator.__tostring__
end
