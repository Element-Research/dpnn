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
   return torch.type(self) .. ' @ ' .. self.module:__tostring__()
end

