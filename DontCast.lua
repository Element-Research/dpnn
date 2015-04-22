local DontCast, parent = torch.class("nn.DontCast", "nn.Container")

function DontCast:__init(module)
   self.module = module
   parent.__init(self)
   self.modules[1] = module
end

function DontCast:updateOutput(input)
   self.output = self.module:updateOutput(input)
   return self.output
end

function DontCast:updateGradInput(input, gradOutput)
   self.gradInput = self.module:updateGradInput(input, gradOutput)
   return self.gradInput
end

function DontCast:accGradParameters(input, gradOutput, scale)
   self.module:accGradParameters(input, gradOutput, scale)
end

function DontCast:accUpdateGradParameters(input, gradOutput, scale)
   self.module:accUpdateGradParameters(input, gradOutput, scale)
end

-- dont cast
function DontCast:type(type)
   return self
end
