local DontCast, parent = torch.class("nn.DontCast", "nn.Decorator")

function DontCast:__init(module, castin, castout, moduleType)
   parent.__init(self, module)
   self.castin = castin
   self.castout = (castout == nil) and castin or castout
   self.moduleType = moduleType
   if not self.moduleType then 
      assert(torch.isTensor(module.output), "cannot extrapolate module type")
      self.moduleType = torch.typename(module.output)
   end
end

function DontCast:updateOutput(input)
   if self.castin and torch.type(input) ~= self.moduleType then
      self._input = self._input or torch.getmetatable(self.moduleType).new()
      self._input:resize(input:size()):copy(input)
      input = self._input
   end
   
   local output = self.module:updateOutput(input)
   
   if self.castout then
      self.output:resize(output:size()):copy(output)
   else
      self.output = output
   end
   return self.output
end

function DontCast:updateGradInput(input, gradOutput)
   if self.castin and torch.type(input) ~= self.moduleType then
      input = self._input
   end
   if self.castout and torch.type(gradOutput) ~= self.moduleType then
      self._gradOutput = self._gradOutput or torch.getmetatable(self.moduleType).new()
      self._gradOutput:resize(gradOutput:size()):copy(gradOutput)
      gradOutput = self._gradOutput
   end
   
   local gradInput = self.module:updateGradInput(input, gradOutput)
   
   if self.castin then
      self.gradInput:resize(gradInput:size()):copy(gradInput)
   else
      self.gradInput = gradInput
   end
   return self.gradInput
end

function DontCast:accGradParameters(input, gradOutput, scale)
   if self.castin and torch.type(input) ~= self.moduleType then
      input = self._input
   end
   if self.castout and torch.type(gradOutput) ~= self.moduleType then
      gradOutput = self._gradOutput
   end
   
   self.module:accGradParameters(input, gradOutput, scale)
end

function DontCast:accUpdateGradParameters(input, gradOutput, lr)
   if self.castin and torch.type(input) ~= self.moduleType then
      input = self._input
   end
   if self.castout and torch.type(gradOutput) ~= self.moduleType then
      gradOutput = self._gradOutput
   end
   
   self.module:accUpdateGradParameters(input, gradOutput, lr)
end

-- dont cast
function DontCast:type(type)
   if self.castout then
      self.output = self.output:type(type)
   end
   if self.castin then
      self.gradInput = self.gradInput:type(type)
   end
   return self
end
