------------------------------------------------------------------------
--[[ NaN ]]--
-- Asserts that outputs and gradInputs do not contain NaNs.
-- Useful for locating the source of NaN errors.
------------------------------------------------------------------------
local NaN, parent = torch.class("nn.NaN", "nn.Decorator")

local idseq = 0
function NaN.newId()
   idseq = idseq + 1
   return idseq
end

function NaN:__init(module, id)
   parent.__init(self, module)
   self.id = id or NaN.newId()
end

function NaN:recursiveIsNaN(tensor)
   local isNaN = false
   if torch.type(tensor) == 'table' then
      for k,v in pairs(tensor) do
         isNaN = self:recursiveIsNaN(v)
         if isNaN then break end
      end
   else
      local _ = require 'moses'
      isNaN = _.isNaN(tensor:sum())
   end
   return isNaN
end

function NaN:updateOutput(input)
   self.output = self.module:updateOutput(input)
   if self:recursiveIsNaN(self.output) then
      if self:recursiveIsNaN(input) then
         error(string.format("NaN found in input of module :\n%s", self:__tostring__()))
      elseif self:recursiveIsNaN(self:parameters()) then
         error(string.format("NaN found in parameters of module :\n%s", self:__tostring__()))
      end
      error(string.format("NaN found in output of module :\n%s", self:__tostring__()))
   end
   return self.output
end

function NaN:updateGradInput(input, gradOutput)
   self.gradInput = self.module:updateGradInput(input, gradOutput)
   if self:recursiveIsNaN(self.gradInput) then
      if self:recursiveIsNaN(gradOutput) then
         error(string.format("NaN found in gradOutput of module :\n%s", self:__tostring__()))
      end
      error(string.format("NaN found in gradInput of module :\n%s", self:__tostring__()))
   end
   return self.gradInput
end

function NaN:accGradParameters(input, gradOutput, scale) 
   self.module:accGradParameters(input, gradOutput, scale)
   local params, gradParams = self:parameters()
   if self:recursiveIsNaN(gradParams) then
      error(string.format("NaN found in gradParameters of module :\n%s", self:__tostring__()))
   end
end

function NaN:__tostring__()
   local selfstring = torch.type(self) .. '(' .. self.id .. ')'
   if self.module.__tostring__ then
      return selfstring .. ' @ ' .. self.module:__tostring__()
   else
      return selfstring .. ' @ ' .. torch.type(self.module)
   end
end
