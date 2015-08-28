------------------------------------------------------------------------
--[[ TotalDropout ]]--
-- Like vanilla Dropout, but on the entire inputs.
-- So either the input is entirely forwarded or entirely zeroed.
------------------------------------------------------------------------
local TotalDropout, parent = torch.class("nn.TotalDropout", "nn.Module")

function TotalDropout:__init(p)
   self.p = p or 0.5
   self.train = true
   if self.p >= 1 or self.p < 0 then
      error('<TotalDropout> illegal percentage, must be 0 <= p < 1')
   end
   parent.__init(self)
end

function TotalDropout:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if self.train then
      self.noise = torch.bernoulli(1-self.p)
      self.output:mul(self.noise)
   end
   return self.output
end

function TotalDropout:updateGradInput(input, gradOutput)
   if self.train then
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      self.gradInput:mul(self.noise) -- simply mask the gradients with the noise vector
   else
      error('backprop only defined while training')
   end
   return self.gradInput
end

function TotalDropout:__tostring__()
  return string.format('%s(%f)', torch.type(self), self.p)
end
