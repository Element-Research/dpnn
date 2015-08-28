------------------------------------------------------------------------
--[[ ReinforceCategorical ]]-- 
-- Ref A. http://incompleteideas.net/sutton/williams-92.pdf
-- Inputs are a vector of categorical prob : (p[1], p[2], ..., p[k]) 
-- Ouputs are samples drawn from this distribution.
-- Uses the REINFORCE algorithm (ref. A sec 6. p.230-236) which is 
-- implemented through the nn.Module:reinforce(r,b) interface.
-- gradOutputs are ignored (REINFORCE algorithm).
------------------------------------------------------------------------
local ReinforceCategorical, parent = torch.class("nn.ReinforceCategorical", "nn.Reinforce")

function ReinforceCategorical:updateOutput(input)
   self.output:resizeAs(input)
   self._index = self._index or ((torch.type(input) == 'torch.CudaTensor') and torch.CudaTensor() or torch.LongTensor())
   if self.train ~= false then
      -- sample from categorical with p = input
      input.multinomial(self._index, input, 1)
      -- one hot encoding
      self.output:zero()
      self.output:scatter(2, self._index, 1)
   else
      -- use p for evaluation
      self.output:copy(input)
   end
   return self.output
end

function ReinforceCategorical:updateGradInput(input, gradOutput)
   -- Note that gradOutput is ignored
   -- f : categorical probability mass function
   -- x : the sampled indices (one per sample) (self.output)
   -- p : probability vector (p[1], p[2], ..., p[k]) 
   -- derivative of log categorical w.r.t. p
   -- d ln(f(x,p))     1/p[i]    if i = x  
   -- ------------ =   
   --     d p          0         otherwise
   self.gradInput:resizeAs(input):zero()
   self.gradInput:copy(self.output)
   self.gradInput:cdiv(input)
   
   -- multiply by reward 
   self.gradInput:cmul(self:rewardAs(input))
   -- multiply by -1 ( gradient descent on input )
   self.gradInput:mul(-1)
   return self.gradInput
end

function ReinforceCategorical:type(type, tc)
   self._index = nil
   return parent.type(self, type, tc)
end
