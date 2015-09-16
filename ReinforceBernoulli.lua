------------------------------------------------------------------------
--[[ ReinforceBernoulli ]]-- 
-- Ref A. http://incompleteideas.net/sutton/williams-92.pdf
-- Inputs are bernoulli probabilities (p) 
-- Ouputs are samples drawn from this distribution.
-- Uses the REINFORCE algorithm (ref. A p.230-236) which is 
-- implemented through the nn.Module:reinforce(reward) interface.
-- gradOutputs are ignored (REINFORCE algorithm).
------------------------------------------------------------------------
local ReinforceBernoulli, parent = torch.class("nn.ReinforceBernoulli", "nn.Reinforce")

function ReinforceBernoulli:updateOutput(input)
   self.output:resizeAs(input)
   if self.stochastic or self.train ~= false then
      -- sample from bernoulli with P(output=1) = input
      self._uniform = self._uniform or input.new()
      self._uniform:resizeAs(input):uniform(0,1)
      self.output:lt(self._uniform, input)
   else
      -- use p for evaluation
      self.output:copy(input)
   end
   return self.output
end

function ReinforceBernoulli:updateGradInput(input, gradOutput)
   -- Note that gradOutput is ignored
   -- f : bernoulli probability mass function
   -- x : the sampled values (0 or 1) (self.output)
   -- p : probability of sampling a 1
   -- derivative of log bernoulli w.r.t. p
   -- d ln(f(x,p))    (x - p)
   -- ------------ = ---------
   --     d p         p(1 - p)
   self.gradInput:resizeAs(input)
   -- (x - p)
   self.gradInput:copy(self.output):add(-1, input)
   -- divide by p(1 - p)
   self._div = self._div or input.new()
   self._div:resizeAs(input)
   self._div:fill(1):add(-1, input):cmul(input)
   self.gradInput:cdiv(self._div)
   
   -- multiply by reward 
   self.gradInput:cmul(self:rewardAs(input))
   -- multiply by -1 ( gradient descent on input )
   self.gradInput:mul(-1)
   return self.gradInput
end


