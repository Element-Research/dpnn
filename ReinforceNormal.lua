------------------------------------------------------------------------
--[[ ReinforceNormal ]]-- 
-- Ref A. http://incompleteideas.net/sutton/williams-92.pdf
-- Inputs are mean (mu) of multivariate normal distribution. 
-- Ouputs are samples drawn from these distributions.
-- Standard deviation is provided as constructor argument.
-- Uses the REINFORCE algorithm (ref. A sec 6. p.237-239) which is 
-- implemented through the nn.Module:reinforce(r,b) interface.
-- gradOutputs are ignored (REINFORCE algorithm).
------------------------------------------------------------------------
local ReinforceNormal, parent = torch.class("nn.ReinforceNormal", "nn.Reinforce")

function ReinforceNormal:__init(stdev)
   parent.__init(self)
   self.stdev = stdev
end

function ReinforceNormal:updateOutput(input)
   -- TODO : input could also be a table of mean and stdev tensors
   self.output:resizeAs(input)
   if self.train ~= false then
      self.output:normal()
      
      -- multiply by standard deviations
      if torch.type(self.stdev) == 'number' then
         self.output:mul(self.stdev)
      elseif torch.isTensor(self.stdev) then
         if self.stdev:dim() == input:dim() then
            assert(self.stdev:isSameSizeAs(input))
            self.output:cmul(self.stdev)
         else
            assert(self.stdev:dim()+1 == input:dim())
            self._stdev = self._stdev or self.stdev.new()
            self._stdev:view(self.stdev,1,table.unpack(self.stdev:size():totable()))
            self.__stdev = self.__stdev or self.stdev.new()
            self.__stdev:expandAs(self._stdev, input)
            self.output:cmul(self.__stdev)
         end
      else
         error"unsupported input type"
      end
      
      -- re-center the means to the input
      self.output:add(input)
   else
      -- use maximum a posteriori (MAP) estimate
      self.output:copy(input)
   end
   return self.output
end

function ReinforceNormal:updateGradInput(input, gradOutput)
   -- Note that gradOutput is ignored
   -- f : normal probability density function
   -- x : the sampled values (self.output)
   -- u : mean (mu) (self.input)
   -- s : standard deviation (sigma) (self.stdev)
   -- derivative of log normal w.r.t. mean
   -- d ln(f(x,u,s))   (x - u)
   -- -------------- = -------
   --      d u           s^2
   self.gradInput:resizeAs(input)
   -- (x - u)
   self.gradInput:copy(self.output):add(-1, input)
   
   -- divide by squard standard deviations
   if torch.type(self.stdev) == 'number' then
      self.gradInput:div(self.stdev^2)
   else
      if self.stdev:dim() == input:dim() then
         self.gradInput:cdiv(self.stdev):cdiv(self.stdev)
      else
         self.gradInput:cdiv(self.__stdev):cdiv(self.__stdev)
      end
   end
   -- multiply by reward (default : reinforceSignal-reinforceBaseline)
   self.gradInput:cmul(self:rewardAs(input))
   return self.gradInput
end
