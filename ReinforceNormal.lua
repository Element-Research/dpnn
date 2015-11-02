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

function ReinforceNormal:__init(stdev, stochastic)
   parent.__init(self, stochastic)
   self.stdev = stdev
   if not stdev then
      self.gradInput = {torch.Tensor(), torch.Tensor()}
   end
end

function ReinforceNormal:updateOutput(input)
   local mean, stdev = input, self.stdev
   if torch.type(input) == 'table' then
      -- input is {mean, stdev}
      assert(#input == 2)
      mean, stdev = unpack(input)
   end
   assert(stdev)
   
   self.output:resizeAs(mean)
   
   if self.stochastic or self.train ~= false then
      self.output:normal()
      -- multiply by standard deviations
      if torch.type(stdev) == 'number' then
         self.output:mul(stdev)
      elseif torch.isTensor(stdev) then
         if stdev:dim() == mean:dim() then
            assert(stdev:isSameSizeAs(mean))
            self.output:cmul(stdev)
         else
            assert(stdev:dim()+1 == mean:dim())
            self._stdev = self._stdev or stdev.new()
            self._stdev:view(stdev,1,table.unpack(stdev:size():totable()))
            self.__stdev = self.__stdev or stdev.new()
            self.__stdev:expandAs(self._stdev, mean)
            self.output:cmul(self.__stdev)
         end
      else
         error"unsupported mean type"
      end
      
      -- re-center the means to the mean
      self.output:add(mean)
   else
      -- use maximum a posteriori (MAP) estimate
      self.output:copy(mean)
   end
   return self.output
end

function ReinforceNormal:updateGradInput(input, gradOutput)
   -- Note that gradOutput is ignored
   -- f : normal probability density function
   -- x : the sampled values (self.output)
   -- u : mean (mu) (mean)
   -- s : standard deviation (sigma) (stdev)
   
   local mean, stdev = input, self.stdev
   local gradMean, gradStdev = self.gradInput, nil
   if torch.type(input) == 'table' then
      mean, stdev = unpack(input)
      gradMean, gradStdev = unpack(self.gradInput)
   end
   assert(stdev)   
    
   -- Derivative of log normal w.r.t. mean :
   -- d ln(f(x,u,s))   (x - u)
   -- -------------- = -------
   --      d u           s^2
   
   gradMean:resizeAs(mean)
   -- (x - u)
   gradMean:copy(self.output):add(-1, mean)
   
   -- divide by squared standard deviations
   if torch.type(stdev) == 'number' then
      gradMean:div(stdev^2)
   else
      if stdev:dim() == mean:dim() then
         gradMean:cdiv(stdev):cdiv(stdev)
      else
         gradMean:cdiv(self.__stdev):cdiv(self.__stdev)
      end
   end
   -- multiply by reward
   gradMean:cmul(self:rewardAs(mean) )
   -- multiply by -1 ( gradient descent on mean )
   gradMean:mul(-1)
   
   -- Derivative of log normal w.r.t. stdev :
   -- d ln(f(x,u,s))   (x - u)^2 - s^2
   -- -------------- = ---------------
   --      d s              s^3
   
   if gradStdev then
      gradStdev:resizeAs(stdev)
      -- (x - u)^2
      gradStdev:copy(self.output):add(-1, mean):pow(2)
      -- subtract s^2
      self._stdev2 = self._stdev2 or stdev.new()
      self._stdev2:resizeAs(stdev):copy(stdev):cmul(stdev)
      gradStdev:add(-1, self._stdev2)
      -- divide by s^3
      self._stdev2:cmul(stdev):add(0.00000001)
      gradStdev:cdiv(self._stdev2)
      -- multiply by reward
      gradStdev:cmul(self:rewardAs(stdev))
       -- multiply by -1 ( gradient descent on stdev )
      gradStdev:mul(-1)
   end
   
   return self.gradInput
end
