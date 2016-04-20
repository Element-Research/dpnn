------------------------------------------------------------------------
--[[ ReinforceGamma ]]-- 
-- Ref A. http://incompleteideas.net/sutton/williams-92.pdf
-- Inputs are shape (k) and scale (theta) of multivariate Gamma distribution. 
-- Ouputs are samples drawn from these distributions.
-- Scale is provided as constructor argument.
-- Uses the REINFORCE algorithm (ref. A sec 6. p.237-239) which is 
-- implemented through the nn.Module:reinforce(r,b) interface.
-- gradOutputs are ignored (REINFORCE algorithm).
------------------------------------------------------------------------


local ReinforceGamma, parent = torch.class("nn.ReinforceGamma", "nn.Reinforce")

function ReinforceGamma:__init(scale, stochastic)
   require('randomkit') -- needed to sample gamma dist : luarocks install randomkit
   require('cephes') -- needed to compute digamma for gradient : 
   parent.__init(self, stochastic)
   self.scale = scale
   if not scale then
      self.gradInput = {torch.Tensor(), torch.Tensor()}
   end
end

function ReinforceGamma:updateOutput(input)
   local shape, scale = input, self.scale
   if torch.type(input) == 'table' then
      -- input is {shape, scale}
      assert(#input == 2)
      shape, scale = unpack(input)
   end
   assert(scale)
   
   self.output:resizeAs(shape)

   if torch.type(scale) == 'number' then
     scale = shape.new():resizeAs(shape):fill(scale)
   elseif torch.isTensor(scale) then
      if scale:dim() == shape:dim() then
         assert(scale:isSameSizeAs(shape))
      else
         assert(scale:dim()+1 == shape:dim())
         self._scale = self._scale or scale.new()
         self._scale:view(scale,1,table.unpack(scale:size():totable()))
         self.__scale = self.__scale or scale.new()
         self.__scale:expandAs(self._scale, shape)
         scale = self.__scale
      end
   else
      error"unsupported shape type"
   end

   if self.stochastic or self.train ~= false then
      self.output:copy(randomkit.gamma(shape:squeeze():float(),scale:squeeze():float()))
   else
      -- use maximum a posteriori (MAP) estimate
      self.output:copy(shape):cmul(scale)
   end

   return self.output
end

function ReinforceGamma:updateGradInput(input, gradOutput)
   -- Note that gradOutput is ignored
   -- f : Gamma probability density function
   -- g : Digamma probability density function
   -- x : the sampled values (self.output)
   -- shape : shape parameter of gamma dist
   -- scale: scale parameter of gamma dist

   local shape, scale = input, self.scale
   local gradShape, gradScale = self.gradInput, nil
   if torch.type(input) == 'table' then
      shape, scale = unpack(input)
      gradShape, gradScale = unpack(self.gradInput)
   end
   assert(scale)
    
   -- Derivative of log gamma w.r.t. shape :
   -- d ln(f(x,shape,scale))
   -- ---------------------- = ln(x) - g(shape) - ln(scale)
   --         d shape
   gradShape:resizeAs(shape)

   if torch.type(scale) == 'number' then
      scale = shape.new():resizeAs(shape):fill(scale)
   else
      if not scale:dim() == shape:dim() then
         scale:copy(self.__scale)
      end
   end
   gradShape:copy(cephes.digamma(shape:float()))
   gradShape:mul(-1)

   self._logOutput = self._logOutput or self.output.new()
   self._logOutput:log( self.output )
   
   self._logScale = self._logScale or scale.new()
   self._logScale:log( scale )

   gradShape:add( self._logOutput )
   gradShape:add(-1, self._logScale )

   -- multiply by variance reduced reward
   gradShape:cmul(self:rewardAs(shape) )
   -- multiply by -1 ( gradient descent on shape )
   gradShape:mul(-1)
   
   -- Derivative of log Gamma w.r.t. scale :
   -- d ln(f(x,shape,scale))      x      shape
   -- ---------------------- = ------- - -----
   --         d scale          scale^2   scale
   
   if gradScale then
      gradScale:resizeAs(scale)
      gradScale:copy( torch.cdiv(self.output, torch.pow(scale,2)) )
      gradScale:add(-1, torch.cdiv(shape, scale) )
      gradScale:cmul( self:rewardAs(scale) )
      gradScale:mul(-1)
   end

   return self.gradInput
end

function ReinforceGamma:type(type,cache)
   self._logOutput = nil
   self._logScale = nil
   return parent.type(self,type,cache)
end
