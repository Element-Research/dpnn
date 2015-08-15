------------------------------------------------------------------------
--[[ CategoricalEntropy ]]--
-- Maximize the entropy of a categorical distribution (e.g. softmax ).
-- H(X) = E(-log(p(X)) = -sum(p(X)log(p(X)) 
-- where X = 1,...,N and N is the number of categories.
-- A batch with an entropy below minEntropy will be maximized.
-- d H(X=x)     p(x)
-- -------- = - ---- - log(p(x)) = -1 - log(p(x))
--   d p        p(x)
------------------------------------------------------------------------
local CE, parent = torch.class("nn.CategoricalEntropy", "nn.Module")

function CE:__init(scale, minEntropy)
   parent.__init(self)
   self.scale = scale or 1
   self.minEntropy = minEntropy
   
   -- get the P(X) using the batch as a prior
   self.module = nn.Sequential()
   self.module:add(nn.Sum(1)) -- sum categorical probabilities over batch
   self._mul = nn.MulConstant(1)
   self.module:add(self._mul) -- make them sum to one (i.e. probabilities)
   
   -- get entropy H(X)
   local concat = nn.ConcatTable()
   concat:add(nn.Identity()) -- p(X)
   local seq = nn.Sequential()
   seq:add(nn.AddConstant(0.000001)) -- prevent log(0) = nan errors
   seq:add(nn.Log())
   concat:add(seq)
   self.module:add(concat) -- log(p(x))
   self.module:add(nn.CMulTable()) -- p(x)log(p(x))
   self.module:add(nn.Sum()) -- sum(p(x)log(p(x)))
   self.module:add(nn.MulConstant(-1)) -- H(x)
   
   self.modules = {self.module}
   
   self.minusOne = torch.Tensor{-self.scale} -- gradient descent on maximization
   self.sizeAverage = true
end

function CE:updateOutput(input)
   assert(input:dim() == 2, "CategoricalEntropy only works with batches")
   self.output:set(input)
   return self.output
end

function CE:updateGradInput(input, gradOutput, scale)
   assert(input:dim() == 2, "CategoricalEntropy only works with batches")
   self.gradInput:resizeAs(input):copy(gradOutput)
   
   self._mul.constant_scalar = 1/input:sum() -- sum to one
   self.entropy = self.module:updateOutput(input)[1]
   if (not self.minEntropy) or (self.entropy < self.minEntropy) then
      local gradEntropy = self.module:updateGradInput(input,  self.minusOne, scale)
      if self.sizeAverage then
         gradEntropy:div(input:size(1))
      end
      self.gradInput:add(gradEntropy)
   end
   
   return self.gradInput
end
