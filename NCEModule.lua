------------------------------------------------------------------------
--[[ Noise Contrast Estimation Module]]--
-- Ref.: A. https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf
------------------------------------------------------------------------
local NCEModule, parent = torch.class("nn.NCEModule", "nn.Linear")

function NCEModule:__init(inputSize, outputSize, k, noise)
   parent.__init(self, inputSize, outputSize)
   assert(torch.type(k) == 'number')
   if torch.isTensor(noise) then
      -- assume it is unigrams
      noise = self:unigramNoise(noise)
   end
   assert(torch.type(noise) == 'table')
   assert(torch.type(noise.sample) == 'function')
   assert(torch.type(noise.prob) == 'function')
   self.k = k
   self.noise = noise
   
   local sampleidx = self.noise:sample(nil, 1, self.k)
   assert(torch.isTensor(sampleidx))
   local sampleprob = self.noise:prob(nil, sampleidx)
   assert(torch.isTensor(sampleprob))
   
   -- output is {P_linear(target|input), P_linear(samples|input), P_noise(target), P_noise(samples)}
   self.output = {torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()}
   self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function NCEModule:updateOutput(inputTable)
   local input, target = unpack(inputTable)
   assert(input:dim() == 2)
   assert(target:dim() == 1)
   local batchsize = input:size(1)
   local inputsize = self.weight:size(2)
   
   if self.normalized then
      -- full linear + softmax
      error"Not implemented"
   elseif self.train ~= false then
      self.sampleidx = self.sampleidx or target.new()
      
      -- the last first column will contain the target indices
      self.sampleidx:resize(batchsize, self.k+1)
      self.sampleidx:select(2,1):copy(target)
      
      self._noiseidx = self._noiseidx or self.sampleidx.new()
      self._noiseidx:resize(batchsize, self.k)
      
      -- sample (batchsize x k+1) noise samples
      self.noise:sample(self._noiseidx, batchsize, self.k)
      
      self.sampleidx:narrow(2,2,self.k):copy(self._noiseidx)
      
      -- make sure that targets are still first column of sampleidx
      if not self.testedtargets then
         for i=1,math.min(target:size(1),3) do
            assert(self.sampleidx[{i,1}] == target[i])
         end
         self.testedtargets = true
      end
      
      -- build (batchsize x k+1 x inputsize) weight tensor
      self._weight = self._weight or self.weight.new()
      self._weight:index(self.weight, 1, self.sampleidx:view(-1))
      assert(self._weight:nElement() == batchsize*(self.k+1)*inputsize)
      self._weight:resize(batchsize, self.k+1, inputsize)
      
      -- build (batchsize x k+1) bias tensor
      self._bias = self._bias or self.bias.new()
      self._bias:index(self.bias, 1, self.sampleidx:view(-1))
      assert(self._bias:nElement() == batchsize*(self.k+1))
      self._bias:resize(batchsize, self.k+1)
      
      -- get score of noise and target (batchsize x k+1) samples
      self._score = self._score or input.new()
      self._score:resizeAs(self._bias):copy(self._bias)
      self._score:resize(batchsize, 1, self.k+1)
      local _input = input:view(batchsize, 1, inputsize)
      self._score:baddbmm(1, self._score, 1, _input, self._weight:transpose(2,3))
      self._score:resize(batchsize, self.k+1)
      self._score:exp()
      
      -- separate target from noise scores
      local tscore = self._score:select(2,1)
      local nscore = self._score:narrow(2,2,self.k)
      
      self.output[1]:set(tscore)
      self.output[2]:set(nscore)
      
      -- get noise probability for all samples
      
      self.sampleprob = self.noise:prob(self.sampleprob, self.sampleidx)
      
      local tprob = self.sampleprob:select(2,1)
      local nprob = self.sampleprob:narrow(2,2,self.k)
      
      self.output[3]:set(tprob)
      self.output[4]:set(nprob)
   else
      -- output : score(Y=target|X=input)
      local input, target = unpack(inputTable)
      assert(input:dim() == 2)
      assert(target:dim() == 1)
      local batchsize = input:size(1)
      
      -- build (batchsize x inputsize) weight tensor
      self._weight = self._weight or self.weight.new()
      self._weight:index(self.weight, 1, target)
      
      -- build (batchsize x 1) bias tensor
      self._bias = self._bias or self.bias.new()
      self._bias:index(self.bias, 1, target)
      
      -- compute score(Y=target|X=input) 
      self._buff = self._buff or input.new()
      self._buff:add(input, self._weight)
      self.output:sum(self._buff, 2):add(self._bias)
      self.output:exp()
   end
   
   return self.output
end

function NCEModule:updateGradInput(inputTable, gradOutput)
   local input, target = unpack(inputTable)
   assert(input:dim() == 2)
   assert(target:dim() == 1)
   local dPmt, dPms = gradOutput[1], gradOutput[2]
   local batchsize = input:size(1)
   local inputsize = self.weight:size(2)
   
   self._gradOutput = self._gradOutput or dPmt.new()
   self._gradOutput:resize(batchsize, self.k+1)
   self._gradOutput:select(2,1):copy(dPmt)
   self._gradOutput:narrow(2,2,self.k):copy(dPms)
   self._gradOutput:resize(batchsize, 1, self.k+1)
   self._gradOutput:cmul(self._score) -- gradient of exp
   
   -- gradient of linear
   self.gradInput[1]:resize(batchsize, 1, inputsize):zero()
   self.gradInput[1]:baddbmm(0, 1, self._gradOutput, self._weight)
   self.gradInput[1]:resizeAs(input)

   if self.gradInput[2]:nElement() ~= target:nElement() then
      self.gradInput[2]:resize(target:size()):zero()
   end
   
   return self.gradInput
end

function NCEModule:accGradParameters(inputTable, gradOutput, scale)
   local input, target = unpack(inputTable)
   assert(input:dim() == 2)
   assert(target:dim() == 1)
   local batchsize = input:size(1)
   local inputsize = self.weight:size(2)
   
   self._gradWeight = self._gradWeight or self.gradWeight.new()
   self._gradWeight:resizeAs(self._weight):zero() -- batchsize x k+1 x inputsize
   self._gradOutput:resize(batchsize, self.k+1, 1)
   self._gradOutput:mul(scale)
   local _input = input:view(batchsize, 1, inputsize)
   self._gradWeight:baddbmm(0, self._gradWeight, 1, self._gradOutput, _input)
   
   local sampleidx = self.sampleidx:view(batchsize * (self.k+1))
   local _gradWeight = self._gradWeight:view(batchsize * (self.k+1), inputsize)
   self.gradWeight:indexAdd(1, sampleidx, _gradWeight)
   
   local _gradOutput = self._gradOutput:view(batchsize * (self.k+1))
   self.gradBias:indexAdd(1, sampleidx, _gradOutput)
end

function NCEModule:type(type, cache)
   if type then
      self.sampleidx = nil
      self.sampleprob = nil
   end
   return parent.type(self, type, cache)
end

function NCEModule:training()
   self.output = {torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()}
   return parent.training(self)
end

function NCEModule:evaluate()
   self.output = torch.Tensor()
   return parent.evaluate(self)
end

function NCEModule:unigramNoise(unigrams)
   assert(unigrams:dim() == 1)
   local noise = {
      unigrams = unigrams,
      sample = function(self, sampleidx, batchsize, k)
         sampleidx = sampleidx or torch.LongTensor()
         self.unigrams.multinomial(sampleidx, self.unigrams, batchsize*k, true)
         sampleidx:resize(batchsize, k)
         return sampleidx
      end,
      prob = function(self, sampleprob, sampleidx)
         sampleprob = sampleprob or self.unigrams.new()
         sampleprob:index(self.unigrams, 1, sampleidx:view(-1))
         sampleprob:resize(sampleidx:size())
         return sampleprob
      end
   }
   return noise
end
