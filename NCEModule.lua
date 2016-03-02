------------------------------------------------------------------------
--[[ Noise Contrast Estimation Module]]--
-- Ref.: A. https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf
------------------------------------------------------------------------
local NCEModule, parent = torch.class("nn.NCEModule", "nn.Linear")

function NCEModule:__init(inputSize, outputSize, k, noise)
   parent.__init(self, inputSize, outputSize)
   assert(torch.type(k) == 'number')
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
end

function NCEModule:updateOutput()
   local input, target = unpack(inputTable)
   assert(input:dim() == 2)
   assert(target:dim() == 1)
   local batchsize = input:size(1)
   
   if self.train ~= false then
      self.sampleidx = self.sampleidx or target.new()
      
      -- the last first column will contain the target indices
      self.sampleidx:resize(batchsize, k+1)
      self.sampleidx:select(2,1):copy(target)
      
      -- sample (batchsize x k+1) noise samples
      self.noise:sample(self.sampleidx:narrow(2,2,k), batchsize, self.k)
      
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
      self._weight:resize(self.batchsize, -1, self._weight:size(2))
      assert(self._weight:size(2) == self.k+1)
      
      -- build (batchsize x k+1) bias tensor
      self._bias = self._bias or self.bias.new()
      self._bias:index(self.bias, 1, self.sampleidx:view(-1))
      self._bias:resize(self.batchsize, -1)
      assert(self._bias:size(2) == self.k+1)
      
      -- get score of noise and target (batchsize x k+1) samples
      self._score = self._score or input.new()
      self._score:resizeAs(self._bias):copy(self._bias)
      self._score:resize(batchSize, 1, self.k)
      local _input = input:view(batchSize, 1, inputSize)
      self._score:baddbmm(1, self._score, 1, _input, self._weight:transpose(2,3))
      self._score:resize(batchsize, self.k+1)
      
      -- separate target from noise scores
      local tscore = self._score:select(2,1)
      local nscore = self._score:narrow(2,2,k)
      
      self.output[1]:resizeAs(tscore):copy(tscore):exp()
      self.output[2]:resizeAs(nscore):copy(nscore):exp()
      
      -- get noise probability for all samples
      
      self.sampleprob = self.noise:prob(self.sampleprob, self.sampleidx)
      
      local tprob = self.sampleprob:select(2,1)
      local nprob = self.sampleprob:narrow(2,2,k)
      
      self.output[3]:set(tprob)
      self.output[4]:set(nprob)
   else
      if self.normalized then
         -- call the full softmax
         error"Not implemented"
      end
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
   end
   
   return self.output
end

function NCEModule:updateGradInput(inputTable, gradOutput)

end

function NCEModule:accGradParameters(inputTable, gradOutput, scale)

end

function NCEModule:type(type, cache)
   if type then
      self.sampleidx = nil
      self.sampleprob = nil
      self.output = {torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()}
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
