------------------------------------------------------------------------
--[[ Noise Contrast Estimation Module]]--
-- Ref.: A. https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf
------------------------------------------------------------------------
local NCEModule, parent = torch.class("nn.NCEModule", "nn.Linear")

-- for efficient serialization
local empty = _.clone(parent.dpnn_mediumEmpty)
table.insert(empty, 'sampleidx')
table.insert(empty, 'sampleprob')
table.insert(empty, '_noiseidx')
table.insert(empty, '_noiseprob')
NCEModule.dpnn_mediumEmpty = empty

-- for sharedClone
local params = _.clone(parent.dpnn_parameters)
table.insert(params, 'unigrams')
NCEModule.dpnn_parameters = params

function NCEModule:__init(inputSize, outputSize, k, unigrams)
   parent.__init(self, inputSize, outputSize)
   assert(torch.type(k) == 'number')
   assert(torch.isTensor(unigrams))
   self.k = k
   self.unigrams = unigrams
   
   self:fastNoise()
   
   -- output is {P_linear(target|input), P_linear(samples|input), P_noise(target), P_noise(samples)}
   self.output = {torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()}
   self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function NCEModule:fastNoise()
   -- we use alias to speedup multinomial sampling (see noiseSample method)
   require 'torchx'
   assert(torch.AliasMultinomial, "update torchx : luarocks install torchx")
   self.unigrams:div(self.unigrams:sum())
   self.aliasmultinomial = torch.AliasMultinomial(self.unigrams)
   self.aliasmultinomial.dpnn_parameters = {'J', 'q'}
end

function NCEModule:updateOutput(inputTable)
   local input, target = unpack(inputTable)
   assert(input:dim() == 2)
   assert(target:dim() == 1)
   local batchsize = input:size(1)
   local inputsize = self.weight:size(2)
   
   if self.train == false and self.normalized then
      self.linout = self.linout or input.new()
      -- full linear + softmax
      local nElement = self.linout:nElement()
      self.linout:resize(batchsize, self.weight:size(1))
      if self.linout:nElement() ~= nElement then
         self.linout:zero()
      end
      self.addBuffer = self.addBuffer or input.new()
      if self.addBuffer:nElement() ~= batchsize then
         self.addBuffer:resize(batchsize):fill(1)
      end
      self.linout:addmm(0, self.linout, 1, input, self.weight:t())
      if self.bias then self.linout:addr(1, self.addBuffer, self.bias) end
      self.output = torch.type(self.output) == 'table' and input.new() or self.output
      if self.logsoftmax then
         input.THNN.LogSoftMax_updateOutput(
            self.linout:cdata(),
            self.output:cdata()
         )
      else
         input.THNN.SoftMax_updateOutput(
            self.linout:cdata(),
            self.output:cdata()
         )
      end
   else
      self.output = (torch.type(self.output) == 'table' and #self.output == 4) and self.output
         or {input.new(), input.new(), input.new(), input.new()}
      self.sampleidx = self.sampleidx or target.new()
      
      -- the last first column will contain the target indices
      self.sampleidx:resize(batchsize, self.k+1)
      self.sampleidx:select(2,1):copy(target)
      
      self._sampleidx = self._sampleidx or self.sampleidx.new()
      self._sampleidx:resize(batchsize, self.k)
      
      -- sample (batchsize x k+1) noise samples
      self:noiseSample(self._sampleidx, batchsize, self.k)
      
      self.sampleidx:narrow(2,2,self.k):copy(self._sampleidx)
      
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
      
      self.sampleprob = self.sampleprob or self._score.new()
      self.sampleprob = self:noiseProb(self.sampleprob, self.sampleidx)
      
      local tprob = self.sampleprob:select(2,1)
      local nprob = self.sampleprob:narrow(2,2,self.k)
      
      self.output[3]:set(tprob)
      self.output[4]:set(nprob)
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
   self.gradInput[1] = self.gradInput[1] or input.new()
   self.gradInput[1]:resize(batchsize, 1, inputsize):zero()
   self.gradInput[1]:baddbmm(0, 1, self._gradOutput, self._weight)
   self.gradInput[1]:resizeAs(input)

   self.gradInput[2] = self.gradInput[2] or input.new()
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
      self._noiseidx = nil
      self._noiseprob = nil
      self._metaidx = nil
   end
   local unigrams = self.unigrams
   self.unigrams = nil
   local am = self.aliasmultinomial
   local rtn = parent.type(self, type, cache)
   self.unigrams = unigrams
   self.aliasmultinomial = am
   return rtn
end

function NCEModule:noiseProb(sampleprob, sampleidx)
   assert(sampleprob)
   assert(sampleidx)
   self._noiseprob = self._noiseprob or self.unigrams.new()
   self._noiseidx = self._noiseidx or torch.LongTensor()
   self._noiseidx:resize(sampleidx:size()):copy(sampleidx)
   
   self._noiseprob:index(self.unigrams, 1, self._noiseidx:view(-1))
   
   sampleprob:resize(sampleidx:size()):copy(self._noiseprob)
   return sampleprob
end

function NCEModule:noiseSample(sampleidx, batchsize, k)
   if torch.type(sampleidx) ~= 'torch.LongTensor' then
      self._noiseidx = self._noiseidx or torch.LongTensor()
      self._noiseidx:resize(batchsize, k)
      self.aliasmultinomial:batchdraw(self._noiseidx)
      sampleidx:resize(batchsize, k):copy(self._noiseidx)
   else
      sampleidx:resize(batchsize, k)
      self.aliasmultinomial:batchdraw(sampleidx)
   end
   return sampleidx
end

function NCEModule:clearState()
   self.sampleidx = nil
   self.sampleprob = nil
   self._noiseidx = nil
   self._noiseprob = nil
   if torch.isTensor(self.output) then
      self.output:set()
   else
      for i,output in ipairs(self.output) do
         output:set()
      end
   end
   for i,gradInput in ipairs(self.gradInput) do
      gradInput:set()
   end
end
