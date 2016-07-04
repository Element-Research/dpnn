------------------------------------------------------------------------
--[[ Noise Contrast Estimation Module]]--
-- Ref.: A. https://www.cs.toronto.edu/~amnih/papers/ncelm.pdf
------------------------------------------------------------------------
local _ = require 'moses'
local NCEModule, parent = torch.class("nn.NCEModule", "nn.Linear")
NCEModule.version = 6 -- better bias init

-- for efficient serialization using nn.Serial
local empty = _.clone(parent.dpnn_mediumEmpty)
table.insert(empty, 'sampleidx')
table.insert(empty, 'sampleprob')
table.insert(empty, '_noiseidx')
table.insert(empty, '_noiseprob')
table.insert(empty, '_weight')
table.insert(empty, '_gradWeight')
table.insert(empty, '_gradOutput')
table.insert(empty, '_tgradOutput')
NCEModule.dpnn_mediumEmpty = empty

-- for sharedClone
local params = _.clone(parent.dpnn_parameters)
table.insert(params, 'unigrams')
table.insert(params, 'Z')
NCEModule.dpnn_parameters = params

function NCEModule:__init(inputSize, outputSize, k, unigrams, Z)
   parent.__init(self, inputSize, outputSize)
   assert(torch.type(k) == 'number')
   assert(torch.isTensor(unigrams))
   self.k = k
   self.unigrams = unigrams
   self.Z = torch.Tensor{Z or -1}
   
   self.batchnoise = true
   
   self:fastNoise()
   
   -- output is {P_linear(target|input), P_linear(samples|input), P_noise(target), P_noise(samples)}
   self.output = {torch.Tensor(), torch.Tensor(), torch.Tensor(), torch.Tensor()}
   self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function NCEModule:reset(stdv)
   if stdv then
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   else
      stdv = stdv or 1./math.sqrt(self.weight:size(2))
      self.weight:uniform(-stdv, stdv)
      -- this is useful for Z = 1
      self.bias:fill(-math.log(self.bias:size(1)))
   end
   return self
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
      self.weight.addmm(self.linout, 0, self.linout, 1, input, self.weight:t())
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
   elseif self.batchnoise then
      self.output = (torch.type(self.output) == 'table' and #self.output == 4) and self.output
         or {input.new(), input.new(), input.new(), input.new()}
      assert(torch.type(target) == 'torch.CudaTensor' or torch.type(target) == 'torch.LongTensor')
      self.sampleidx = self.sampleidx or target.new()
      
      -- the last elements contain the target indices
      self.sampleidx:resize(self.k + batchsize)
      self.sampleidx:narrow(1,self.k+1,batchsize):copy(target)
      
      -- sample k noise samples
      self:noiseSample(self.sampleidx, 1, self.k)
      self.sampleidx:resize(self.k + batchsize)
      
      -- build (batchsize+k, inputsize) weight tensor
      self._weight = self._weight or self.bias.new()
      self.weight.index(self._weight, self.weight, 1, self.sampleidx)
      assert(self._weight:nElement() == (self.k+batchsize)*inputsize)
      self._weight:resize(self.k+batchsize, inputsize)
      
      -- build (batchsize+k,) bias tensor
      self._bias = self._bias or self.bias.new()
      self._bias:index(self.bias, 1, self.sampleidx)
      assert(self._bias:nElement() == (self.k+batchsize))
      self._bias:resize(self.k+batchsize)
      
      -- separate sample and target weight matrices and bias vectors
      local sweight = self._weight:narrow(1, 1, self.k)
      local tweight = self._weight:narrow(1, self.k+1, batchsize)
      local sbias = self._bias:narrow(1, 1, self.k)
      local tbias = self._bias:narrow(1, self.k+1, batchsize)
      
      -- get model probability of targets (batchsize,)
      local Pmt = self.output[1]
      self._pm = self._pm or input.new()
      self._pm:cmul(input, tweight)
      Pmt:sum(self._pm, 2):resize(batchsize)
      Pmt:add(tbias)
      Pmt:exp()
      
      -- get model probability of samples (batchsize x k) samples
      local Pms = self.output[2]
      Pms:resize(batchsize, self.k)
      Pms:copy(sbias:view(1,self.k):expand(batchsize, self.k))
      Pms:addmm(1, Pms, 1, input, sweight:t())
      Pms:exp()
      
      if self.Z[1] <= 0 then
         -- approximate Z using current batch
         self.Z[1] = Pms:mean()*self.weight:size(1)
         print("normalization constant Z approximated to "..self.Z[1])
      end
      
      -- divide by normalization constant
      Pms:div(self.Z[1]) 
      Pmt:div(self.Z[1])
      
      -- get noise probability (pn) for all samples
      
      self.sampleprob = self.sampleprob or Pms.new()
      self.sampleprob = self:noiseProb(self.sampleprob, self.sampleidx)
      
      local Pnt = self.sampleprob:narrow(1,self.k+1,target:size(1))
      local Pns = self.sampleprob:narrow(1,1,self.k)
      Pns = Pns:resize(1, self.k):expand(batchsize, self.k)
      
      self.output[3]:set(Pnt)
      self.output[4]:set(Pns)
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
      self._weight = self._weight or self.bias.new()
      self.weight.index(self._weight, self.weight, 1, self.sampleidx:view(-1))
      assert(self._weight:nElement() == batchsize*(self.k+1)*inputsize)
      self._weight:resize(batchsize, self.k+1, inputsize)
      
      -- build (batchsize x k+1) bias tensor
      self._bias = self._bias or self.bias.new()
      self._bias:index(self.bias, 1, self.sampleidx:view(-1))
      assert(self._bias:nElement() == batchsize*(self.k+1))
      self._bias:resize(batchsize, self.k+1)
      
      -- get model probability (pm) of sample and target (batchsize x k+1) samples
      self._pm = self._pm or input.new()
      self._pm:resizeAs(self._bias):copy(self._bias)
      self._pm:resize(batchsize, 1, self.k+1)
      local _input = input:view(batchsize, 1, inputsize)
      self._pm:baddbmm(1, self._pm, 1, _input, self._weight:transpose(2,3))
      self._pm:resize(batchsize, self.k+1)
      self._pm:exp()
      
      if self.Z[1] <= 0 then
         -- approximate Z using current batch
         self.Z[1] = self._pm:mean()*self.weight:size(1)
         print("normalization constant Z approximated to "..self.Z[1])
      end
      
      self._pm:div(self.Z[1]) -- divide by normalization constant
      
      -- separate target from sample model probabilities
      local Pmt = self._pm:select(2,1)
      local Pms = self._pm:narrow(2,2,self.k)
      
      self.output[1]:set(Pmt)
      self.output[2]:set(Pms)
      
      -- get noise probability (pn) for all samples
      
      self.sampleprob = self.sampleprob or self._pm.new()
      self.sampleprob = self:noiseProb(self.sampleprob, self.sampleidx)
      
      local Pnt = self.sampleprob:select(2,1)
      local Pns = self.sampleprob:narrow(2,2,self.k)
      
      self.output[3]:set(Pnt)
      self.output[4]:set(Pns)
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
   
   if self.batchnoise then
      local Pmt, Pms = self.output[1], self.output[2]
      
      -- separate sample and target weight matrices
      local sweight = self._weight:narrow(1, 1, self.k)
      local tweight = self._weight:narrow(1, self.k+1, batchsize)
      
      -- the rest of equation 7
      -- d Pm / d linear = exp(linear)/z
      self._gradOutput = self._gradOutput or dPms.new()
      self._tgradOutput = self._tgradOutput or dPmt.new()
      self._gradOutput:cmul(dPms, Pms)
      self._tgradOutput:cmul(dPmt, Pmt)
      
      -- gradient of linear
      self.gradInput[1] = self.gradInput[1] or input.new()
      self.gradInput[1]:cmul(self._tgradOutput:view(batchsize, 1):expandAs(tweight), tweight)
      self.gradInput[1]:addmm(1, 1, self._gradOutput, sweight)
   else
      -- the rest of equation 7 (combine both sides of + sign into one tensor)
      self._gradOutput = self._gradOutput or dPmt.new()
      self._gradOutput:resize(batchsize, self.k+1)
      self._gradOutput:select(2,1):copy(dPmt)
      self._gradOutput:narrow(2,2,self.k):copy(dPms)
      self._gradOutput:resize(batchsize, 1, self.k+1)
      -- d Pm / d linear = exp(linear)/z
      self._gradOutput:cmul(self._pm)
      
      -- gradient of linear
      self.gradInput[1] = self.gradInput[1] or input.new()
      self.gradInput[1]:resize(batchsize, 1, inputsize):zero()
      self.gradInput[1]:baddbmm(0, 1, self._gradOutput, self._weight)
      self.gradInput[1]:resizeAs(input)
   end
   
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
   
   if self.batchnoise then
      self._gradWeight = self._gradWeight or self.bias.new()
      self._gradWeight:resizeAs(self._weight):zero() -- (batchsize + k) x inputsize
      
      local sgradWeight = self._gradWeight:narrow(1, 1, self.k)
      local tgradWeight = self._gradWeight:narrow(1, self.k+1, batchsize)
      
      self._gradOutput:mul(scale)
      self._tgradOutput:mul(scale)
      
      sgradWeight:addmm(0, sgradWeight, 1, self._gradOutput:t(), input)
      tgradWeight:cmul(self._tgradOutput:view(batchsize, 1):expandAs(self.gradInput[1]), input)
      
      self.gradWeight:indexAdd(1, self.sampleidx, self._gradWeight)
      self.gradBias:indexAdd(1, self.sampleidx:narrow(1,self.k+1,batchsize), self._tgradOutput)
      self._tgradOutput:sum(self._gradOutput, 1) -- reuse buffer
      self.gradBias:indexAdd(1, self.sampleidx:sub(1,self.k), self._tgradOutput:view(-1))
      
   else
      self._gradWeight = self._gradWeight or self.bias.new()
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
end

function NCEModule:type(type, cache)
   if type then
      self.sampleidx = nil
      self.sampleprob = nil
      self._noiseidx = nil
      self._noiseprob = nil
      self._metaidx = nil
      self._gradOutput = nil
      self._tgradOutput = nil
      self._gradWeight = nil
      self._weight = nil
   end
   local unigrams = self.unigrams
   self.unigrams = nil
   local am = self.aliasmultinomial
   
   local rtn
   if type and torch.type(self.weight) == 'torch.MultiCudaTensor' then
      assert(type == 'torch.CudaTensor', "Cannot convert a multicuda NCEModule to anything other than cuda")
      local weight = self.weight
      local gradWeight = self.gradWeight
      self.weight = nil
      self.gradWeight = nil
      
      rtn = parent.type(self, type, cache)
      
      assert(torch.type(self.aliasmultinomial.J) ~= 'torch.CudaTensor')
      self.weight = weight
      self.gradWeight = gradWeight
   else
      rtn = parent.type(self, type, cache)
   end
   
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
   self._tgradOutput = nil
   self._gradOutput = nil
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

function NCEModule:multicuda(device1, device2)
   assert(device1 and device2, "specify two devices as arguments")
   require 'torchx'
   assert(torchx.version and torchx.version >= 1, "update torchx: luarocks install torchx")
   
   self:float()
   
   local isize = self.weight:size(2)
   local weights = {
      cutorch.withDevice(device1, function() return self.weight[{{}, {1, torch.round(isize/2)}}]:cuda() end),
      cutorch.withDevice(device2, function() return self.weight[{{}, {torch.round(isize/2)+1, isize}}]:cuda() end)
   }
   self.weight = torch.MultiCudaTensor(2, weights)
   local gradWeights = {
      cutorch.withDevice(device1, function() return self.gradWeight[{{}, {1, torch.round(isize/2)}}]:cuda() end),
      cutorch.withDevice(device2, function() return self.gradWeight[{{}, {torch.round(isize/2)+1, isize}}]:cuda() end)
   }
   self.gradWeight = torch.MultiCudaTensor(2, gradWeights)
   
   self:cuda()
end
