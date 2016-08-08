------------------------------------------------------------------------
--[[ Linear Sparse Reinforce Categorical]]--
-- Input is {inputvalues, indices} where
-- inputvalues is batchsize x inputsize and
-- indices is batchsize x nindices
-- Output is a batchsize x nindices tensor of sampled indices
-- each output[i] is sampled from indices is indices[i]
-- This module uses the REINFORCE learning rule.
-- The linear is sparse in that each row need only compute 
-- nindices dot products.
------------------------------------------------------------------------
local _ = require 'moses'
local LSRC, parent = torch.class("nn.LSRC", "nn.Linear")
LSRC.version = 1

for i,method in ipairs{'reinforce', 'rewardAs'} do
   LSRC[method] = nn.Reinforce[method]
end

function LSRC:__init(inputsize, outputsize, epsilon)
   parent.__init(self, inputsize, outputsize)

   self.gradInput = {torch.Tensor(), torch.LongTensor()}
   self.output = torch.LongTensor()
   self.epsilon = epsilon or 0.1
end

function LSRC:reset(stdv)
   if stdv then
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   else
      stdv = stdv or 1./math.sqrt(self.weight:size(2))
      self.weight:uniform(-stdv, stdv)
      self.bias:fill(-math.log(self.bias:size(1)))
   end
   return self
end

function LSRC:updateOutput(inputTable)
   local input, indices = unpack(inputTable)
   assert(input:dim() == 2)
   assert(indices:dim() == 2)
   
   local batchsize = input:size(1)
   local inputsize = self.weight:size(2)
   local nindex = indices:size(2)
   
   assert(input:size(2) == inputsize)
   assert(torch.type(indices) == torch.type(self.output), 'Expecting input[2] to be '..torch.type(self.output))

   -- build (batchsize x nindex x inputsize) weight tensor
   self._weight = self._weight or self.bias.new()
   self.weight.index(self._weight, self.weight, 1, indices:view(-1))
   assert(self._weight:nElement() == batchsize*nindex*inputsize)
   self._weight:resize(batchsize, nindex, inputsize)
   
   -- build (batchsize x nindex) bias tensor
   self._bias = self._bias or self.bias.new()
   self._bias:index(self.bias, 1, indices:view(-1))
   assert(self._bias:nElement() == batchsize*nindex)
   self._bias:resize(batchsize, nindex)
   
   -- get output of linear with sparse outputs
   self._linearoutput = self._linearoutput or input.new()
   self._linearoutput:resizeAs(self._bias):copy(self._bias)
   self._linearoutput:resize(batchsize, 1, nindex)
   local _input = input:view(batchsize, 1, inputsize)
   self._linearoutput:baddbmm(1, self._linearoutput, 1, _input, self._weight:transpose(2,3))
   self._linearoutput:resize(batchsize, nindex)
   
   -- normalize using softmax
   self._softmaxoutput = self._softmaxoutput or input.new()
   input.THNN.SoftMax_updateOutput(self._linearoutput:cdata(),self._softmaxoutput:cdata())
   
   -- sample from categorical distribution
   self._index = self._index or ((torch.type(input) == 'torch.CudaTensor') and torch.CudaTensor() or torch.LongTensor())
   -- prevent division by zero error
   self._softmaxoutput:add(0.00000001) 
   if self.train~=false and self.epsilon > 0 then
      -- epsilon-greedy allows for more exploration
      self._softmaxoutput:mul(1-self.epsilon)
      self._softmaxoutput:add(self.epsilon/nindex)
      input.multinomial(self._index, self._softmaxoutput, 1)
   else
      input.multinomial(self._index, self._softmaxoutput, 1)
   end
   
   assert(self._index:dim() == 2)
   self.output:resizeAs(self._index)
   self.output:gather(indices, 2, self._index)
   assert(self.output:nElement() == batchsize)
   self.output:resize(batchsize)

   return self.output
end

function LSRC:updateGradInput(inputTable, gradOutput)
   local input, indices = unpack(inputTable)
   assert(input:dim() == 2)
   assert(indices:dim() == 2)
   
   local batchsize = input:size(1)
   local inputsize = self.weight:size(2)
   local nindex = indices:size(2)
   
   assert(input:size(2) == inputsize)
   assert(torch.type(indices) == torch.type(self.output), 'Expecting input[2] to be '..torch.type(self.output))
   
   -- Note that gradOutput is ignored
   -- f : categorical probability mass function
   -- x : the sampled indices (one per sample) (self.output)
   -- p : probability vector (p[1], p[2], ..., p[k]) 
   -- derivative of log categorical w.r.t. p
   -- d ln(f(x,p))     1/p[i]    if i = x  
   -- ------------ =   
   --     d p          0         otherwise
   self._gradReinforce = self._gradReinforce or input.new()
   self._gradReinforce:resizeAs(self._softmaxoutput):zero()
   self._gradReinforce:scatter(2, self._index, 1)
   self._gradReinforce:cdiv(self._softmaxoutput)
   
   -- multiply by reward 
   self._gradReinforce:cmul(self:rewardAs(self._softmaxoutput))
   -- multiply by -1 ( gradient descent on input )
   self._gradReinforce:mul(-1)
   
   if self.epsilon > 0 then
      self._gradReinforce:mul(1-self.epsilon)
   end
   
   self._gradSoftmax = self._gradSoftmax or input.new()
   input.THNN.SoftMax_updateGradInput(
      self._linearoutput:cdata(), 
      self._gradReinforce:cdata(), 
      self._gradSoftmax:cdata(), 
      self._softmaxoutput:cdata()
   )
   
   -- gradient of linear
   self.gradInput[1] = self.gradInput[1] or input.new()
   self.gradInput[1]:resize(batchsize, 1, inputsize):zero()
   assert(self._gradSoftmax:nElement() == batchsize*nindex)
   self._gradSoftmax:resize(batchsize, 1, nindex)
   self.gradInput[1]:baddbmm(0, 1, self._gradSoftmax, self._weight)
   self.gradInput[1]:resizeAs(input)
   
   self.gradInput[2] = self.gradInput[2] or input.new()
   if self.gradInput[2]:nElement() ~= indices:nElement() then
      self.gradInput[2]:resize(indices:size()):zero()
   end
   
   return self.gradInput
end

function LSRC:accGradParameters(inputTable, gradOutput, scale)
   local input, indices = unpack(inputTable)
   assert(input:dim() == 2)
   assert(indices:dim() == 2)
   
   local batchsize = input:size(1)
   local inputsize = self.weight:size(2)
   local nindex = indices:size(2)
   
   assert(input:size(2) == inputsize)
   assert(torch.type(indices) == torch.type(self.output), 'Expecting input[2] to be '..torch.type(self.output))
   
   self._gradWeight = self._gradWeight or self.bias.new()
   self._gradWeight:resizeAs(self._weight):zero() -- batchsize x nindex x inputsize
   self._gradSoftmax:resize(batchsize, nindex, 1)
   self._gradSoftmax:mul(scale)
   local _input = input:view(batchsize, 1, inputsize)
   self._gradWeight:baddbmm(0, self._gradWeight, 1, self._gradSoftmax, _input)
   
   local _indices = indices:view(batchsize * nindex)
   local _gradWeight = self._gradWeight:view(batchsize * nindex, inputsize)
   self.gradWeight:indexAdd(1, _indices, _gradWeight)
   
   local _gradSoftmax = self._gradSoftmax:view(batchsize * nindex)
   self.gradBias:indexAdd(1, _indices, _gradSoftmax)
end

function LSRC:type(type, cache)
   if type then
      self._gradWeight = nil
      self._gradBias = nil
      self._weight = nil
      self._bias = nil
      self._linearoutput = nil
      self._softmaxoutput = nil
      self._gradReinforce = nil
      self._gradSoftmax = nil
   end
  
   local rtn
   if type and torch.type(self.weight) == 'torch.MultiCudaTensor' then
      assert(type == 'torch.CudaTensor', "Cannot convert a multicuda LSRC to anything other than cuda")
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
   
   if type then
      if type == 'torch.CudaTensor' then
         self.output = torch.CudaTensor()
         self.gradInput[2] = torch.CudaTensor()
      else
         self.output = torch.LongTensor()
         self.gradInput[2] = torch.CudaTensor()
      end
   end
   
   return rtn
end

function LSRC:clearState()
   self._gradWeight = nil
   self._gradBias = nil
   self._weight = nil
   self._bias = nil
   self._linearoutput = nil
   self._softmaxoutput = nil
   self._gradReinforce = nil
   self._gradSoftmax = nil
   self.output:set()
   self.gradInput[1]:set()
   self.gradInput[2]:set()
end

-- use a Linear instance to initialize LSRC
function LSRC:fromLinear(linear)
   self.weight = linear.weight
   self.gradWeight = linear.gradWeight
   self.bias = linear.bias
   self.gradBias = linear.gradBias
   self.inputsize = linear.weight:size(2)
   self.outputsize = linear.weight:size(1)
   return self
end

function LSRC:multicuda(device1, device2)
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

-- for efficient serialization using nn.Serial
local empty = _.clone(parent.dpnn_mediumEmpty)
table.insert(empty, '_weight')
table.insert(empty, '_gradWeight')
table.insert(empty, '_bias')
table.insert(empty, '_gradBias')
table.insert(empty, '_linearoutput')
table.insert(empty, '_softmaxoutput')
table.insert(empty, '_gradReinforce')
table.insert(empty, '_gradSoftmax')
LSRC.dpnn_mediumEmpty = empty
