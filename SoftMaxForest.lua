local SoftMaxForest, parent = torch.class("nn.SoftMaxForest", "nn.Container")

function SoftMaxForest:__init(inputSize, trees, rootIds, gaterSize, gaterAct, accUpdate)
   local gaterAct = gaterAct or nn.Tanh() 
   local gaterSize = gaterSize or {} 
   
   -- experts
   self.experts = nn.ConcatTable()
   self.smts = {}
   for i,tree in ipairs(trees) do
      local smt = nn.SoftMaxTree(inputSize, tree, rootIds[i], accUpdate)
      table.insert(self._smts, smt)
      self.experts:add(smt)
   end
   
   -- gater
   self.gater = nn.Sequential()
   self.gater:add(nn.SelectTable(1)) -- ignore targets
   for i,hiddenSize in ipairs(gaterSize) do 
      self.gater:add(nn.Linear(inputSize, hiddenSize))
      self.gater:add(gaterAct:clone())
      inputSize = hiddenSize
   end
   self.gater:add(nn.Linear(inputSize, self.experts:size()))
   self.gater:add(nn.SoftMax())
   
   -- mixture
   self.trunk = nn.ConcatTable()
   self.trunk:add(self._gater)
   self.trunk:add(self._experts)
   self.mixture = nn.MixtureTable()
   self.module = nn.Sequential()
   self.module:add(self.trunk)
   self.module:add(self.mixture)
   parent.__init(self)
   self.modules[1] = self.module
end

function SoftMaxForest:updateOutput(input)
   self.output = self.module:updateOutput(input)
   return self.output
end

function SoftMaxForest:updateGradInput(input, gradOutput)
   self.gradInput = self.module:updateGradInput(input, gradOutput)
   return self.gradInput
end

function SoftMaxForest:accGradParameters(input, gradOutput, scale)
   self.module:accGradParameters(input, gradOutput, scale)
end

function SoftMaxForest:accUpdateGradParameters(input, gradOutput, lr)
   self.module:accUpdateGradParameters(input, gradOutput, lr)
end
