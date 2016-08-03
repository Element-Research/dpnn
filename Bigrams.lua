local Bigrams, parent = torch.class("nn.Bigrams", "nn.Module")

function Bigrams:__init(bigrams, nsample)
   require 'torchx'
   self.nsample = nsample
   
   assert(#bigrams > 1, "Expecting bigrams")
   -- table of torch.AliasMultinomial() instances
   self.bigrams = {}
   
   for wordid, bigram in ipairs(bigrams) do 
      assert(bigram.prob)
      assert(bigram.index)
      local am = torch.AliasMultinomial(bigram.prob)
      am.index = bigram.index
      self.bigrams[wordid] = am
   end
   
end

function Bigrams:updateOutput(input)
   assert(torch.type(input) == 'torch.LongTensor')
   assert(input:dim() == 1)
   local batchsize = input:size(1)
   self.output = torch.type(self.output) == 'torch.LongTensor' and self.output or torch.LongTensor()
   self.output:resize(batchsize, self.nsample) 

   self._output = self._output or self.output.new()
   self._output:resizeAs(self.output)
   
   for i=1,batchsize do
      local am = self.bigrams[input[i]]
      if not am then
         error("Missing index "..input[i]..". Only have bigrams for "..#self.bigrams.." words")
      end
      am:batchdraw(self._output[i])
      self.output[i]:index(am.index, 1, self._output[i])
   end
   
   return self.output   
end

function Bigrams:updateGradInput(input, gradOutput)
   self.gradInput = torch.type(self.gradInput) == 'torch.LongTensor' and self.gradInput or torch.LongTensor()
   self.gradInput:resizeAs(input):fill(0) 
   return self.gradInput
end

function Bigrams:statistics()
   local sum, count = 0, 0
   for wordid, bigram in ipairs(self.bigrams) do 
      sum = sum + bigram.index:nElement()
      count = count + 1
   end
   local meansize = sum/count
   return meansize
end
