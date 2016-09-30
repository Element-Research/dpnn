------------------------------------------------------------------------
--[[ Bigrams ]]--
-- samples (w/o replacement) nsample next words given previous word.
-- The problem is sampling from the larger bigram distributions.
-- For these, we sample the most frequent using torch.multinomial, and
-- remainder with torch.AliasMultinomial (and hope for few duplicates)
------------------------------------------------------------------------
local Bigrams, parent = torch.class("nn.Bigrams", "nn.Module")

function Bigrams:__init(bigrams, nsample, multiratio, multimax) 
   require 'torchx'
   self.nsample = nsample
   self.multiratio = multiratio or 0.1
   self.multimax = multimax or 1000
   
   assert(#bigrams > 1, "Expecting bigrams")
   -- table of torch.AliasMultinomial() instances
   self.bigrams = {}
   
   local sortval, sortidx, _index
   for wordid, bigram in ipairs(bigrams) do 
      assert(bigram.prob)
      assert(bigram.index)
      local nbg = bigram.prob:size(1)
      assert(bigram.index:size(1) == nbg)
      
      local bg = {}
      if nbg <= self.nsample then
         bg.all = bigram.index
      elseif nbg <= self.multimax then
         bg.mindex = bigram.index
         bg.mprob = bigram.prob
         bg.nmulti = self.nsample
      else
         sortval = sortval or bigram.prob.new()
         sortidx = sortidx or torch.LongTensor()
         
         -- order by probability
         sortval:sort(sortidx, bigram.prob, 1, true)
         bigram.prob:copy(sortval)
         _index = _index or bigram.index.new()
         _index:resizeAs(bigram.index):copy(bigram.index)
         bigram.index:index(_index, 1, sortidx)
         
         local nmulti = math.max(self.nsample, math.floor(self.multiratio*nbg))
         
         -- torch.multinomial
         bg.mindex=bigram.index:narrow(1,1,nmulti)
         bg.mprob=bigram.prob:narrow(1,1,nmulti)
         bg.probmulti = bg.mprob:sum()
         bg.mprob:div(bg.mprob:sum()) -- renormalize
         
         -- torch.AliasMultinomial
         bg.aindex=bigram.index:sub(nmulti+1,nbg)
         bg.aprob=bigram.prob:sub(nmulti+1,nbg)
         bg.aprob:div(bg.aprob:sum()) -- renormalize
         bg.alias = torch.AliasMultinomial(bg.aprob)
         
         -- #samples drawn using torch.multinomial
         bg.nmulti = math.max(self.nsample-1, math.floor(bg.probmulti*self.nsample))
         -- #samples drawn using torch.AliasMultinomial
         bg.nalias = self.nsample - bg.nmulti
      end
      
      self.bigrams[wordid] = bg
   end
   
end

function Bigrams:updateOutput(input)
   assert(torch.type(input) == 'torch.LongTensor')
   assert(input:dim() == 1, tostring(input:size()))
   local batchsize = input:size(1)
   self.output = torch.type(self.output) == 'torch.LongTensor' and self.output or torch.LongTensor()
   self.output:resize(batchsize, self.nsample):zero()

   self._output = self._output or self.output.new()
   self._output:resize(self.nsample)
   
   for i=1,batchsize do
      local bg = self.bigrams[input[i]]
      
      local output = self.output[i]
      
      if bg then
         if bg.nmulti then
            local moutput = self._output:sub(1,bg.nmulti)
            bg.mprob.multinomial(moutput, bg.mprob, bg.nmulti, false) -- sample without replacement
            assert(moutput:size(1) == bg.nmulti)
            output:sub(1,bg.nmulti):index(bg.mindex, 1, moutput)
            
            if bg.alias then
               local aoutput = self._output:sub(bg.nmulti+1, self.nsample)
               bg.alias:batchdraw(aoutput)
               output:sub(bg.nmulti+1, self.nsample):index(bg.aindex, 1, aoutput)
            end
         else
            local nbg = bg.all:size(1)
            output:sub(1, nbg):copy(bg.all)
            if nbg < self.nsample then -- fill the rest with random negative samples (TODO use unigrams instead)
               output:sub(nbg+1,self.nsample):random(1,#self.bigrams)
            end
         end
      else
         if self.maskzero then
            output:random(1,#self.bigrams)
         else
            error("Missing index "..input[i]..". Only have bigrams for "..#self.bigrams.." words")
         end
      end
      
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
