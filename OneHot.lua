local OneHot, parent = torch.class('nn.OneHot', 'nn.Module')

-- adapted from https://github.com/karpathy/char-rnn
-- and https://github.com/hughperkins/char-rnn-er

function OneHot:__init(outputSize)
   parent.__init(self)
   self.outputSize = outputSize
end

function OneHot:updateOutput(input)
   local size = input:size():totable()
   table.insert(size, self.outputSize)
   
   self.output:resize(unpack(size)):zero()
   
   size[input:dim()+1] = 1
   local input_ = input:view(unpack(size))
   
   if torch.type(input) == 'torch.CudaTensor' or torch.type(input) == 'torch.ClTensor' then
      self.output:scatter(self.output:dim(), input_, 1)
   else
      if torch.type(input) ~= 'torch.LongTensor' then
         self._input = self._input or torch.LongTensor()
         self._input:resize(input_:size()):copy(input_)
         input_ = self._input
      end
      self.output:scatter(self.output:dim(), input_, 1)
   end
   
   return self.output
end

function OneHot:updateGradInput(input, gradOutput)
   self.gradInput:resize(input:size()):zero()
   return self.gradInput
end

function OneHot:type(type, typecache)
   self._input = nil
   return parent.type(self, type, typecache)
end
