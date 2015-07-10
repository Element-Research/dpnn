local ParallelTable, parent = nn.ParallelTable, nn.Container

function ParallelTable:profile()
   function ParallelTable:updateOutput(input)
      for i=1,#self.modules do
         local start = sys.clock()
         self.output[i] = self.modules[i]:updateOutput(input[i])
         if cutorch then cutorch.synchronize() end
         print(torch.type(self.modules[i])..' updateOutput: '..sys.clock() - start.." s")
      end
      return self.output
   end

   function ParallelTable:updateGradInput(input, gradOutput)
      for i,module in ipairs(self.modules) do
         local start = sys.clock()
         self.gradInput[i]= module:updateGradInput(input[i], gradOutput[i])
         if cutorch then cutorch.synchronize() end
         print(torch.type(module)..' updateGradInput: '..sys.clock() - start.." s")
      end
      return self.gradInput
   end

   function ParallelTable:accGradParameters(input, gradOutput, scale)
      scale = scale or 1
      for i,module in ipairs(self.modules) do
         local start = sys.clock()
         module:accGradParameters(input[i], gradOutput[i], scale)
         if cutorch then cutorch.synchronize() end
         print(torch.type(module)..' accGradParameters: '..sys.clock() - start.." s")
      end
   end

   function ParallelTable:accUpdateGradParameters(input, gradOutput, lr)
      lr = lr or 1
      for i,module in ipairs(self.modules) do
         local start = sys.clock()
         module:accUpdateGradParameters(input[i], gradOutput[i], lr)
         if cutorch then cutorch.synchronize() end
         print(torch.type(module)..' accUpdateGradParameters: '..sys.clock() - start.." s")
      end
   end
   parent.profile(self)
end
