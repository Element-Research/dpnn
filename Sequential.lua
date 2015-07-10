local Sequential, parent = nn.Sequential, nn.Container

function Sequential:profile()

   function Sequential:updateOutput(input)
      local currentOutput = input
      for i=1,#self.modules do
         local start = sys.clock()
         currentOutput = self.modules[i]:updateOutput(currentOutput)
         if cutorch then cutorch.synchronize() end
         print(torch.type(self.modules[i])..' updateOutput: '..sys.clock() - start.." s")
      end
      self.output = currentOutput
      return currentOutput
   end

   function Sequential:updateGradInput(input, gradOutput)
      local currentGradOutput = gradOutput
      local currentModule = self.modules[#self.modules]
      for i=#self.modules-1,1,-1 do
         local previousModule = self.modules[i]
         local start = sys.clock()
         currentGradOutput = currentModule:updateGradInput(previousModule.output, currentGradOutput)
         if cutorch then cutorch.synchronize() end
         print(torch.type(currentModule)..' updateGradInput: '..sys.clock() - start.." s")
         currentModule = previousModule
      end
      local start = sys.clock()
      currentGradOutput = currentModule:updateGradInput(input, currentGradOutput)
      if cutorch then cutorch.synchronize() end
      print(torch.type(currentModule)..' updateGradInput: '..sys.clock() - start.." s")
      self.gradInput = currentGradOutput
      return currentGradOutput
   end

   function Sequential:accGradParameters(input, gradOutput, scale)
      scale = scale or 1

      local currentGradOutput = gradOutput
      local currentModule = self.modules[#self.modules]
      for i=#self.modules-1,1,-1 do
         local previousModule = self.modules[i]
         local start = sys.clock()
         currentModule:accGradParameters(previousModule.output, currentGradOutput, scale)
         if cutorch then cutorch.synchronize() end
         print(torch.type(currentModule)..' accGradParameters: '..sys.clock() - start.." s")
         currentGradOutput = currentModule.gradInput
         currentModule = previousModule
      end
      
      local start = sys.clock()
      currentModule:accGradParameters(input, currentGradOutput, scale)
      if cutorch then cutorch.synchronize() end
      print(torch.type(currentModule)..' accGradParameters: '..sys.clock() - start.." s")
   end

   function Sequential:backward(input, gradOutput, scale)
      scale = scale or 1
      local currentGradOutput = gradOutput
      local currentModule = self.modules[#self.modules]
      for i=#self.modules-1,1,-1 do
         local previousModule = self.modules[i]
         local start = sys.clock()
         currentGradOutput = currentModule:backward(previousModule.output, currentGradOutput, scale)
         if cutorch then cutorch.synchronize() end
         print(torch.type(currentModule)..' backward: '..sys.clock() - start.." s")
         currentModule.gradInput = currentGradOutput
         currentModule = previousModule
      end
      local start = sys.clock()
      currentGradOutput = currentModule:backward(input, currentGradOutput, scale)
      if cutorch then cutorch.synchronize() end
      print(torch.type(currentModule)..' backward: '..sys.clock() - start.." s")
      self.gradInput = currentGradOutput
      return currentGradOutput
   end

   function Sequential:accUpdateGradParameters(input, gradOutput, lr)
      local currentGradOutput = gradOutput
      local currentModule = self.modules[#self.modules]
      for i=#self.modules-1,1,-1 do
         local previousModule = self.modules[i]
         local start = sys.clock()
         currentModule:accUpdateGradParameters(previousModule.output, currentGradOutput, lr)
         if cutorch then cutorch.synchronize() end
         print(torch.type(currentModule)..' accUpdateGradParameters: '..sys.clock() - start.." s")
         currentGradOutput = currentModule.gradInput
         currentModule = previousModule
      end

      local start = sys.clock()
      currentModule:accUpdateGradParameters(input, currentGradOutput, lr)
      if cutorch then cutorch.synchronize() end
      print(torch.type(currentModule)..' accUpdateGradParameters: '..sys.clock() - start.." s")
   end

   parent.profile(self)
end
