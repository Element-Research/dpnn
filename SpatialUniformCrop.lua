local SpatialUniformCrop, parent = torch.class("nn.SpatialUniformCrop", "nn.Module")

function SpatialUniformCrop:__init(oheight, owidth)
   parent.__init(self)
   self.oheight = oheight
   self.owidth = owidth or oheight
end

function SpatialUniformCrop:updateOutput(input)
   assert(input:dim() == 4, "only batchmode is supported")
   
   self.output:resize(input:size(1), input:size(2), self.oheight, self.owidth)
   self.coord = self.coord or torch.IntTensor()
   self.coord:resize(input:size(1), 2)
   
   local iH, iW = input:size(3), input:size(4)
   if self.train ~= false then
      for i=1,input:size(1) do
         -- do random crop
         local h1 = math.ceil(torch.uniform(1e-2, iH-self.oheight))
         local w1 = math.ceil(torch.uniform(1e-2, iW-self.owidth))
         local crop = input[i]:narrow(2,h1,self.oheight):narrow(3,w1,self.owidth)
         self.output[i]:copy(crop)
         -- save crop coordinates for backward
         self.coord[{i,1}] = h1
         self.coord[{i,2}] = w1
      end
   else
      -- use center crop
      local h1 = math.ceil((iH-self.oheight)/2)
      local w1 = math.ceil((iW-self.owidth)/2)
      local crop = input[i]:narrow(2,h1,self.oheight):narrow(3,w1,self.owidth)
      self.output[i]:copy(crop)
   end
   
   return self.output
end

function SpatialUniformCrop:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   
   for i=1,input:size(1) do
      local h1, w1 = self.coord[{i,1}], self.coord[{i,2}]
      self.gradInput[i]:narrow(2,h1,self.oheight):narrow(3,w1,self.owidth):copy(gradOutput[i])
   end
   
   return self.gradInput
end

function SpatialUniformCrop:type(type, cache)
   self.coord = nil
   return parent.type(self, type, cache)
end
