local SpatialUniformCrop, parent = torch.class("nn.SpatialUniformCrop", "nn.Module")

function SpatialUniformCrop:__init(oheight, owidth, scale)
   parent.__init(self)
   self.scale = scale or nil
   if self.scale ~= nil then
      assert(torch.type(scale)=='table')
      self.scaler = nn.SpatialReSampling{owidth=owidth, oheight=oheight}
   end
   self.oheight = oheight
   self.owidth = owidth or oheight
end

function SpatialUniformCrop:updateOutput(input)
   input = self:toBatch(input, 3)
   
   self.output:resize(input:size(1), input:size(2), self.oheight, self.owidth)
   self.coord = self.coord or torch.IntTensor()
   self.coord:resize(input:size(1), 2)

   if self.scale ~= nil then
      self.scales = self.scales or torch.FloatTensor()
      self.scales:resize(input:size(1))
   end
  
   local iH, iW = input:size(3), input:size(4)
   if self.train ~= false then
      if self.scale ~= nil then
         for i=1,input:size(1) do
            -- do random crop
            local s = torch.uniform(self.scale['min'] or self.scale[1], self.scale['max'] or self.scale[2])
            local soheight = math.ceil(s*self.oheight)
            local sowidth = math.ceil(s*self.owidth)

            local h = math.ceil(torch.uniform(1e-2, iH-soheight))
            local w = math.ceil(torch.uniform(1e-2, iW-sowidth))
           
            local ch = math.ceil(iH/2 - (iH-soheight)/2 + h)
            local cw = math.ceil(iW/2 - (iH-sowidth)/2 + w)

            local h1 = ch - math.ceil(soheight/2)
            local w1 = cw - math.ceil(sowidth/2)
            if h1 < 1 then h1 = 1 end
            if w1 < 1 then w1 = 1 end

            local crop = input[i]:narrow(2, h1, soheight):narrow(3, w1, sowidth)

            self.output[i]:copy(self.scaler:forward(crop))
            -- save crop coordinates and scale for backward
            self.scales[i] = s
            self.coord[{i,1}] = h
            self.coord[{i,2}] = w
         end
      else
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
      end
   else
      -- use center crop
      local h1 = math.ceil((iH-self.oheight)/2)
      local w1 = math.ceil((iW-self.owidth)/2)
      local crop = input:narrow(3,h1,self.oheight):narrow(4,w1,self.owidth)
      self.output:copy(crop)
   end
   
   self.output = self:fromBatch(self.output, 1)
   return self.output
end

function SpatialUniformCrop:updateGradInput(input, gradOutput)
   input = self:toBatch(input, 3)
   gradOutput = self:toBatch(gradOutput, 3)
   
   self.gradInput:resizeAs(input):zero()
   if self.scale ~= nil then
      local iH, iW = input:size(3), input:size(4)
      for i=1,input:size(1) do
         local s = self.scales[i]
         local soheight = math.ceil(s*self.oheight)
         local sowidth = math.ceil(s*self.owidth)

         local h, w = self.coord[{i,1}], self.coord[{i,2}]
        
         local ch = math.ceil(iH/2 - (iH-soheight)/2 + h)
         local cw = math.ceil(iW/2 - (iH-sowidth)/2 + w)

         local h1 = ch - math.ceil(soheight/2)
         local w1 = cw - math.ceil(sowidth/2)
         if h1 < 1 then h1 = 1 end
         if w1 < 1 then w1 = 1 end

         local crop = input[i]:narrow(2, h1, soheight):narrow(3, w1, sowidth)
         local samplerGradInput = self.scaler:updateGradInput(crop, gradOutput[i])

         self.gradInput[i]:narrow(2, h1, soheight):narrow(3, w1, sowidth):copy(samplerGradInput)
      end
   else
      for i=1,input:size(1) do
         local h1, w1 = self.coord[{i,1}], self.coord[{i,2}]
         self.gradInput[i]:narrow(2,h1,self.oheight):narrow(3,w1,self.owidth):copy(gradOutput[i])
      end
   end
   
   self.gradInput = self:fromBatch(self.gradInput, 1)
   return self.gradInput
end

function SpatialUniformCrop:type(type, cache)
   self.coord = nil
   return parent.type(self, type, cache)
end
