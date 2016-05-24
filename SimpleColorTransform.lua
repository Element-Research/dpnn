--[[
   Simple Color transformation module: This module implements a simple data
   augmentation technique of changing the pixel values of input image by adding
   sample sampled small quantities.
   Works only
--]]

local SimpleColorTransform, Parent = torch.class('nn.SimpleColorTransform', 'nn.Module')

function SimpleColorTransform:__init(inputChannels, range)
   Parent.__init(self)

   self.train = true
   self.inputChannels = inputChannels
   assert(inputChannels == range:nElement(),
          "Number of input channels and number of range values don't match.")
   self.range = range
end

function SimpleColorTransform:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if self.train then
      self.noise = self.noise or self.output.new()
      self._tempNoise = self._tempNoise or self.output.new()
      self._tempNoiseExpanded = self._tempNoiseExpanded or self.output.new()
      self._tempNoiseSamples = self._tempNoiseSamples or self.output.new()

      if self.output:nDimension() == 4 then
         local batchSize = self.output:size(1)
         local channels = self.output:size(2)
         local height = self.output:size(3)
         local width = self.output:size(4)
         assert(channels == self.inputChannels)
         
         -- Randomly sample noise for each channel 
         self.noise:resize(batchSize, channels)
         for i=1, channels do
            self.noise[{{}, {i}}]:uniform(-self.range[i], self.range[i])
         end
         self._tempNoise = self.noise:view(batchSize, self.inputChannels, 1, 1)
         self._tempNoiseExpanded:expand(self._tempNoise, batchSize,
                                        channels, height, width)
         self._tempNoiseSamples:resizeAs(self._tempNoiseExpanded)
                               :copy(self._tempNoiseExpanded)
         self.output:add(self._tempNoiseSamples)

      elseif self.output:nDimension() == 3 then
         local channels = self.output:size(1)
         local height = self.output:size(2)
         local width = self.output:size(3)
         assert(channels == self.inputChannels)

         -- Randomly sample noise for each channel 
         self.noise:resize(channels)
         for i=1, channels do
            self.noise[i] = torch.uniform(-self.range[i], self.range[i])
         end
         self._tempNoise = self.noise:view(self.inputChannels, 1, 1)
         self._tempNoiseExpanded:expand(self._tempNoise, channels,
                                        height, width)
         self._tempNoiseSamples:resizeAs(self._tempNoiseExpanded)
                               :copy(self._tempNoiseExpanded)
         self.output:add(self._tempNoiseSamples)
      else
         error("Invalid input dimensionality.")
      end
   end
   return self.output
end

function SimpleColorTransform:updateGradInput(input, gradOutput)
   if self.train then
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   else
      error('backprop only defined while training')
   end
   return self.gradInput
end

function SimpleColorTransform:type(type, tensorCache)
   self.noise = nil
   self._tempNoise = nil
   self._tempNoiseExpanded = nil
   self._tempNoiseSamples = nil
   Parent.type(self, type, tensorCache)
end

function SimpleColorTransform:__tostring__()
  return string.format('SimpleColorTransform', torch.type(self))
end
