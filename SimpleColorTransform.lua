--[[
   Simple Color transformation module: This module implements a simple data
   augmentation technique of changing the pixel values of input image by adding
   sample sampled small quantities.
   Works only
--]]

local SimpleColorTransform, Parent = torch.class('nn.SimpleColorTransform', 'nn.Module')

function SimpleColorTransform:__init(inputChannels, range)
   self.train = true
   self.inputChannels = inputChannels
   assert(inputChannels == range:nElement(),
          "Number of input channels and number of range values don't match.")
   self.range = range
end

function SimpleColorTransform:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if self.train then
      if self.output:nDimension() == 4 then
         local channels = self.output:size(2)
         assert(channels == self.inputChannels)

      else if self.output:nDimension() == 3 then

      else
         error("Invalid input dimensionality.")
      end
   end
   return self.output
end

function SimpleColortransform:updateGradInput(input, gradOutput)
   if self.train then
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   else
      error('backprop only defined while training')
   end
   return self.gradInput
end
