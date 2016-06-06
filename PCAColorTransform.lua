--[[
   Color transformation module: Commonly used data augmentation technique.
   Random color noise is added to input image/images based on the Principal
   Component Analysis (PCA) of pixel values.

   Arguments
   -> eigenVectors: Each row represent an eigen vector.
   -> eigenValues: Corresponding eigen values.
   -> std: std of gaussian distribution for augmentation (default 0.1).
--]]

local PCAColorTransform, Parent = torch.class('nn.PCAColorTransform', 'nn.Module')

function PCAColorTransform:__init(inputChannels, eigenVectors, eigenValues, std)
   Parent.__init(self)

   self.train = true
   self.inputChannels = inputChannels
   assert(inputChannels == eigenVectors:size(1),
          "Number of input channels do not match number of eigen vectors.")
   assert(eigenVectors:size(2) == eigenVectors:size(1),
          "Invalid dimensionality: eigen vectors.")
   assert(inputChannels == eigenValues:nElement(),
          "Number of input channels do not match number of eigen values.")

   self.eigenVectors = eigenVectors
   self.eigenValues = eigenValues
   self.std = std or 0.1
end

function PCAColorTransform:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if self.train then
      self.noise = self.noise or self.output.new()
      self.alphas = self.alphas or self.output.new()
      self._tempNoise = self._tempNoise or self.output.new()
      self._tempNoiseExpanded = self._tempNoiseExpanded or self.output.new()
      self._tempNoiseSamples = self._tempNoiseSamples or self.output.new()
      self._tempLambda = self._tempLambda or self.output.new()
      self._tempLambdaExpanded = self._tempLambdaExpanded or self.output.new()

      if self.output:nDimension() == 4 then
         local batchSize = self.output:size(1)
         local channels = self.output:size(2)
         local height = self.output:size(3)
         local width = self.output:size(4)
         assert(channels == self.inputChannels)
         
         -- Randomly sample noise for each channel and scale by eigen values
         self.alphas:resize(channels, batchSize)
         self.alphas:normal(0, self.std)
         self._tempLambda = self.eigenValues:view(self.inputChannels, 1)
         self._tempLambdaExpanded = self._tempLambda:expand(channels, batchSize)
         self.alphas:cmul(self._tempLambdaExpanded)

         -- Scale by eigen vectors 
         self.noise:resize(batchSize, self.inputChannels):zero()
         self.noise:t():addmm(self.eigenVectors, self.alphas)

         -- Add noise to the input
         self._tempNoise = self.noise:view(batchSize, self.inputChannels, 1, 1)
         self._tempNoiseExpanded:expand(self._tempNoise, batchSize,
                                        channels, height, width)
         self.output:add(self._tempNoiseExpanded)

      elseif self.output:nDimension() == 3 then
         local channels = self.output:size(1)
         local height = self.output:size(2)
         local width = self.output:size(3)
         assert(channels == self.inputChannels)

         -- Randomly sample noise for each channel and scale by eigen values
         self.alphas:resize(channels, 1)
         self.alphas:normal(0, self.std)
         self._tempLambda = self.eigenValues:view(self.inputChannels, 1)
         self._tempLambdaExpanded = self._tempLambda:expand(channels, 1)
         self.alphas:cmul(self._tempLambdaExpanded)

         -- Scale by eigen vectors 
         self.noise:resize(1, self.inputChannels):zero()
         self.noise:t():addmm(self.eigenVectors, self.alphas)

         -- Add noise to the input
         self._tempNoise = self.noise:view(self.inputChannels, 1, 1)
         self._tempNoiseExpanded:expand(self._tempNoise, channels,
                                        height, width)
         self.output:add(self._tempNoiseExpanded)
      else
         error("Invalid input dimensionality.")
      end
   end
   return self.output
end

function PCAColorTransform:updateGradInput(input, gradOutput)
   if self.train then
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   else
      error('backprop only defined while training')
   end
   return self.gradInput
end

function PCAColorTransform:type(type, tensorCache)
   self.noise = nil
   self.alphas = nil
   self._tempLambda = nil
   self._tempLambdaExpanded = nil
   self._tempNoise = nil
   self._tempNoiseExpanded = nil
   Parent.type(self, type, tensorCache)
end

function PCAColorTransform:__tostring__()
  return string.format('%s channels: %d, std: %f', torch.type(self),
                        self.inputChannels, self.std)
end
