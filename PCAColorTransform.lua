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
   self.eigenValues = self.eigenValues
   self.std = std or 0.1
end
