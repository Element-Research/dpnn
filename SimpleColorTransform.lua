--[[
   Simple Color transformation module: This module implements a simple data
   augmentation technique of changing the pixel values of input image by adding
   sample sampled small quantities.
   Works only
--]]

local SimpleColorTransform, Parent = torch.class('nn.SimpleColorTransform', 'nn.Module')

function SimpleColorTransform:__init(channels, range)
   self.train = true
end
