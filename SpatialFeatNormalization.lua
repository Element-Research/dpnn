--[[
   Color normalization (mean zeroing and dividing by standard deviation).
   Basic preprocessing step widely used in training classifier with images.
--]]

local SpatialFeatNormalization, Parent = torch.class('nn.SpatialFeatNormalization', 'nn.Module')

function SpatialFeatNormalization:__init(mean, std)
   Parent.__init(self)
   if mean:dim() ~= 1 then
      error('<SpatialFeatNormalization> Mean/Std should be 1D.')
   end
   self.mean = torch.Tensor()
   self.mean:resizeAs(mean):copy(mean)
   self.std = torch.Tensor()
   self.std:resizeAs(mean)
   if std ~= nil then self.std:copy(std) else self.std:fill(1) end
   self.noOfFeats = mean:size(1)
end

function SpatialFeatNormalization:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if input:dim() == 4 then
      -- Batch of image/s
      if input:size(2) ~= self.noOfFeats then
         error('<SpatialFeatNormalization> No. of Feats dont match.')
      else
         for i=1, self.noOfFeats do
            self.output[{{}, i, {}, {}}]:add(-self.mean[i])
            self.output[{{}, i, {}, {}}]:div(self.std[i])
         end
      end
   elseif input:dim() == 3 then
      -- single image
      if input:size(1) ~= self.noOfFeats then
         error('<SpatialFeatNormalization> No. of Feats dont match.')
      else
         for i=1, self.noOfFeats do
            self.output[{i, {}, {}}]:add(-self.mean[i])
            self.output[{i, {}, {}}]:div(self.std[i])
         end
      end
   else
      error('<SpatialFeatNormalization> invalid input dims.')
   end
   return self.output 
end

function SpatialFeatNormalization:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   if self.gradInput:dim() == 4 then
      -- Batch of image/s
      if self.gradInput:size(2) ~= self.noOfFeats then
         error('<SpatialFeatNormalization> No. of Feats dont match.')
      else
         for i=1, self.noOfFeats do
            self.gradInput[{{}, i, {}, {}}]:div(self.std[i])
         end
      end
   elseif self.gradInput:dim() == 3 then
      -- single image
      if self.gradInput:size(1) ~= self.noOfFeats then
         error('<SpatialFeatNormalization> No. of Feats dont match.')
      else
         for i=1, self.noOfFeats do
            self.gradInput[{i, {}, {}}]:div(self.std[i])
         end
      end
   else
      error('<SpatialFeatNormalization> invalid self.gradInput dims.')
   end
   return self.gradInput
end
