--[[
   Dropout edges rows or columns to simulate imperfect bounding boxes. 
--]]

local SpatialRegionDropout, Parent = torch.class('nn.SpatialRegionDropout', 'nn.Module')

function SpatialRegionDropout:__init(p)
   Parent.__init(self)
   self.p = p or 0.2 -- ratio of total number of rows or cols
   self.train = true
   self.noise = torch.Tensor()
   if self.p >= 1 or self.p < 0 then
      error('<SpatialRegionDropout> illegal percentage, must be 0 <= p < 1')
   end
end

function SpatialRegionDropout:setp(p)
   self.p = p
end

-- Region Types
-- 1: Dropout p ratio of top rows
-- 2: Dropout p ratio of bottom rows
-- 3: Dropout p ratio of leftmost cols
-- 4: Dropout p ratio of rightmost cols
function SpatialRegionDropout:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   if self.train then
      self.noise:resizeAs(input):fill(1)
      self.regionType = torch.random(4)
      if input:dim() == 4 then
         local height = input:size(3)
         local width = input:size(4)
         if self.regionType == 1 then
            self.noise[{{}, {}, {1, math.floor(height*self.p)}}]:fill(0)
         elseif self.regionType == 2 then
            self.noise[{{}, {}, 
                      {height-math.floor(height*self.p)+1, height}}]:fill(0)
         elseif self.regionType == 3 then
            self.noise[{{}, {}, {}, {1, math.floor(width*self.p)}}]:fill(0)
         elseif self.regionType == 4 then
            self.noise[{{}, {}, {},
                       {width-math.floor(width*self.p)+1, width}}]:fill(0)
         end
      elseif input:dim() == 3 then
         local height = input:size(2)
         local width = input:size(3)
         if self.regionType == 1 then
            self.noise[{{}, {1, math.floor(height*self.p)}}]:fill(0)
         elseif self.regionType == 2 then
            self.noise[{{}, 
                       {height-math.floor(height*self.p)+1, height}}]:fill(0)
         elseif self.regionType == 3 then
            self.noise[{{}, {}, {1, math.floor(width*self.p)}}]:fill(0)
         elseif self.regionType == 4 then
            self.noise[{{}, {}, 
                       {width-math.floor(width*self.p)+1, width}}]:fill(0)
         end
      else
         error('Input must be 4D (nbatch, nfeat, h, w) or 3D (nfeat, h, w)')
      end
      self.noise:div(1-self.p)
      self.output:cmul(self.noise)
   end
   return self.output
end

function SpatialRegionDropout:updateGradInput(input, gradOutput)
   if self.train then
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      self.gradInput:cmul(self.noise)
   else
      error('Backpropagation is only defined for training.')
   end
   return self.gradInput
end

function SpatialRegionDropout:__tostring__()
   return string.format('%s p: %f', torch.type(self), self.p)
end
