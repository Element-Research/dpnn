------------------------------------------------------------------------
--[[ SpatialGlimpse ]]--
-- Ref A.: http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- input is a pair of Tensors: {image, location}
-- locations are x,y coordinates of the center of cropped patches. 
-- Coordinates are between -1,-1 (top-left) and 1,1 (bottom right)
-- output is a batch of glimpses taken in image at location (x,y)
-- size specifies width = height of glimpses
-- depth is number of patches to crop per glimpse (one patch per scale)
-- scale is the scale * size of successive cropped patches
------------------------------------------------------------------------
local SpatialGlimpse, parent = torch.class("nn.SpatialGlimpse", "nn.Module")

function SpatialGlimpse:__init(size, depth, scale)
   require 'image'
   self.size = size -- height == width
   self.depth = depth or 3
   self.scale = scale or 2
   
   assert(torch.type(self.size) == 'number')
   assert(torch.type(self.depth) == 'number')
   assert(torch.type(self.scale) == 'number')
   parent.__init(self)
   self.gradOutput = {torch.Tensor(), torch.Tensor()}
end

-- a bandwidth limited sensor which focuses on a location.
-- locations index the x,y coord of the center of the output glimpse
function SpatialGlimpse:updateOutput(inputTable)
   assert(torch.type(inputTable) == 'table')
   assert(#inputTable >= 2)
   local input, location = unpack(inputTable)
   
   self.output:resize(input:size(1), self.depth, input:size(2), self.size, self.size)
   
   self._crop = self._crop or output.new()
   self._pad = self._pad or input.new()
   
   for sampleIdx=1,output:size(1) do
      local outputSample = output[sampleIdx]
      local inputSample = input[sampleIdx]
      local xy = location[sampleIdx]
      -- (-1,-1) top left corner, (1,1) bottom right corner of image
      local x, y = xy:select(1,1), xy:select(1,2)
      -- (0,0), (1,1)
      x, y = (x+1)/2, (y+1)/2
      -- (0,0), (input:size(3)-1, input:size(4)-1)
      x, y = x*(input:size(3)-1), y*(input:size(4)-1)
      x = math.min(input:size(3)-1,math.max(0,x))
      y = math.min(input:size(4)-1,math.max(0,y))
      
      -- for each depth of glimpse : pad, crop, downscale
      for depth=1,self.depth do 
         local dst = outputSample[depth]
         local glimpseSize = self.size*(self.scale^(depth-1))
         
         -- add zero padding (glimpse could be partially out of bounds)
         local padSize = glimpseSize/2
         self._pad:resize(input:size(2), input:size(3)+padSize*2, input:size(4)+padSize*2):zero()
         local center = self._pad:narrow(2,padSize,input:size(3)):narrow(3,padSize,input:size(4))
         center:copy(inputSample)
         
         -- coord of top-left corner of patch that will be cropped w.r.t padded input
         -- is coord of center of patch w.r.t original non-padded input.
        
         -- crop it
         self._crop:resize(input:size(2), glimpseSize, glimpseSize)
         image.crop(self._crop, self._pad, x, y) -- crop is zero-indexed
         image.scale(dst, self._crop)
      end
   end
   
   self.output:resize(input:size(1), self.depth*input:size(2), self.size, self.size)
   return self.output
end

function SpatialGlimpse:updateGradInput(inputTable, gradOutput)
   local input, location = unpack(inputTable)
   local gradInput, gradLocation = unpack(self.gradInput)
   
   gradInput:resizeAs(input):zero()
   gradLocation:resizeAs(location):zero() -- no backprop through location
   
   gradOutput = gradOutput:view(input:size(1), self.depth, input:size(2), self.size, self.size)
   
   for sampleIdx=1,gradOutput:size(1) do
      local gradOutputSample = gradOutput[sampleIdx]
      local gradInputSample = gradInput[sampleIdx]
      local xy = location[sampleIdx]
      -- (-1,-1) top left corner, (1,1) bottom right corner of image
      local x, y = xy:select(1,1), xy:select(1,2)
      -- (0,0), (1,1)
      x, y = (x+1)/2, (y+1)/2
      -- (0,0), (input:size(3)-1, input:size(4)-1)
      x, y = x*(input:size(3)-1), y*(input:size(4)-1)
      x = math.min(input:size(3)-1,math.max(0,x))
      y = math.min(input:size(4)-1,math.max(0,y))
      
      -- for each depth of glimpse : pad, crop, downscale
      for depth=1,self.depth do 
         local src = gradOutputSample[depth]
         local glimpseSize = self.size*(self.scale^(depth-1))
         
         -- upscale glimpse for different depths
         self._crop:resizeAs(input:size(2), glimpseSize, glimpseSize)
         image.scale(self._crop, src)
         
         -- add zero padding (glimpse could be partially out of bounds)
         local padSize = glimpseSize/2
         self._pad:resize(input:size(2), input:size(3)+padSize*2, input:size(4)+padSize*2):zero()
         
         -- coord of top-left corner of patch that will be cropped w.r.t padded input
         -- is coord of center of patch w.r.t original non-padded input.
         local pad = self._pad:narrow(2, x+1, glimpseSize):narrow(3, y+1, glimpseSize)
         pad:copy(self._crop)
        
         -- copy into gradInput tensor (excluding padding)
         gradInputSample:add(pad:narrow(2, padSize+1, input:size(3)):narrow(3, padSize+1, input:size(4)))
      end
   end
end
