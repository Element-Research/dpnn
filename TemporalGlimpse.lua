------------------------------------------------------------------------
--[[ TemporalGlimpse ]]--
-- Ref A.: http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf
-- a glimpse is the concatenation of down-scaled cropped sub-sequences 
-- of increasing scale around a given location in a given sequence.
-- input is a pair of Tensors: {sequence, location}
-- locations are the x coordinate of the center of cropped sub-sequence. 
-- Coordinates are between -1 (left) and 1 (right)
-- output is a batch of glimpses taken in sequence at locations x
-- size specifies width (number of elements) of glimpses (sub-sequences)
-- The input is expected to be of shape : batchSize x width x nChannel
------------------------------------------------------------------------
local TemporalGlimpse, parent = torch.class("nn.TemporalGlimpse", "nn.Module")

function TemporalGlimpse:__init(size)
   self.size = size -- width (number of elements in sequence)\
   assert(torch.type(self.size) == 'number')
   parent.__init(self)
   self.gradInput = {torch.Tensor(), torch.Tensor()}
end

-- a bandwidth limited sensor which focuses on a location.
-- locations index the x,y coord of the center of the output glimpse
function TemporalGlimpse:updateOutput(inputTable)
   assert(torch.type(inputTable) == 'table')
   assert(#inputTable >= 2)
   local input, location = unpack(inputTable)
   input, location = self:toBatch(input, 2), self:toBatch(location, 1)
   assert(input:dim() == 3 and location:dim() == 2)
   
   self.output:resize(input:size(1), self.size, input:size(3))
   
   self._pad = self._pad or input.new()
   
   for sampleIdx=1,self.output:size(1) do
      local outputSample = self.output[sampleIdx]
      local inputSample = input[sampleIdx]
      -- (-1) left corner, (1) right corner of sequence
      local x = location[{sampleIdx, 1}]
      -- (0 -> 1)
      x = (x+1)/2
      
      local glimpseSize = self.size
      
      -- add zero padding (glimpse could be partially out of bounds)
      local padSize = math.floor((glimpseSize-1)/2)
      self._pad:resize(input:size(2)+padSize*2, input:size(3)):zero()
      self._pad:narrow(1,padSize+1,input:size(2)):copy(inputSample)
      
      -- crop it
      local w = self._pad:size(1)-glimpseSize
      local x = math.min(w,math.max(0,x*w))
      outputSample:copy(self._pad:narrow(1,x+1,glimpseSize))
   end
   
   self.output = self:fromBatch(self.output, 1)
   return self.output
end

function TemporalGlimpse:updateGradInput(inputTable, gradOutput)
   local input, location = unpack(inputTable)
   local gradInput, gradLocation = unpack(self.gradInput)
   input, location = self:toBatch(input, 2), self:toBatch(location, 1)
   gradOutput = self:toBatch(gradOutput, 2)
   
   gradInput:resizeAs(input):zero()
   gradLocation:resizeAs(location):zero() -- no backprop through location
   
   for sampleIdx=1,gradOutput:size(1) do
      local gradOutputSample = gradOutput[sampleIdx]
      local gradInputSample = gradInput[sampleIdx]
      -- (-1) left corner, (1) right corner of sequence
      local x = location[{sampleIdx, 1}]
      -- (0 -> 1)
      x = (x+1)/2
      
      -- for each depth of glimpse : pad, crop, downscale
      local glimpseSize = self.size
      
      -- add zero padding (glimpse could be partially out of bounds)
      local padSize = math.floor((glimpseSize-1)/2)
      self._pad:resize(input:size(2)+padSize*2, input:size(3)):zero()
      
      local w = self._pad:size(1)-glimpseSize
      local x = math.min(w,math.max(0,x*w))
      self._pad:narrow(1, x+1, glimpseSize):copy(gradOutputSample)
     
      -- copy into gradInput tensor (excluding padding)
      gradInputSample:add(self._pad:narrow(1, padSize+1, input:size(2)))
   end
   
   self.gradInput[1] = self:fromBatch(gradInput, 1)
   self.gradInput[2] = self:fromBatch(gradLocation, 1)
   
   return self.gradInput
end
