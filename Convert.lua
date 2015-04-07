------------------------------------------------------------------------
--[ nn.Convert ]--
-- Module to convert between different data formats
-- nn.Convert('bchw', 'bf') or nn.Convert('chw', 'f')
------------------------------------------------------------------------
local Convert, parent = torch.class("nn.Convert", "nn.Container")

function Convert:__init(inputShape, outputShape)
   inputShape = inputShape:find('b') and inputShape or ('b'..inputShape)
   self.inputShape = inputShape
   outputShape = outputShape:find('b') and outputShape or ('b'..outputShape)
   self.outputShape = outputShape
   -- number of dims in batch mode
   self.nInputDim = #inputShape
   self.nOutputDim = #outputShape
   -- is the outputShape just a transposition of the inputShape?
   if self.nInputDim == self.nOutputDim then
      self.transposition = true
      for i=1,self.nInputDim do
         if not self.outputShape:find(self.inputShape:sub(i,i)) then
            self.transposition = false
            break
         end
      end
   end
   parent.__init(self)
end

-- post-initialization
function Convert:buildConverter(input)
   assert(input:dim() == self.nInputDim, "Only supports batch mode (for now)")
   if self.transposition then
      self.converter = self:transpose(self.outputShape)
   else
      if (torch.type(self[self.outputShape]) ~= 'function') then
         error(string.format("Unrecognized conversion of shape %s to %s", self.inputShape, self.outputShape))
      end
      self.converter = self[self.outputShape](self, input)
   end
   self.modules[1] = self.converter
end

function Convert:updateOutput(input)
   if not self.converter then
      self:buildConverter(input)
   end
   self.output = self.converter:updateOutput(input)
   return self.output
end

function Convert:updateGradInput(input, gradOutput)
   self.gradInput = self.converter:updateGradInput(input, gradOutput)
   return self.gradInput
end

function Convert:accGradParameters(input, gradOutput, scale)
   self.converter:accGradParameters(input, gradOutput, scale)
end

function Convert:accUpdateGradParameters(input, gradOutput, lr)
   self.converter:accUpdateGradParameters(input, gradOutput, lr)
end

-- batch feature
function Convert:bf(input)
   local b_pos = self:findAxis('b', self.inputShape)
   local dim = #self.inputShape
   if self.inputShape == 'bt' then
      error"Conversion of shape bt to bf not supported: open an issue on github"
   end
   -- was b
   if dim == 1 then
      return nn.Reshape(1)
   end
   -- was b...
   local modula
   if b_pos ~= 1 then
      modula = nn.Transpose({1, b_pos})
   end
   if dim > 2 then
      local transpose = modula
      local sampleSize = input:select(self:findAxis('b'),1):nElement()
      local reshape = nn.Reshape(sampleSize)
      if transpose then
         modula = nn.Sequential()
         modula:add(transpose)
         modula:add(reshape)
      else
         modula = reshape
      end
   end
   return modula or nn.Identity()
end

-- each example is a scalar; batch is a vector
function Convert:b(input)
   local b_pos = self:findAxis('b')
   if self.inputShape == 'bt' or self.inputShape == 'tb' then
      local t_pos = self:findAxis('t')
      -- select first set of classes
      return nn.Select(t_pos, 1)
   elseif self.inputShape == 'bf' or self.inputShape == 'fb' then
      -- this wont work as expected with size(f) > 1
      local f_pos = self:findAxis('f')
      if input:size(f_pos) > 1 then
         error("Cannot convert shape "..self.inputShape.." to b when feature > 1")
      end
      return nn.Select(f_pos, 1)
   else
      error("Cannot convert shape "..self.inputShape.." to shape b")
   end
end

-- returns the current shape of the data
function Convert:default()
   return nn.Identity()
end

-- multi-class (batch target)
function Convert:bt()
   local b_pos = self:findAxis('b')
   local modula
   if self.inputShape == 'b' then
      modula = nn.Reshape(1)
   else
      error("cannot convert shape '"..self.inputShape.."' to bt")
   end
   return modula
end

-- a generic function for transposing shape axes
function Convert:transpose(newShape)
   if newShape == self.inputShape then
      return nn.Identity()
   end
   local inputShape = {}
   for i=1,#self.inputShape do
      table.insert(inputShape, self.inputShape:sub(i,i))
   end
   local transpositions = {}
   for i=1,#newShape do
      local j = _.indexOf(inputShape, newShape:sub(i,i))
      if i ~= j then
         local char = inputShape[i]
         inputShape[i] = inputShape[j]
         inputShape[j] = char
         table.insert(transpositions, {j, i})
      end
   end
   return nn.Transpose(unpack(transpositions))
end

function Convert:findAxis(axis_char, shape, silent)
   shape = shape or self.inputShape
   local axis_pos = shape:find(axis_char)
   if (not silent) and (not axis_pos) then
      error("Provided shape '"..shape.."' has no axis '"..axis_char.."'", 2)
   end
   return axis_pos
end
