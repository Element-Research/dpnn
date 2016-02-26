--[[
  Fire module as explained in SqueezeNet http://arxiv.org/pdf/1602.07360v1.pdf.
--]]

local FireModule, Parent = torch.class('nn.FireModule', 'nn.Module')

function FireModule:__init(nInputPlane, s1x1, e1x1, e1x3, activation)
   self.nInputPlane = nInputPlane
   self.s1x1 = s1x1
   self.e1x1 = e1x1
   self.e1x3 = e1x3
   self.activation = activation or 'ReLU'

   if self.s1x1 > (self.e1x1 + self.e1x3) then
      print('Warning: <FireModule> s1x1 is recommended to be smaller then e1x1+e1x3')
   end
   
   self.module = nn.Sequential()
   self.squeeze = nn.SpatialConvolution(nInputPlane, s1x1, 1, 1)
   self.expand = nn.Concat(1)
   self.expand:add(nn.SpatialConvolution(s1x1, e1x1, 1, 1))
   self.expand:add(nn.SpatialConvolution(s1x1, e1x3, 3, 3, 1, 1, 1, 1))

   -- Fire Module
   self.module:add(self.squeeze)
   self.module:add(nn[self.activation]())
   self.module:add(self.expand)
   self.module:add(nn[self.activation]())
end

function FireModule:updateOutput(input)
   self.output = self.module:updateOutput(input)
end

function FireModule:updateGradInput(input, gradOutput)
   self.gradInput = self.module:updateGradInput(input, gradOutput)
end

function FireModule:accGradParameters(input, gradOutput)
   self.module:accGradParameters(input, gradOutput)
end

function FireModule:__tostring__()
   return string.format('%s Squeeze: %f, Expand: %f %f, activation: %s',
                        torch.type(self), self.s1x1, self.e1x1, self.e1x3,
                        self.activation)
end
