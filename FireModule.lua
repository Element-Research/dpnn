--[[
  Fire module as explained in SqueezeNet http://arxiv.org/pdf/1602.07360v1.pdf.
--]]
--FIXME works only for batches.

local FireModule, Parent = torch.class('nn.FireModule', 'nn.Decorator')

function FireModule:__init(nInputPlane, s1x1, e1x1, e3x3, activation)
   self.nInputPlane = nInputPlane
   self.s1x1 = s1x1
   self.e1x1 = e1x1
   self.e3x3 = e3x3
   self.activation = activation or 'ReLU'

   if self.s1x1 > (self.e1x1 + self.e3x3) then
      print('Warning: <FireModule> s1x1 is recommended to be smaller'..
            ' then e1x1+e3x3')
   end
   
   self.module = nn.Sequential()
   self.squeeze = nn.SpatialConvolution(nInputPlane, s1x1, 1, 1)
   self.expand = nn.Concat(2)
   self.expand:add(nn.SpatialConvolution(s1x1, e1x1, 1, 1))
   self.expand:add(nn.SpatialConvolution(s1x1, e3x3, 3, 3, 1, 1, 1, 1))

   -- Fire Module
   self.module:add(self.squeeze)
   self.module:add(nn[self.activation]())
   self.module:add(self.expand)
   self.module:add(nn[self.activation]())
   
   Parent.__init(self, self.module)
end

--[[
function FireModule:type(type, tensorCache)
   assert(type, 'Module: must provide a type to convert to')
   self.module = nn.utils.recursiveType(self.module, type, tensorCache)
end
--]]

function FireModule:__tostring__()
   return string.format('%s inputPlanes: %d -> Squeeze Planes: %d -> '..
                        'Expand: %d(1x1) + %d(3x3), activation: %s',
                        torch.type(self), self.nInputPlane, self.s1x1,
                        self.e1x1, self.e3x3, self.activation)
end
