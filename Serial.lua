------------------------------------------------------------------------
--[[ Serial ]]--
-- Decorator that modifies the serialization/deserialization 
-- behaviour of encapsulated module.
------------------------------------------------------------------------
local Serial, parent = torch.class("nn.Serial", "nn.Decorator")

function Serial:write(file)
   local state = self:getSerialState()
   file:writeObject(state)
end

function Serial:read(file)
   local state = file:readObject()
   for k,v in pairs(state) do
      self[k] = v
   end
   local function recursiveSetMetatable(state)
      if torch.type(state) == 'table' and state.dpnn_typename then
         torch.setmetatable(state, state.dpnn_typename)
      elseif torch.isTypeOf(state, 'nn.Module') then
         for k,v in pairs(state) do
            recursiveSetMetatable(v)
         end
      end
   end
   recursiveSetMetatable(self)
end


