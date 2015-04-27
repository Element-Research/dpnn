------------------------------------------------------------------------
--[[ Serial ]]--
-- Decorator that modifies the serialization/deserialization 
-- behaviour of encapsulated module.
------------------------------------------------------------------------
local Serial, parent = torch.class("nn.Serial", "nn.Decorator")

function Serial:write(file)
   local state = self:getSerialState()
   
   local function recursiveType(state)
      for k,v in pairs(state) do
         if torch.type(v) == 'table' then
            recursiveType(v)
         end
      end
      
      if state.dpnn_serialType then
         torch.setmetatable(state, state.dpnn_typename)
         assert(state.dpnn_typename)
         assert(torch.isTypeOf(state, 'nn.Module'))
         local type = state.dpnn_serialType
         if type:find('torch') then
            state:type(type)
         else
            state[type](state)
         end
      end
   end
   
   -- cast to type before serialization (useful for cuda)
   recursiveType(state)
   
   -- removes self's metatable
   state = _.map(state, function(k,v) return v end)
   
   file:writeObject(state)
end

function Serial:read(file)
   local state = file:readObject()
   for k,v in pairs(state) do
      self[k] = v
   end
end


