------------------------------------------------------------------------
--[[ Serial ]]--
-- Decorator that modifies the serialization/deserialization 
-- behaviour of encapsulated module.
------------------------------------------------------------------------
local Serial, parent = torch.class("nn.Serial", "nn.Decorator")

function Serial:__init(module, tensortype)
   parent.__init(self, module)
   self.tensortype = tensortype
   if self.tensortype then
      assert(tensortype:find('torch.*Tensor'), "Expecting tensortype (e.g. torch.LongTensor) at arg1")
   end
end

function Serial:write(file)
   local state = self:getSerialState()
   
   local function recursiveSetMetaTable(state)
      for k,v in pairs(state) do
         if torch.type(v) == 'table' then
            recursiveSetMetaTable(v)
         end
      end
      
      if state.dpnn_typename then
         torch.setmetatable(state, state.dpnn_typename)
      end
   end
   
   -- typecast before serialization (useful for cuda)
   recursiveSetMetaTable(state)
   
   if self.tensortype then
      state:type(self.tensortype)
   end
   
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


