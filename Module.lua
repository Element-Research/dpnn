local Module = nn.Module

function Module:sparseParameters()
   return self:parameters()
end

function Module:updateParameters(learningRate)
   -- sparse params can have different learningRate scales per param
   local params, gradParams, scales = self:sparseParameters()
   if params then
      for i,param in pairs(params) do -- pairs for sparse params
         local scale = scales and scales[i] or 1
         param:add(-learningRate*scale, gradParams[i])
      end
   end
end

function Module:zeroGradParameters()
   local _,gradParams = self:sparseParameters()
   if gradParams then
      for i,gradParam in pairs(gradParams) do -- pairs for sparse params
         gradParam:zero()
      end
   end
end

------------------------ clone and type --------------------------------

Module.dpnn_parameters = {'weight', 'bias'}
Module.dpnn_gradParameters = {'gradWeight', 'gradBias'}

-- Note : this method expects component modules to be stored in either
-- self.modules, self.sharedClones or as an attribute to self.
-- So if you store a module in self.mytbl = {mymodule}, it will be cloned
-- independently of sharedClone (i.e. deep copy).
function Module:sharedClone(shareParams, shareGradParams, clones, pointers, stepClone)   
   shareParams = (shareParams == nil) and true or shareParams
   shareGradParams = (shareGradParams == nil) and true or shareGradParams
   
   clones = clones or {} -- module clones (clone once)
   pointers = pointers or {} -- parameters (dont clone params/gradParams)
   
   -- 1. remove all the component modules
   -- containers keep modules in self.modules table
   local moduleClones, modules
   if self.modules then
      moduleClones = {}
      for i,module in ipairs(self.modules) do
         local clone
         if not clones[torch.pointer(module)] then
            clone = module:sharedClone(shareParams, shareGradParams, clones, pointers, stepClone)
            clones[torch.pointer(module)] = clone
         else
            clone = clones[torch.pointer(module)]
         end
         moduleClones[i] = clone
      end
      modules = self.modules
      self.modules = nil -- to prevent recloning
   end
   
   -- rnns keep modules in self.sharedClones table
   local sharedCloneClones, sharedClones
   if self.sharedClones then
      sharedCloneClones = {}
      for i,sharedClone in pairs(self.sharedClones) do
         local clone
         if not clones[torch.pointer(sharedClone)] then
            clone = sharedClone:sharedClone(shareParams, shareGradParams, clones, pointers, stepClone)
            clones[torch.pointer(sharedClone)] = clone
         else
            clone = clones[torch.pointer(sharedClone)]
         end
         sharedCloneClones[i] = clone
      end
      sharedClones = self.sharedClones
      self.sharedClones = nil -- to prevent recloning
   end
   
   -- some modules will keep modules as attributes (self.[key])
   local attributeClones, attributes = {}, {}
   for k,module in pairs(self) do
      if torch.isTypeOf(module,'nn.Module') then
         
         local clone
         if not clones[torch.pointer(module)] then
            clone = module:sharedClone(shareParams, shareGradParams, clones, pointers, stepClone)
            clones[torch.pointer(module)] = clone
         else
            clone = clones[torch.pointer(module)]
         end
         attributeClones[k] = clone
         attributes[k] = module
         self[k] = nil
      end
   end
   
   -- 2. remove the params, gradParams. Save for later.
   local params = {}
   if shareParams then
      for i,paramName in ipairs(self.dpnn_parameters) do
         local param = self[paramName]
         if param then
            params[paramName] = param
            self[paramName] = nil
            if param:storage() then
               pointers[torch.pointer(param:storage():data())] = true
            end
         end
      end
   end
   
   if shareGradParams then
      for i,paramName in ipairs(self.dpnn_gradParameters) do
         local gradParam = self[paramName]
         if gradParam then
            params[paramName] = gradParam
            self[paramName] = nil
            if gradParam:storage() then
               pointers[torch.pointer(gradParam:storage():data())] = true
            end
         end
      end
   end
   
   -- find all the tensors that share storage with the shared params
   for paramName, param in pairs(self) do
      if torch.isTensor(param) and param:storage() then
         if pointers[torch.pointer(param:storage():data())] then
            params[paramName] = param
            self[paramName] = nil
         end
      end
   end
   
   -- 3. clone everything but parameters, gradients and modules (removed above)
   local clone = self:clone()
   
   
   -- 4. add back to self/clone everything that was removed in steps 1 and 2
   for paramName, param in pairs(params) do
      assert(self[paramName] == nil)
      self[paramName] = param
      clone[paramName] = param.new():set(param)
   end
   
   if moduleClones then
      assert(self.modules == nil)
      self.modules = modules
      clone.modules = moduleClones
   end
   
   if sharedCloneClones then
      assert(self.sharedClones == nil)
      self.sharedClones = sharedClones
      clone.sharedClones = sharedCloneClones
   end
   
   for k,v in pairs(attributes) do
      self[k] = v
      clone[k] = attributeClones[k]
   end
   
   return clone
end      

-- by default, Module:type() will preserve shared Tensors.
-- Its more sensible this way, necessary for RNNs and fits 
-- in with existing overriden methods.
-- for preserving shared params created with sharedClones
function Module:type(type)
   assert(type, 'Module:type() must provide a type to convert to')
   -- key: pointer to old storage ; value : new storage
   local castmap = dpnn.castmap
   local root
   if not castmap then
      -- Contains torch.Storage instances used in Modules.
      -- The use of a global variable is ugly. But It is the only way 
      -- to fit in with existing overriden Module:type() methods.
      root = true
      dpnn.castmap = {}
      castmap = dpnn.castmap
   end
   
   local function recursiveType(param, type_str)
      if torch.type(param) == 'table' then
         for k,v in pairs(param) do
            param[k] = recursiveType(v, type_str)
         end
      elseif torch.isTypeOf(param, 'nn.Module') or torch.isTypeOf(param, 'nn.Criterion') then
         param:type(type_str)
      else
         if torch.isTensor(param) then
            if param:storage() then
               local pointer = torch.pointer(param:storage():data())
               local storage = castmap[pointer]
               -- empty storages (cuda) have zero pointers.
               -- we assume that these aren't shared.
               -- https://github.com/torch/cutorch/issues/147
               if pointer > 0 then 
                  if not storage then
                     local _param = param
                     -- cast entire storage
                     param = param.new(param:storage()):type(type_str)
                     if param:storage() then -- to handle cuda tensors ...
                        param:set(param:storage(), _param:storageOffset(), _param:size(), _param:stride())
                        castmap[pointer] = param:storage()
                        -- in case the module gets cast more than once:
                        castmap[torch.pointer(param:storage():data())] = param:storage()
                     end
                  else
                     -- set to point to existing storage
                     local _param = param
                     param = torch.getmetatable(type_str).new()
                     param:set(storage, _param:storageOffset(), _param:size(), _param:stride())
                  end
               else
                  assert(not storage)
                  param = param:type(type_str)
               end
            else
               param = param:type(type_str)
            end
         end
      end
      return param
   end
   
   -- find all tensors and convert them
   for key,param in pairs(self) do
      self[key] = recursiveType(param, type)
   end
   
   if root then
      -- reset the cast map
      dpnn.castmap = nil
   end
   return self
end

----------------- serialization (see nn.Serial) -------------------

Module.dpnn_mediumEmpty = {'output', 'gradInput', 'momGradParams', 'dpnn_input'}
Module.dpnn_lightEmpty = Module.dpnn_gradParameters
-- defaults to heavy serialization
Module.dpnn_serialEmpty = {}
Module.dpnn_serialType = false 

-- sets the serialization behavior of the entire module structure
function Module:serialMode(empty, type)
   assert(torch.type(empty) == 'table', "Expecting table at arg 1")
   self.dpnn_serialEmpty = empty
   self.dpnn_serialType = type
   -- set the serial of all encapsulated modules
   local function recursiveSerial(tbl)
      for k,v in pairs(tbl) do
         if torch.isTypeOf(v, 'nn.Module') then
            v:serialMode(empty, type)
         elseif torch.type(v) == 'table' then
            recursiveSerial(v)
         end
      end
   end
   recursiveSerial(self)
   return self
end

-- serialMode : serialize everything
function Module:heavySerial(type)
   return self:serialMode({}, type)
end

-- serialMode : serialize everything except dpnn_mediumEmpty attributes
function Module:mediumSerial(type)
   
   self.dpnn_serialEmpty = self.dpnn_mediumEmpty
   self.dpnn_serialType = (type == nil) and 'float' or type
   
   -- set the serial of all encapsulated modules
   local function recursiveSerial(tbl)
      for k,v in pairs(tbl) do
         if torch.isTypeOf(v, 'nn.Module') then
            v:mediumSerial(type)
         elseif torch.type(v) == 'table' then
            recursiveSerial(v)
         end
      end
   end
   recursiveSerial(self)
   return self
end

-- serialMode : serialize everything except dpnn_mediumEmpty and dpnn_lightEmpty attributes
function Module:lightSerial(type)
   
   self.dpnn_serialEmpty = _.clone(self.dpnn_mediumEmpty)
   for k,v in ipairs(self.dpnn_lightEmpty) do
      table.insert(self.dpnn_serialEmpty, v)
   end
   
   self.dpnn_serialType = (type == nil) and 'float' or type
   
   -- set the serial of all encapsulated modules
   local function recursiveSerial(tbl)
      for k,v in pairs(tbl) do
         if torch.isTypeOf(v, 'nn.Module') then
            v:lightSerial(type)
         elseif torch.type(v) == 'table' then
            recursiveSerial(v)
         end
      end
   end
   recursiveSerial(self)
   
   return self
end

function Module:getSerialState(states)
   states = states or {}
   
   -- dont get the serial state of the same module twice (reuse existing)
   if states[self] then
      return states[self]
   end
   
   -- returns the object structure as tables (i.e. without metatables)
   local function recursiveState(tbl)
      local state = _.map(tbl, 
         function(k,v) 
            if torch.isTypeOf(tbl, 'nn.Module') and _.contains(tbl.dpnn_serialEmpty, k) then 
               -- "empties" module attributes found in empty
               if torch.type(v) == 'table' then
                  -- empty table
                  return {} 
               elseif torch.isTensor(v) then
                  -- empty tensor
                  return v.new() 
               else
                  -- not table nor tensor? then serialize as is
                  return v
               end
            elseif torch.isTypeOf(v, 'nn.Module') then
               -- recursive, yet can be overwritten
               return v:getSerialState(states)
            elseif torch.type(v) == 'table' then
               -- in case it is a table of modules
               if not states[v] then
                  states[v] = recursiveState(v)
               end
               return states[v]
            else
               return v
            end
         end
      )
      return state
   end
   
   local state = recursiveState(self)
   
   -- include typename so that module can be reconstructed from the state
   state.dpnn_typename = torch.type(self)
   states[self] = state
   
   return state
end

-- decorates self with nn.Serial
function Module:Serial()
   return nn.Serial(self)
end

----------------------- for training -----------------------------

-- useful to get the output size
-- I chose this method name because it is less likely to be overriden.
function Module:outside(insize)
   local input
   if torch.type(insize) == 'table' then
      input = torch.randn(table.unpack(insize))
   else
      input = torch.randn(insize)
   end
   local output = self:updateOutput(input)
   return output:size()
end

-- for those interested in implementing the visitor design pattern
function Module:accept(visitor)
   visitor:visit(self)
end

-- Can be used as a regularizer instead of weight decay
-- Assumes that parameters are arranged (output dim x ... x input dim)
function Module:maxParamNorm(maxOutNorm, maxInNorm)
   -- this allows each module to set its own max[Out,In]Norm
   maxOutNorm = self.maxOutNorm or maxOutNorm
   maxInNorm = self.maxInNorm or maxInNorm
   if not (maxOutNorm or maxInNorm) then
      return
   end
   
   if self.modules then
      for i,module in ipairs(self.modules) do
         module:maxParamNorm(maxOutNorm, maxInNorm)
      end
   else
      local params = self:parameters() 
      if not params or gradParams then
         return
      end
      for k,param in pairs(params) do -- pairs for sparse params
         -- By default, only affects non-1D params.
         if param:dim() > 1 then
            if maxOutNorm and maxOutNorm > 0 then
               -- rows feed into output neurons 
               param:renorm(2, 1, maxOutNorm)
            end
            if maxInNorm and maxInNorm > 0 then
               -- cols feed out from input neurons
               param:renorm(2, param:dim(), maxInNorm)
            end
         end
      end
   end
end

-- Similar to maxParamNorm, but norm is global to Module for which 
-- this is called. Unless moduleLocal is true, in which case, the
-- norm constraint is applied to the norm of all parameters in each
-- component (non-container) module.
function Module:gradParamClip(cutoffNorm, moduleLocal)
   -- this allows each module to set its own cutoffNorm
   cutoffNorm = self.cutoffNorm or cutoffNorm
   if cutoffNorm <= 0 then
      return
   end
   if self.moduleLocal ~= nil then
      moduleLocal = self.moduleLocal
   end
   
   local norm = 0
   if moduleLocal and self.modules then
      for i,module in ipairs(self.modules) do
         norm = norm + math.pow(module:gradParamClip(maxOutNorm, maxInNorm), 2)
      end
      norm = math.sqrt(norm)
   else
      local params, gradParams = self:parameters()
      if not (params and gradParams) then
         return norm
      end
      for k,gradParam in pairs(gradParams) do -- pairs for sparse params
         norm = norm + math.pow(gradParam:norm(),2)
      end
      norm = math.sqrt(norm)
      if norm > cutoffNorm then
         -- rescale gradParams to obtain desired cutoffNorm
         for k,gradParam in pairs(gradParams) do
            gradParam:mul(cutoffNorm/norm)
         end
      end
   end
   return norm
end

-- Adds weight decay constraint on params with dims > 2 (default).
-- TODO : allow inplace weightDecay (before calling accUpdateGradParameters)
function Module:weightDecay(wdFactor, wdMinDim)
   -- this allows each module to set its own hyper-parameters
   wdFactor = self.wdFactor or wdFactor
   if wdFactor <= 0 then
      return
   end
   wdMinDim = self.wdMinDim or wdMinDim or 2
   
   if self.modules then
      for i,module in ipairs(self.modules) do
         module:weightDecay(wdFactor, wdMinDim)
      end
   else
      local params, gradParams = self:parameters()
      if not (params and gradParams) then
         return
      end
      
      for i,param in pairs(params) do -- pairs for sparse params
         if param:dim() >= wdMinDim then
            gradParams[i]:add(wdFactor, param)
         end
      end
   end
end

function Module:momentumGradParameters()
   if (not self.momGradParams) or _.isEmpty(self.momGradParams) then
      local params, gradParams = self:parameters()
      if not gradParams or _.isEmpty(gradParams) then
         return
      end
      self.momGradParams = {}
      for i,gradParam in pairs(gradParams) do 
         self.momGradParams[i] = gradParam.new():resizeAs(gradParam):copy(gradParam)
      end
   end
   return self.momGradParams
end

-- uses momentum learning to update gradParams
function Module:updateGradParameters(momFactor, momDamp, momNesterov)
   -- this allows each module to set its own hyper-parameters
   momFactor = self.momFactor or momFactor
   if momFactor <= 0 then
      return
   end
   momDamp = self.momDamp or momDamp or momFactor
   if self.momNesterov ~= nil then
      momNesterov = self.momNesterov
   end
   
   if self.modules then
      for i,module in ipairs(self.modules) do
         module:updateGradParameters(momFactor, momDamp, momNesterov)
      end
   else
      local params, gradParams = self:parameters()
      if (not params) or _.isEmpty(params) then
         return
      end
      local momGradParams = self:momentumGradParameters()
      for i,gradParam in pairs(gradParams) do
         momGradParams[i]:mul(momFactor):add(1-momDamp, gradParam)
      end
      
      if momNesterov then
         for i,gradParam in pairs(gradParams) do
            gradParam:add(momFactor, momGradParams[i])
         end
      else
         for i,gradParam in pairs(gradParams) do
            gradParam:copy(momGradParams[i])
         end
      end
   end
end

function Module:checkParameters()
   local params = self:parameters() or {}
   for k,param in pairs(params) do
      if _.isNaN(param:sum()) then
         error("NaN Error for param at index" ..k)
      end
   end
end

function Module:dontBackward()
   self.updateGradInput = function() end
   self.accGradParameters = function() end
   self.accUpdateGradParameters = function() end
   return self
end

function Module:contiguousInput(input, backward)
   if backward then
      return self.dpnn_cinput or input
   end
   if not input:isContiguous() then
      self.dpnn_cinput = self.dpnn_cinput or input.new()
      self.dpnn_cinput:resizeAs(input):copy(input)
      input = self.dpnn_cinput
   end
   return input
end

function Module:toBatch(tensor, nDim, batchDim)
   local batchDim = batchDim or 1
   if tensor:dim() == nDim then
      self.dpnn_online = true
      local size = tensor:size():totable()
      table.insert(size, batchDim, 1)
      tensor = tensor:view(table.unpack(size))
   else
      self.dpnn_online = false
   end
   return tensor
end

function Module:fromBatch(tensor, batchDim)
   if self.dpnn_online then
      local size = tensor:size():totable()
      assert(table.remove(size, batchDim) == 1)
      tensor = tensor:view(table.unpack(size))
   end
   return tensor
end

function Module:extrapolateType()
   local params = module:parameters()
   if params then
      -- extrapolate the tensor type of the module
      local types = {}
      for i, param in ipairs(params) do
         local tensorType = torch.type(param)
         types[tensorType] = (types[tensorType] or 0) + 1
      end
      local maxCount = 0
      local maxType
      for tensorType, count in pairs(types) do
         if count > maxCount then
            maxtype = tensorType
            maxCount = count
         end
      end
      return maxType
   end
   return nil --unknown otherwise
end

function Module:profile()
   if self.modules then
      for i, module in ipairs(self.modules) do
         module:profile()
      end
   end
   self.dpnn_profile = true
end

function Module:reinforce(reward)
   if self.modules then
      for i, module in ipairs(self.modules) do
         module:reinforce(reward)
      end
   end
end
