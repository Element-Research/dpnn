local Module = nn.Module

function Module:updateParameters(learningRate)
   -- sparse params can have different learningRate scales per param
   local params, gradParams, scales = self:parameters()
   if params then
      for i,param in pairs(params) do -- pairs for sparse params
         local scale = scales and scales[i] or 1
         param:add(-learningRate*scale, gradParams[i])
      end
   end
end

function Module:zeroGradParameters()
   local _,gradParams = self:parameters()
   if gradParams then
      for i,gradParam in pairs(gradParams) do -- pairs for sparse params
         gradParam:zero()
      end
   end
end

Module.__parameters__ = {'weight', 'bias'}
Module.__gradParameters__ = {'gradWeight', 'gradBias'}

-- TODO make this recursive (for table params)
function Module:sharedClone(shareParams, shareGradParams)
   shareParams = (shareParams == nil) and true or shareParams
   shareGradParams = (shareGradParams == nil) and true or shareGradParams
   
   local moduleClones, modules
   if self.modules then
      moduleClones = {}
      for i,module in ipairs(self.modules) do
         moduleClones[i] = module:sharedClone(shareParams, shareGradParams)
      end
      modules = self.modules
      self.modules = nil -- to prevent recloning
   end
   
   local params, pointers = {}, {}
   if shareParams then
      for i,paramName in ipairs(self.__parameters__) do
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
      for i,paramName in ipairs(self.__gradParameters__) do
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
   
   -- clone everything but parameters and/or gradients
   local clone = self:clone()
   
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
   return clone
end      

-- for preserving shared params created with sharedClones
function Module:sharedType(type, castmap)
   assert(type, 'Module:sharedType must provide a type to convert to')
   -- key: pointer to old storage 
   -- value : new storage
   castmap = castmap or {} --contains torch.Storage instances
   
   local function recursiveType(param, type_str)
      if torch.type(param) == 'table' then
         for i = 1, #param do
            param[i] = recursiveType(param[i], type_str)
         end
      else
         if torch.isTensor(param) then
            if param:storage() then
               local pointer = torch.pointer(param:storage():data())
               local storage = castmap[pointer]
               if not storage then
                  local _param = param
                  -- cast entire storage
                  param = param.new(param:storage()):type(type_str)
                  param:set(param:storage(), _param:storageOffset(), _param:size(), _param:stride())
                  castmap[pointer] = param:storage()
               else
                  -- set to point to existing storage
                  local _param = param
                  param = torch.getmetatable(type_str).new()
                  param:set(storage, _param:storageOffset(), _param:size(), _param:stride())
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
      -- Many modules (like CDivTable) have output or gradInput fields which
      -- are table's of tensors.  To be general we need to recursively
      -- cast fields that may be nested tables.
      if key ~= 'modules' then
        self[key] = recursiveType(self[key], type)
      end
   end
   -- find submodules in classic containers 'modules'
   if self.modules then
      for _,module in ipairs(self.modules) do
         module:sharedType(type, castmap)
      end
   end
   return self
end

-- by default, uses sharedType. Why? more sensible way, 
-- necessary for RNNs and fits in with existing overwritten methods 
function Module:type(type, shared)
   shared = (shared == nil) and true or shared
   return self:sharedType(type)
end

function Module:float(shared)
   return self:type('torch.FloatTensor', shared)
end

function Module:double(shared)
   return self:type('torch.DoubleTensor', shared)
end

function Module:cuda(shared)
   return self:type('torch.CudaTensor', shared)
end

function Module:int(shared)
   return self:type('torch.IntTensor', shared)
end

function Module:long(shared)
   return self:type('torch.LongTensor', shared)
end

-- useful to get the output size
-- I chose this method name because it is less likely to be overriden.
function Module:outside(insize)
   local input
   if torch.type(insize) == 'table' then
      input = torch.randn(unpack(insize))
   else
      input = torch.randn(insize)
   end
   local output = self:updateOutput(input)
   return output:size()
end

Module.__dontSerialize__ = {'output', 'gradInput', 'momGradParams', 'dpnn_input'}
Module.__serialMode = 'heavy'
Module.__serialType = false 

function Module:serial(mode, type)
   self.__serialMode = mode or 'light'
   self.__serialType = (type == nil) and 'float' or type
   return self
end

function Module:write(file)
   local state
   if self.__serialMode == 'light' then
      state = _.map(self, 
         function(k,v) 
            -- light mode ignores gradInputs and outputs by default
            if _.contains(self.__dontSerialize__, k) then 
               if torch.type(v) == 'table' then
                  return {}
               elseif torch.isTensor(v) then
                  return v.new()
               else
                  -- not table nor tensor? serialize as is
                  return v
               end
            else
               return v
            end
         end)
   else
      -- otherwise, serialize everything (default Module behavior)
      state = _.map(self, function(k,v) return v end)
   end

   if self.__serialType then
      -- cast to type before serialization (useful for cuda)
      torch.setmetatable(state, torch.type(self))
      local type = self.__serialType
      if type:find('torch') then
         state:type(type)
      else
         state[type](state)
      end
      -- remove metatable (I don't know any better way)
      state = _.map(state, function(k,v) return v end)
   end
   file:writeObject(state)
end

function Module:read(file)
   local state = file:readObject()
   for k,v in pairs(state) do
      self[k] = v
   end
end

local __clone__ = Module.clone

function Module:clone(...)
   local serialMode = self.__serialMode
   local serialType = self.__serialType
   self.__serialMode = Module.__serialMode
   self.__serialType = Module.__serialType
   
   -- call the original Module:clone() method
   -- Note that subclasses that override clone() without calling parent.clone will fail
   local clone = __clone__(self,...)
   clone:serial(serialMode, serialType)
   
   self:serial(serialMode, serialType)
   return clone
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
      local params = self:parameters() or {}, {}
      for k,param in pairs(params) do -- pairs for sparse params
         -- By default, only affects non-1D params.
         if param:dim() > 1 then
            if maxOutNorm then
               -- rows feed into output neurons 
               param:renorm(2, 1, maxOutNorm)
            end
            if maxInNorm then
               -- cols feed out from input neurons
               param:renorm(2, param:dim(), maxInNorm)
            end
         end
      end
   end
end

-- Similar to maxParamNorm, but norm is global to Module for which 
-- this is called. Unless moduleLocal is true, in which case, the
-- norm is constraint is applied to the norm of all parameters in each
-- component (non-container) module.
function Module:gradParamClip(cutoffNorm, moduleLocal)
   -- this allows each module to set its own cutoffNorm
   cutoffNorm = self.cutoffNorm or cutoffNorm
   if cutoffNorm <= 0 then
      return
   end
   if self.moduleLocal ~= nil then
      moduleLocal =  self.moduleLocal
   end
   
   if moduleLocal and self.modules then
      for i,module in ipairs(self.modules) do
         module:gradParamClip(maxOutNorm, maxInNorm)
      end
   else
      local params, gradParams = self:parameters() or {}, {}
      local norm = 0
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
      local params, gradParams = self:parameters() or {}, {}
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
   local params = self:parameters() or {}, {}
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
