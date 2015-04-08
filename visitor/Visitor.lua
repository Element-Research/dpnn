------------------------------------------------------------------------
--[[ Visitor ]]--
-- Visits a composite struture of Modules and modifies their states.
-- Visitors should try to access a module method assigned to 
-- each visitor (if itexists). This allows modules to implement
-- visitor specifics.
------------------------------------------------------------------------
local Visitor = torch.class("nn.Visitor")
Visitor.isVisitor = true

function Visitor:__init(include, exclude, verbose)
   self.include = include
   self.exclude = exclude or {'modules'}
   self.verbose = (verbose == nil) and true or verbose
   self.name = torch.type(self)
end

-- compares model to filter to see if it can be visited
function Visitor:canVisit(module)
   if self.exclude then
      for i, tag in ipairs(self.exclude) do
         if module[tag] then
            return false
         end
      end
   end
   if self.include then
      for i, tag in ipairs(self.include) do
         if module[tag] then
            return true
         end
      end
      return false
   end
   return true
end

function Visitor:visit(module)
   -- each module will host a model-visitor state
   local mvstate = module.mvstate
   if not mvstate then
      mvstate = {}
      module.mvstate = mvstate
   end
   
   local mself = mvstate[self.name]
   if not mself then 
      mself = {canVisit = self:canVisit(module)}
      mvstate[self.name] = mself
      
   end
   
   -- can we visit model?
   if not mself.canVisit then 
      return 
   end
   
   self:visitModule(module)
end

function Visitor:visitModule(model)
   return
end
