local Container = nn.Container

function Container:extend(...)
   for i,module in ipairs{...} do
      self:add(module)
   end
   return self
end
