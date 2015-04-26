local DontCast, parent = torch.class("nn.DontCast", "nn.Decorator")

-- dont cast
function DontCast:type(type)
   return self
end
