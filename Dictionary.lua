local Dictionary, parent = torch.class("nn.Dictionary", "nn.LookupTable")

-- don't use this with optim (useless), use nn.LookupTable instead
function Dictionary:__init(dictSize, embeddingSize, accUpdate)
   error"DEPRECATED Jan 14, 2016"
end
