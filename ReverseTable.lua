local ReverseTable, parent = torch.class("nn.ReverseTable", "nn.Module")

function ReverseTable:__init()
   parent.__init(self)
   self.output = {}
   self.gradInput = {}
end

function ReverseTable:updateOutput(inputTable)
   assert(torch.type(inputTable) == 'table', "Expecting table at arg 1")
   
   -- empty output table
   for k,v in ipairs(self.output) do
      self.output[k] = nil
   end
   
   -- reverse input
   local k = 1
   for i=#inputTable,1,-1 do
      self.output[k] = inputTable[i]
      k = k + 1
   end
   return self.output
end

function ReverseTable:updateGradInput(inputTable, gradOutputTable)
   -- empty gradInput table
   for k,v in ipairs(self.gradInput) do
      self.gradInput[k] = nil
   end
   
   -- reverse gradOutput
   local k = 1
   for i=#gradOutputTable,1,-1 do
      self.gradInput[k] = gradOutputTable[i]
      k = k + 1
   end
   return self.gradInput
end
