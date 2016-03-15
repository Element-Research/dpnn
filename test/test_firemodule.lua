require 'nn'
require 'dpnn'
require 'cunn'
require 'cutorch'

--torch.setdefaulttensortype('torch.FloatTensor')

-- FireModule issue 45
--[[
m = nn.Sequential()
m:add(nn.FireModule(1,1,1,1))
_, p = m:getParameters()
print(p:sum())

m = m:cuda()
_, p = m:getParameters()
print(p:sum())

m:zeroGradParameters()
print(p:sum())--]]


-- Testing FireModule
input = torch.rand(1, 3, 6, 6)
model = nn.FireModule(3, 1, 1, 1, 'Tanh')
print(model)
print(model.module)
parameters, gradParameters = model:getParameters()
output = model:forward(input)
grads = torch.rand(output:size())
gi = model:backward(input, grads)
print(gi:mean(), gi:std(), gi:min(), gi:max())

cutorch.setDevice(1)
model:cuda()
print(model.module.modules[1].finput)
cinput = input:cuda()
output = model:forward(cinput)
gi = model:backward(input:cuda(), grads:cuda())
print(gi:mean(), gi:std(), gi:min(), gi:max())
