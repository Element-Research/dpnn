
local dpnntest = {}
local precision = 1e-5
local mytester

function dpnntest.Module_sharedClone()

   local function test(mlp, name)
      mlp:zeroGradParameters()
      local clone = mlp:clone()
      clone:share(mlp,"weight","bias","gradWeight","gradBias")
      
      local mlp2 = mlp:clone() -- not shared with mlp
      local clone2 = mlp2:sharedClone(true, true)
      
      local input = torch.randn(2,3)
      local gradOutput = torch.randn(2,4)
      
      local output = mlp:forward(input)
      local gradInput = mlp:backward(input, gradOutput)
      local output4 = clone:forward(input)
      local gradInput4 = clone:backward(input, gradOutput)
      
      mytester:assertTensorEq(output, output4, 0.00001, name.." updateOutput")
      mytester:assertTensorEq(gradInput, gradInput4, 0.00001, name.." updateGradInput")
      
      local output2 = clone2:forward(input)
      local gradInput2 = clone2:backward(input, gradOutput)
      
      mytester:assertTensorEq(output, output2, 0.00001, name.." updateOutput")
      mytester:assertTensorEq(gradInput, gradInput2, 0.00001, name.." updateGradInput")
      
      local output3 = mlp2:forward(input)
      local gradInput3 = mlp2:backward(input, gradOutput)
      
      mlp:updateParameters(0.1)
      mlp2:updateParameters(0.1)
      
      mytester:assertTensorEq(output3, output2, 0.00001, name.." updateOutput")
      mytester:assertTensorEq(gradInput3, gradInput2, 0.00001, name.." updateGradInput")
      
      local params, gradParams = mlp:parameters()
      local params4, gradParams4 = clone:parameters()
      local params2, gradParams2 = clone2:parameters()
      local params3, gradParams3 = mlp2:parameters()
      
      mytester:assert(#params == #params2, name.." num params err")
      mytester:assert(#params3 == #params2, name.." num params err")
      mytester:assert(#gradParams == #gradParams2, name.." num gradParams err")
      mytester:assert(#gradParams == #gradParams3, name.." num gradParams err")
      
      for i,param in ipairs(params) do
         mytester:assertTensorEq(param, params2[i], 0.00001, name.." params2 err "..i)
         mytester:assertTensorEq(param, params4[i], 0.00001, name.." params4 err "..i)
         mytester:assertTensorEq(param, params3[i], 0.00001, name.." params3 err "..i)
         mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.00001, name.." gradParams2 err "..i)
         mytester:assertTensorEq(gradParams[i], gradParams4[i], 0.00001, name.." gradParams4 err "..i)
         mytester:assertTensorEq(gradParams[i], gradParams3[i], 0.00001, name.." gradParams3 err "..i)
      end
   end
   
   test(nn.Linear(3,4), 'linear')
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(3,7))
   mlp:add(nn.Tanh())
   mlp:add(nn.Euclidean(7,4))
   mlp:add(nn.LogSoftMax())
   test(mlp, 'sequential')
end

function dpnntest.Module_sharedType()
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(3,7))
   mlp:add(nn.Tanh())
   mlp:add(nn.Euclidean(7,4))
   mlp:add(nn.LogSoftMax())
   mlp:zeroGradParameters()
   local mlp2 = mlp:sharedClone()
   local concat = nn.ConcatTable()
   concat:add(mlp):add(mlp2)
   
   concat:float(true) -- i.e. sharedType('torch.FloatTensor')
   
   local input = torch.randn(2,3):float()
   local gradOutput = torch.randn(2,4):float()
   
   local output = mlp:forward(input)
   local gradInput = mlp:backwardUpdate(input, gradOutput, 0.1)
   
   local params, gradParams = mlp:parameters()
   local params2, gradParams2 = mlp2:parameters()
   
   mytester:assert(#params == #params2, "num params err")
   mytester:assert(#gradParams == #gradParams2, "num gradParams err")
   
   for i,param in ipairs(params) do
      mytester:assertTensorEq(param, params2[i], 0.00001, " params err "..i)
      mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.00001, " gradParams err "..i)
   end
end

function dpnntest.Module_serial()
   local mlp = nn.Linear(3,7)
   mlp:forward(torch.randn(4,3))
   mlp:backward(torch.randn(4,3), torch.randn(4,7))
   mlp:serial()
   local mlp2 = mlp:clone()
   mytester:assert(torch.type(mlp2.output) == torch.type(mlp.output), "serial clone err")
   mytester:assertTensorEq(mlp2.output, mlp.output, 0.000001, "serial clone err")
   local mlp3 = torch.deserialize(torch.serialize(mlp))
   mytester:assert(mlp3.output:nElement() == 0, "serial light size err")
   mytester:assert(torch.type(mlp3.output) == 'torch.FloatTensor', "serial light type err")
end

function dpnntest.Convert()
   local c = nn.Convert('bchw', 'chwb')
   local input = torch.randn(8,3,5,5)
   local output = c:forward(input)
   local output2 = input:transpose(1,4):transpose(1,3):transpose(1,2)
   mytester:assertTensorEq(output, output2, 0.000001, "Convert fwd bchw->chwb")
   local gradInput = c:backward(input, output)
   mytester:assertTensorEq(gradInput, input, 0.000001, "Convert bwd bchw->chwb")
   local c = nn.Convert('bchw', 'bf')
   local output = c:forward(input)
   local output2 = input:view(8,-1)
   mytester:assertTensorEq(output, output2, 0.000001, "Convert fwd bchw->bf")
   c:float()
   local output = c:forward(input:float())
   mytester:assertTensorEq(output, output2:float(), 0.000001, "Convert:type()")
   local output = c:forward(input)
   mytester:assertTensorEq(output, output2:float(), 0.000001, "Convert:type() double->float")
end

function dpnn.test(tests)
   mytester = torch.Tester()
   mytester:add(dpnntest)
   math.randomseed(os.time())
   mytester:run(tests)
end
