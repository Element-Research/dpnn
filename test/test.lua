
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

function dpnntest.Serial()
   local mlp = nn.Linear(3,7)
   mlp:forward(torch.randn(4,3))
   mlp:backward(torch.randn(4,3), torch.randn(4,7))
   local mlp2 = nn.Serial(mlp)
   mlp2:mediumSerial()
   local mlp3:clone()
   mytester:assert(mlp3.module.output:nElement() == 0, "serial medium empty err")
   mytester:assert(torch.type(mlp3.module.output) == 'torch.FloatTensor', "serial medium type err")
end

function dpnntest.Convert()
   -- batch mode
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
   -- non-batch mode
   local c = nn.Convert('chw', 'hwc')
   local input = torch.randn(3,5,5)
   local output = c:forward(input)
   local output2 = input:transpose(1,3):transpose(1,2)
   mytester:assertTensorEq(output, output2, 0.000001, "Convert fwd chw->hwc non-batch")
   local gradInput = c:backward(input, output)
   mytester:assertTensorEq(gradInput, input, 0.000001, "Convert bwd chw->hwc non-batch")
   local c = nn.Convert('chw', 'f')
   local output = c:forward(input)
   local output2 = input:view(-1)
   mytester:assertTensorEq(output, output2, 0.000001, "Convert fwd chw->bf non-batch")
   c:float()
   local output = c:forward(input:float())
   mytester:assertTensorEq(output, output2:float(), 0.000001, "Convert:type() non-batch")
   local output = c:forward(input)
   mytester:assertTensorEq(output, output2:float(), 0.000001, "Convert:type() double->float non-batch")
end

function dpnntest.Collapse()
   local c = nn.Collapse(3)
   local input = torch.randn(8,3,4,5)
   local output = c:forward(input)
   mytester:assertTensorEq(input:view(8,-1), output, 0.000001, "Collapse:forward")
   local gradInput = c:backward(input, output)
   mytester:assertTensorEq(gradInput, input, 0.000001, "Collapse:backward")
   mytester:assertTableEq(gradInput:size():totable(), input:size():totable(), 0.000001, "Collapse:backward size")
   local input2 = input:transpose(1,4)
   local output2 = c:forward(input2)
   mytester:assertTensorEq(input2, output2, 0.000001, "Collapse:forward non-contiguous")
   local gradInput2 = c:backward(input2, output2)
   mytester:assertTensorEq(gradInput2, input2, 0.000001, "Collapse:backward non-contiguous")
   mytester:assertTableEq(gradInput2:size():totable(), input2:size():totable(), 0.000001, "Collapse:backward size non-contiguous")
end

function dpnntest.ZipTable()
   -- input : { {a1,a2}, {b1,b2}, {c1,c2} }
   -- output : { {a1,b1,c1}, {a2,b2,c2} }
   local z = nn.ZipTable()
   local input = {
      {torch.randn(3,4), torch.randn(3,4)},
      {torch.randn(3,4), torch.randn(3,4)},
      {torch.randn(3,4), torch.randn(3,4)}
   }
   local output = z:forward(input)
   mytester:assert(#output == 2, "ZipTable #output")
   mytester:assert(#(output[1]) == 3, "ZipTable #output[1]")
   mytester:assertTensorEq(input[1][1], output[1][1], 0.000001, "ZipTable input11")
   mytester:assertTensorEq(input[1][2], output[2][1], 0.000001, "ZipTable input12")
   mytester:assertTensorEq(input[3][2], output[2][3], 0.000001, "ZipTable input32")
   local gradInput = z:backward(input, output)
   mytester:assert(#gradInput == 3, "ZipTable #gradInput")
   mytester:assert(#(gradInput[1]) == 2, "ZipTable #gradInput[1]")
   mytester:assertTensorEq(input[1][1], gradInput[1][1], 0.000001, "ZipTable gradInput11")
   mytester:assertTensorEq(input[1][2], gradInput[1][2], 0.000001, "ZipTable gradInput12")
   mytester:assertTensorEq(input[3][2], gradInput[3][2], 0.000001, "ZipTable gradInput32")
end

function dpnntest.Inception()
   local size = {8,3,32,32}
   local outputSize = {8,16+24+8+12,32,32}
   local input = torch.rand(unpack(size))
   local gradOutput = torch.randn(unpack(outputSize))
   local incep = nn.Inception{inputSize=3, outputSize={16,24}, reduceSize={14,16,8,12}}
   for i, param in ipairs(incep:parameters()) do
      mytester:assert(_.isFinite(param:sum()), 'inception init error')
   end
   local output = incep:forward(input)
   mytester:assertTableEq(output:size():totable(), outputSize, 0.00001)
   mytester:assert(_.isFinite(output:sum()))
   incep:zeroGradParameters()
   local gradInput = incep:backward(input, gradOutput)
   mytester:assertTableEq(gradInput:size():totable(), size, 0.00001)
   mytester:assert(_.isFinite(gradInput:sum()))
   incep:updateParameters(0.1)
   for i, param in ipairs(incep:parameters()) do
      mytester:assert(_.isFinite(param:sum()), 'inception update error')
   end
   incep:maxParamNorm(1)
   for i, param in ipairs(incep:parameters()) do
      mytester:assert(_.isFinite(param:sum()), 'inception maxNorm error')
   end
end

function dpnntest.ModuleCriterion()
   local input = torch.randn(8,4)
   local target = torch.randn(8,4)
   local inputModule = nn.Tanh()
   local criterion = nn.MSECriterion()
   local mc = nn.ModuleCriterion(criterion, inputModule)
   
   local err = mc:forward(input, target)
   local gradInput = mc:backward(input, target)
   
   local output = inputModule:forward(input)
   local err2 = criterion:forward(output, target)
   local gradOutput = criterion:backward(output, target)
   local gradInput2 = inputModule:backward(input, gradOutput)

   mytester:assert(err == err2, "ModuleCriterion backward err")
   mytester:assertTensorEq(gradInput, gradInput2, 0.000001, "ModuleCriterion backward err")
end

function dpnn.test(tests)
   mytester = torch.Tester()
   mytester:add(dpnntest)
   math.randomseed(os.time())
   mytester:run(tests)
end
