
local dpnntest = {}
local dpnnbigtest = {}
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

function dpnntest.Module_type()
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(3,7))
   mlp:add(nn.Tanh())
   mlp:add(nn.Euclidean(7,4))
   mlp:add(nn.LogSoftMax())
   mlp:zeroGradParameters()
   local mlp2 = mlp:sharedClone()
   local concat = nn.ConcatTable()
   concat:add(mlp):add(mlp2)
   
   concat:float()
   
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
   
   if cutorch then
      require 'cunn'
      local input = torch.randn(3,32,32)
      local cnn = nn.Sequential()
      cnn:add(nn.SpatialConvolutionMM(3,8,5,5))
      cnn:add(nn.ReLU())
      cnn:add(nn.SpatialAveragePooling(2,2,2,2))
      cnn:add(nn.SpatialConvolutionMM(8,12,5,5))
      cnn:add(nn.ReLU())
      cnn:add(nn.SpatialAveragePooling(2,2,2,2))
      local outsize = cnn:outside{1,3,32,32}
      cnn:add(nn.Collapse(3))
      cnn:add(nn.Linear(outsize[2]*outsize[3]*outsize[4],20))
      cnn:add(nn.ReLU())
      cnn:add(nn.Linear(20,10))
      local output = cnn:forward(input):clone()
      local gradOutput = output:clone()
      local gradInput = cnn:backward(input, gradOutput):clone()
      cnn:float()
      local input3 = input:float()
      local output3 = cnn:forward(input3):clone()
      local gradOutput3 = output3:clone()
      local gradInput3 = cnn:backward(input3, gradOutput3):clone()
      mytester:assertTensorEq(output3:float(), output:float(), 0.000001, "type float fwd err")
      mytester:assertTensorEq(gradInput3:float(), gradInput:float(), 0.00001, "type float bwd err") 
      cnn:cuda()
      local input2 = input3:cuda()
      local gradOutput2 = gradOutput3:cuda()
      local output2 = cnn:forward(input2)
      local gradInput2 = cnn:backward(input2, gradOutput2)
      mytester:assertTensorEq(output2:float(), output3, 0.000001, "type cuda fwd err")
      mytester:assertTensorEq(gradInput2:float(), gradInput3, 0.00001, "type cuda bwd err") 
   end
end

function dpnntest.Module_gradParamClip()
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(10,10))
   mlp:add(nn.Euclidean(15,12))
   mlp:add(nn.SpatialConvolution(5,5,5,5))
   mlp:add(nn.LookupTable(100,100))
   local param, gradParam = mlp:getParameters()
   gradParam:uniform(-1,1)
   local norm = gradParam:norm()
   local mlp2 = mlp:clone()
   local cutoff = norm/2
   local norm2 = mlp2:gradParamClip(cutoff)
   mytester:assert(math.abs(norm2-norm) < 0.000001, "Module:gradParamClip norm err "..norm2.." ~= "..norm)
   local shrink_factor = cutoff / norm
   gradParam:mul(shrink_factor)
   local param2, gradParam2 = mlp2:getParameters()
   mytester:assertTensorEq(gradParam, gradParam2, 0.000001, "Module:gradParamClip clip err")
   
   local norm = gradParam:norm()
   local cutoff = norm*2
   local norm2 = mlp2:gradParamClip(cutoff)
   mytester:assert(math.abs(norm2-norm) < 0.000001, "Module:gradParamClip norm 2 err "..norm2.." ~= "..norm)
   mytester:assertTensorEq(gradParam, gradParam2, 0.000001, "Module:gradParamClip clip 2 err")
end

function dpnntest.Serial()
   local mlp = nn.Sequential():extend(
      nn.Linear(3,4),
      nn.Tanh(),
      nn.Linear(4,5),
      nn.Sequential():extend(
         nn.Linear(5,6),
         nn.Tanh(),
         nn.Linear(6,7)
      )
   )
   mlp:forward(torch.randn(4,3))
   mlp:backward(torch.randn(4,3), torch.randn(4,7))
   local mlp2 = mlp:Serial()
   mlp2:mediumSerial()
   local mlp3 = mlp2:clone()
   mytester:assert(mlp3.module.output:nElement() == 0, "serial medium empty err")
   mytester:assert(torch.type(mlp3.module.output) == 'torch.FloatTensor', "serial medium type err")
   local input = torch.randn(4,3)
   local gradOutput = torch.randn(4,7)
   local output = mlp:forward(input)
   local gradInput = mlp:backward(input, gradOutput)
   local output2 = mlp3:forward(input:float())
   local gradInput2 = mlp3:backward(input:float(), gradOutput:float())
   mytester:assertTensorEq(output:float(), output2, 0.000001, "serial forward error")
   mytester:assertTensorEq(gradInput:float(), gradInput2, 0.00001, "serial backward error")
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

function dpnntest.ReverseTable()
   -- input : { a, b, c, d }
   -- output : { c, b, a, d }
   local r = nn.ReverseTable()
   local input = {torch.randn(3,4), torch.randn(3,4), torch.randn(3,4), torch.randn(3,4)}
   local output = r:forward(input)
   
   mytester:assert(#output == 4, "ReverseTable #output")
   local k = 1
   for i=#input,1,-1 do
      mytester:assertTensorEq(input[i], output[k], 0.00001, "ReverseTable output err "..k)
      k = k + 1
   end
   
   local gradInput = r:backward(input, output)
   mytester:assert(#gradInput == 4, "ReverseTable #gradInput")
   for i=1,#input do
      mytester:assertTensorEq(gradInput[i], input[i], 0.00001, "ReverseTable gradInput err "..i)
   end
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

function dpnntest.Dictionary()
   local input = torch.randperm(80):resize(8,10)
   local gradOutput = torch.randn(8,10,50)
   local d = nn.Dictionary(100, 50)
   local output = d:forward(input)
   mytester:assertTableEq(output:size():totable(), {8,10,50}, 0.00001)
   d:zeroGradParameters()
   d:backward(input, gradOutput)
   local d2 = nn.LookupTable(100,50)
   d2 = d2:share(d, 'weight'):clone()
   local output2 = d2:forward(input)
   mytester:assertTensorEq(output, output2, 0.00001)
   d2:zeroGradParameters()
   d2:backward(input, gradOutput)
   mytester:assertTensorEq(d.gradWeight, d2.gradWeight, 0.000001)
   d:updateParameters(0.1)
   d2:updateParameters(0.1)
   mytester:assertTensorEq(d.weight, d2.weight, 0.00001)
end

function dpnntest.SpatialUniformCrop()
   local input = torch.Tensor(8,3,10,10):copy(torch.range(1,8):view(8,1,1,1):expand(8,3,10,10))
   local gradOutput = torch.Tensor(8,3,4,4):copy(torch.range(1,8):view(8,1,1,1):expand(8,3,4,4))
   local sc = nn.SpatialUniformCrop(4)
   local output, gradInput
   for i=1,100 do
      output = sc:forward(input)
      gradInput = sc:backward(input, gradOutput)
   end
   for i=1,8 do
      mytester:assert(math.abs(output[i]:mean() - i) < 0.0001, "SpatialUniformCrop output err "..i)
      mytester:assert(math.abs(gradInput[i]:mean() - ((i*4*4)/(10*10))) < 0.0001, "SpatialUniformCrop gradInput err"..i)
   end
end

function dpnntest.DontCast()
   local input = torch.randn(3,4)
   local gradOutput = torch.randn(3,2)
   local linear = nn.Linear(4,2):float()
   local mlp = nn.DontCast(linear, true)
   linear:zeroGradParameters()
   local linear = linear:clone()
   local output = mlp:forward(input)
   local gradInput = mlp:backward(input, gradOutput)
   mytester:assert(torch.type(output) == 'torch.DoubleTensor')
   mytester:assert(torch.type(gradInput) == 'torch.DoubleTensor')
   local output2 = linear:forward(input:float())
   local gradInput2 = linear:backward(input:float(), gradOutput:float())
   mytester:assertTensorEq(output:float(), output2, 0.000001)
   mytester:assertTensorEq(gradInput:float(), gradInput2, 0.000001)
   local mlp3 = nn.DontCast(linear:clone())
   mlp3:zeroGradParameters()
   local output3 = mlp3:forward(input:float())
   local gradInput3 = mlp3:backward(input:float(), gradOutput:float())
   mytester:assert(torch.type(output3) == 'torch.FloatTensor')
   mytester:assert(torch.type(gradInput3) == 'torch.FloatTensor')
   mytester:assertTensorEq(output3, output2, 0.000001)
   mytester:assertTensorEq(gradInput3, gradInput2, 0.000001)
   mlp:float()
   local output4 = mlp:forward(input:float())
   local gradInput4 = mlp:backward(input:float(), gradOutput:float())
   mytester:assert(torch.type(output4) == 'torch.FloatTensor')
   mytester:assert(torch.type(gradInput4) == 'torch.FloatTensor')
   mytester:assertTensorEq(output3, output4, 0.000001)
   mytester:assertTensorEq(gradInput3, gradInput4, 0.000001)
   mlp:double()
   mytester:assert(torch.type(linear.output) == 'torch.FloatTensor')
   local output = mlp:forward(input)
   local gradInput = mlp:backward(input, gradOutput)
   mytester:assert(torch.type(output4) == 'torch.FloatTensor')
   mytester:assert(torch.type(gradInput4) == 'torch.FloatTensor')
   mytester:assertTensorEq(output3, output:float(), 0.000001)
   mytester:assertTensorEq(gradInput3, gradInput:float(), 0.000001)
   
   -- test table inputs/outputs
   local input = {torch.randn(3,4), torch.randn(3,4)}
   local gradOutput = {torch.randn(3,2), torch.randn(3,2)}
   local linear = nn.ParallelTable():add(nn.Linear(4,2)):add(nn.Linear(4,2)):float()
   local mlp = nn.DontCast(linear, true)
   linear:zeroGradParameters()
   local linear = linear:clone()
   local output = mlp:forward(input)
   local gradInput = mlp:backward(input, gradOutput)
   mytester:assert(torch.type(output[1]) == 'torch.DoubleTensor')
   mytester:assert(torch.type(gradInput[1]) == 'torch.DoubleTensor')
   mytester:assert(torch.type(output[2]) == 'torch.DoubleTensor')
   mytester:assert(torch.type(gradInput[2]) == 'torch.DoubleTensor')
   local finput = _.map(input, function(k,v) return v:float() end)
   local foutput = _.map(output, function(k,v) return v:float() end)
   local fgradInput = _.map(gradInput, function(k,v) return v:float() end)
   local fgradOutput = _.map(gradOutput, function(k,v) return v:float() end)
   local output2 = linear:forward(finput)
   local gradInput2 = linear:backward(finput, fgradOutput)
   mytester:assertTensorEq(foutput[1], output2[1], 0.000001)
   mytester:assertTensorEq(foutput[2], output2[2], 0.000001)
   mytester:assertTensorEq(fgradInput[1], gradInput2[1], 0.000001)
   mytester:assertTensorEq(fgradInput[2], gradInput2[2], 0.000001)
   local mlp3 = nn.DontCast(linear:clone())
   mlp3:zeroGradParameters()
   local output3 = mlp3:forward(finput)
   local gradInput3 = mlp3:backward(finput, fgradOutput)
   mytester:assert(torch.type(output3[1]) == 'torch.FloatTensor')
   mytester:assert(torch.type(gradInput3[1]) == 'torch.FloatTensor')
   mytester:assert(torch.type(output3[2]) == 'torch.FloatTensor')
   mytester:assert(torch.type(gradInput3[2]) == 'torch.FloatTensor')
   mytester:assertTensorEq(output3[1], output2[1], 0.000001)
   mytester:assertTensorEq(gradInput3[1], gradInput2[1], 0.000001)
   mytester:assertTensorEq(output3[2], output2[2], 0.000001)
   mytester:assertTensorEq(gradInput3[2], gradInput2[2], 0.000001)
   mlp:float()
   local output4 = mlp:forward(finput)
   local gradInput4 = mlp:backward(finput, fgradOutput)
   mytester:assert(torch.type(output4[1]) == 'torch.FloatTensor')
   mytester:assert(torch.type(gradInput4[1]) == 'torch.FloatTensor')
   mytester:assert(torch.type(output4[2]) == 'torch.FloatTensor')
   mytester:assert(torch.type(gradInput4[2]) == 'torch.FloatTensor')
   mytester:assertTensorEq(output3[1], output4[1], 0.000001)
   mytester:assertTensorEq(gradInput3[1], gradInput4[1], 0.000001)
   mytester:assertTensorEq(output3[2], output4[2], 0.000001)
   mytester:assertTensorEq(gradInput3[2], gradInput4[2], 0.000001)
   mlp:double()
   mytester:assert(torch.type(linear.output) == 'table')
   mytester:assert(torch.type(linear.output[1]) == 'torch.FloatTensor')
   mytester:assert(torch.type(linear.output[2]) == 'torch.FloatTensor')
   local output = mlp:forward(input)
   local gradInput = mlp:backward(input, gradOutput)
   mytester:assertTensorEq(output3[1], output[1]:float(), 0.000001)
   mytester:assertTensorEq(gradInput3[1], gradInput[1]:float(), 0.000001)
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

function dpnntest.ReinforceNormal()
   local input = torch.randn(500,1000) -- means
   local gradOutput = torch.Tensor() -- will be ignored
   local reward = torch.randn(500)
   -- test scalar stdev
   local stdev = 1
   local rn = nn.ReinforceNormal(stdev)
   local output = rn:forward(input)
   mytester:assert(input:isSameSizeAs(output), "ReinforceNormal forward size err")
   local outstd = math.sqrt((input - output):pow(2):mean())
   local err = math.abs(outstd - stdev)
   mytester:assert(err < 0.1, "ReinforceNormal forward std err")
   rn:reinforce(reward)
   local gradInput = rn:updateGradInput(input, gradOutput)
   local gradInput2 = output:clone()
   gradInput2:add(-1, input):div(stdev^2)
   local reward2 = reward:view(500,1):expandAs(input)
   gradInput2:cmul(reward2):mul(-1)
   mytester:assertTensorEq(gradInput2, gradInput, 0.00001, "ReinforceNormal backward err")
   -- TODO test tensor stdev
end

function dpnntest.ReinforceBernoulli()
   local input = torch.Tensor(1000,10) 
   local p = torch.rand(1,10) -- probability of sampling a 1
   input:copy(p:expandAs(input))
   local gradOutput = torch.Tensor() -- will be ignored
   local reward = torch.randn(1000)
   local rb = nn.ReinforceBernoulli()
   local output = rb:forward(input)
   mytester:assert(input:isSameSizeAs(output), "ReinforceBernoulli forward size err")
   mytester:assert(output:min() == 0, "ReinforceBernoulli forward min val err")
   mytester:assert(output:max() == 1, "ReinforceBernoulli forward max val err")
   local binary = true
   output:apply(function(x) if not (x == 1 or x == 0) then binary = false end end)
   mytester:assert(binary, "ReinforceBernoulli forward binary val err")
   local p2 = output:mean(1)
   local err = (p - p2):abs():mean()
   mytester:assert(err < 0.05, "ReinforceBernoulli forward p err")
   rb:reinforce(reward)
   local gradInput = rb:updateGradInput(input, gradOutput)
   local gradInput2 = output:clone()
   local div = output:clone():fill(1):add(-1, input):cmul(input)
   gradInput2:add(-1, input):cdiv(div)
   local reward2 = reward:view(1000,1):expandAs(input)
   gradInput2:cmul(reward2):mul(-1)
   mytester:assertTensorEq(gradInput2, gradInput, 0.00001, "ReinforceBernoulli backward err")
end

function dpnntest.ReinforceCategorical()
   local input = torch.Tensor(1000,10) 
   local p = torch.rand(1,10)
   p:div(p:sum())
   input:copy(p:expandAs(input))
   local gradOutput = torch.Tensor() -- will be ignored
   local reward = torch.randn(1000)
   local rc = nn.ReinforceCategorical()
   local output = rc:forward(input)
   mytester:assert(input:isSameSizeAs(output), "ReinforceCategorical forward size err")
   mytester:assert(output:min() == 0, "ReinforceCategorical forward min val err")
   mytester:assert(output:max() == 1, "ReinforceCategorical forward max val err")
   mytester:assert(output:sum() == 1000, "ReinforceCategorical forward sum err")
   local binary = true
   output:apply(function(x) if not (x == 1 or x == 0) then binary = false end end)
   mytester:assert(binary, "ReinforceCategorical forward binary val err")
   local p2 = output:mean(1)
   local err = (p - p2):abs():mean()
   mytester:assert(err < 0.05, "ReinforceCategorical forward p err")
   rc:reinforce(reward)
   local gradInput = rc:updateGradInput(input, gradOutput)
   local gradInput2 = output:clone()
   gradInput2:cdiv(input)
   local reward2 = reward:view(1000,1):expandAs(input)
   gradInput2:cmul(reward2):mul(-1)
   mytester:assertTensorEq(gradInput2, gradInput, 0.00001, "ReinforceCategorical backward err")
end

function dpnntest.VRClassReward()
   local input = {torch.randn(13,10), torch.randn(13,1)}
   local target = torch.IntTensor(13):random(1,10)
   local rf = nn.Reinforce()
   local vrc = nn.VRClassReward(rf)
   local err = vrc:forward(input, target)
   local gradInput = vrc:backward(input, target)
   local val, idx = input[1]:max(2)
   local reward = torch.eq(idx:select(2,1):int(), target):double()
   local err2 = -reward:mean()
   mytester:assert(err == err2, "VRClassReward forward err")
   local gradInput2 = nn.MSECriterion():backward(input[2], reward)
   mytester:assertTensorEq(gradInput[2], gradInput2, 0.000001, "VRClassReward backward baseline err")
   mytester:assertTensorEq(gradInput[1], input[1]:zero(), 0.000001, "VRClassReward backward class err")
end

function dpnntest.Clip()
   local input = torch.randn(200,300)
   local gradOutput = torch.randn(200,300)
   local minval, maxval = -0.05, 0.1
   local clip = nn.Clip(minval, maxval)
   local output = clip:forward(input)
   local output2 = input:clone()
   local mask = input.new()
   mask:gt(input, maxval)
   output2[mask:type("torch.ByteTensor")] = maxval
   mask:lt(input, minval)
   output2[mask:type("torch.ByteTensor")] = minval   
   mytester:assertTensorEq(output, output2, 0.00001, "Clip forward err")
   local gradInput = clip:backward(input, gradOutput)
   mytester:assertTensorEq(gradInput, gradOutput, 0.00001, "Clip backward err")
end

function dpnntest.Constant()
   local input = torch.randn(20,3,7)
   local gradOutput = torch.randn(20,30,6)
   local value = torch.randn(30,6)
   local const = nn.Constant(value:clone(), 2)
   local output = const:forward(input)
   local gradInput = const:backward(input, output)
   local output2 = value:view(1,30,6):expand(20,30,6)
   mytester:assertTensorEq(output2, output, 0.000001, "Constant forward err")
   mytester:assertTensorEq(gradInput, input:zero(), 0.000001, "Constant backward err")
end

function dpnntest.SpatialGlimpse()
   if not pcall(function() require "image" end) then return end -- needs the image package
   local batchSize = 1
   local inputSize = {2,8,8}
   local glimpseSize = 4
   local input = torch.Tensor(batchSize, unpack(inputSize))
   input:range(1,input:nElement())
   input:resize(batchSize, unpack(inputSize))
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize, glimpseSize)
   local output2 = input:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse center 4 output depth=1 err")
   local outputSize = {batchSize, inputSize[1]*3, glimpseSize, glimpseSize}
   mytester:assertTableEq(output:size():totable(), outputSize, 0.000001, "SpatialGlimpse output size err")
   
   local glimpseSize = 5
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize, glimpseSize)
   local output2 = input:narrow(3,2,glimpseSize):narrow(4,2,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse center 5 output depth=1 err")
   
   local glimpseSize = 4
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(-1) -- top left corner patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize, glimpseSize)
   local padSize = math.floor((glimpseSize-1)/2)
   local pad = torch.Tensor(batchSize, inputSize[1], inputSize[2]+padSize*2, inputSize[3]+padSize*2):zero()
   pad:narrow(3, padSize + 1, inputSize[2]):narrow(4, padSize + 1, inputSize[3]):copy(input)
   local output2 = pad:narrow(3,1,glimpseSize):narrow(4,1,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse top-left 4 output depth=1 err")
   
   local glimpseSize = 5
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(-1) -- top left corner patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize, glimpseSize)
   local pad = torch.Tensor(batchSize, inputSize[1], inputSize[2]+glimpseSize, inputSize[3]+glimpseSize):zero()
   pad:narrow(3, (glimpseSize-1)/2 + 1, inputSize[2]):narrow(4, (glimpseSize-1)/2 + 1, inputSize[3]):copy(input)
   local output2 = pad:narrow(3,1,glimpseSize):narrow(4,1,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse top-left 5 output depth=1 err")
  
   local glimpseSize = 4
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(1) -- bottom-right corner patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize, glimpseSize)
   local pad = torch.Tensor(batchSize, inputSize[1], inputSize[2]+glimpseSize, inputSize[3]+glimpseSize):zero()
   pad:narrow(3, (glimpseSize-1)/2 + 1, inputSize[2]):narrow(4, (glimpseSize-1)/2 + 1, inputSize[3]):copy(input)
   local output2 = pad:narrow(3,inputSize[2]-1,glimpseSize):narrow(4,inputSize[3]-1,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse bottom-right 4 output depth=1 err")
   
   local glimpseSize = 5
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(1) -- bottom-right corner patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize, glimpseSize)
   local pad = torch.Tensor(batchSize, inputSize[1], inputSize[2]+glimpseSize, inputSize[3]+glimpseSize):zero()
   pad:narrow(3, (glimpseSize-1)/2, inputSize[2]):narrow(4, (glimpseSize-1)/2, inputSize[3]):copy(input)
   local output2 = pad:narrow(3,inputSize[2]-1,glimpseSize):narrow(4,inputSize[3]-1,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse bottom-right 5 output depth=1 err")
   
   local glimpseSize = 4
   local sg = nn.SpatialGlimpse(glimpseSize, 1)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 1, inputSize[1], glimpseSize, glimpseSize)
   local output2 = input:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse center 4 output depth=1 err")
   local gradInput = sg:backward({input,location}, output)
   local gradInput2 = input:clone():zero()
   gradInput2:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize):copy(output_:select(2,1))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpse backward 4 depth 1 error")
   
   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize, glimpseSize)
   local output2 = input:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse center 4 output depth=1 err")
   local gradOutput = output:clone()
   gradOutput:view(batchSize, 2, 2, glimpseSize, glimpseSize):select(2,1):fill(0) -- ignore first scale of glimpse
   local gradInput = sg:backward({input,location}, gradOutput)
   local srs = nn.SpatialReSampling{oheight=glimpseSize*2,owidth=glimpseSize*2}
   local gradInput2 = srs:updateGradInput(gradInput[1], output_:select(2,2))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpse backward 4 depth 2 error")
   
   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize, glimpseSize)
   local output2 = input:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse center 4 output depth=1 err")
   local gradOutput = output:clone()
   local gradInput = sg:backward({input,location}, gradOutput)
   local gradInput2 = input:clone():zero()
   gradInput2:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize):copy(output_:select(2,1))
   gradInput2:add(srs:updateGradInput(gradInput[1], output_:select(2,2)))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpse backward 4 depth 2 full error")
end

function dpnnbigtest.Reinforce()
   -- let us try to reinforce an mlp to learn a simple distribution
   local n = 10
   local inputs = torch.Tensor(n,3):uniform(0,0.1)
   local targets = torch.Tensor(n):fill(0)
   local stdev = 0.5
   local beta = 0.9
   local alpha = 1
   local lr = 0.1
   
   for i=1,inputs:size(1) do
      local j = (i % inputs:size(2)) + 1
      inputs[{i,j}] = torch.uniform(0.9,1.1)
      targets[i] = j
   end
   
   local M = 10
   local function train(mlp, cost, N, name) 
      local converged = false
      local baseReward
      local reward
      for i=1,M do
         mlp:reset()
         
         baseReward = 0
         for i=1,inputs:size(1) do
            mlp:evaluate()
            local target = targets:narrow(1,i,1)
            local output = mlp:forward(inputs:narrow(1,i,1))
            baseReward = baseReward - cost:forward(output, target)
         end
         baseReward = baseReward/inputs:size(1)

         for k=1,N do
            
            for i=1,inputs:size(1) do
               mlp:training()
               mlp:zeroGradParameters()
               local target = targets:narrow(1,i,1)
               local output = mlp:forward(inputs:narrow(1,i,1))
               local err = cost:forward(output, target)
               local gradOutput = cost:backward(output, target)
               mlp:backward(inputs:narrow(1,i,1), gradOutput)
               mlp:updateParameters(lr)
            end
            
            reward = 0
            for i=1,inputs:size(1) do
               mlp:evaluate()
               local target = targets:narrow(1,i,1)
               local output = mlp:forward(inputs:narrow(1,i,1))
               reward = reward - cost:forward(output, target)
            end
            reward = reward/inputs:size(1)
            
            if reward*0.7 >= baseReward then
               converged = true
               break
            end
         end
         
         if reward*0.7 >= baseReward then
            converged = true
            break
         end
      end
      
      mytester:assert(converged, name.." did not converge : "..reward.."*0.7 < "..baseReward)
   end
   
   -- ReinforceNormal
   local hiddenSize = 200
   local N = 10
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(inputs:size(2),hiddenSize))
   mlp:add(nn.Tanh())
   mlp:add(nn.ReinforceNormal(stdev))
   mlp:add(nn.Clip(-1,1))
   mlp:add(nn.Linear(hiddenSize, inputs:size(2)))
   mlp:add(nn.SoftMax())
   
   local cost = nn.VRClassReward(mlp, beta, alpha)
   
   train(mlp, cost, N, 'ReinforceNormal')
   
   -- ReinforceBernoulli
   local hiddenSize = 20
   local N = 30
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(inputs:size(2),hiddenSize))
   mlp:add(nn.Sigmoid())
   mlp:add(nn.ReinforceBernoulli())
   mlp:add(nn.Linear(hiddenSize, inputs:size(2)))
   mlp:add(nn.SoftMax())
   
   local cost = nn.VRClassReward(mlp, beta, alpha)
   
   train(mlp, cost, N, 'ReinforceBernoulli')
   
   -- ReinforceCategorical
   local hiddenSize = 200
   local N = 10
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(inputs:size(2),hiddenSize))
   mlp:add(nn.Tanh())
   mlp:add(nn.Linear(hiddenSize, inputs:size(2)))
   mlp:add(nn.SoftMax())
   mlp:add(nn.AddConstant(0.00001))
   mlp:add(nn.ReinforceCategorical())
   
   local cost = nn.VRClassReward(mlp, beta, alpha)
   
   train(mlp, cost, N, 'ReinforceCategorical')
end

-- Unit Test WhiteNoise
function dpnntest.WhiteNoise()
   local input = torch.zeros(3, 28, 28)
   local addNoise = nn.WhiteNoise()
   local output = addNoise:forward(input)
   local meanValue = output:mean()
   local stdValue = output:std()
   mytester:assert(meanValue > -0.01 and meanValue < 0.01)
   mytester:assert(stdValue < 0.15 and stdValue >= 0)

   -- Evaluate
   addNoise:evaluate() 
   output = addNoise:forward(input)
   meanValue = output:mean()
   stdValue = output:std()
   mytester:assert(meanValue == 0)
   mytester:assert(stdValue == 0)

   -- backprop
   addNoise:training()
   local gradOutput = torch.rand(3, 28, 28)
   local gradInput = addNoise:updateGradInput(input, gradOutput)
   mytester:assertTensorEq(gradOutput, gradInput, 0.000001, "WhiteNoise backward err")
end

function dpnn.test(tests)
   mytester = torch.Tester()
   mytester:add(dpnntest)
   math.randomseed(os.time())
   mytester:run(tests)
end

function dpnn.bigtest(tests)
   mytester = torch.Tester()
   mytester:add(dpnnbigtest)
   math.randomseed(os.time())
   mytester:run(tests)
end
