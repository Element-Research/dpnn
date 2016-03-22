
local dpnntest = {}
local dpnnbigtest = {}
local precision = 1e-5
local mytester

function dpnntest.Module_sharedClone()
   
   local function testrnn(mlp, name)
      mlp:zeroGradParameters()
      local mlp = mlp:clone()
      local clone = mlp:clone():sharedClone(true, true)
      
      for i=1,2 do
         local input = torch.randn(2,3)
         local gradOutput = torch.randn(2,4)
         
         local output = mlp:forward(input)
         local gradInput = mlp:backward(input, gradOutput)
         local output4 = clone:forward(input)
         local gradInput4 = clone:backward(input, gradOutput)
         
         mytester:assertTensorEq(output, output4, 0.00001, name.." updateOutput")
         mytester:assertTensorEq(gradInput, gradInput4, 0.00001, name.." updateGradInput")
         
         mlp:updateParameters(0.1)
         clone:updateParameters(0.1)
         
         local params, gradParams = mlp:parameters()
         local params2, gradParams2 = clone:parameters()
         
         mytester:assert(#params == #params2, name.." num params err")
         mytester:assert(#gradParams == #gradParams2, name.." num gradParams err")
         
         for i,param in ipairs(params) do
            mytester:assertTensorEq(param, params2[i], 0.00001, name.." params2 err "..i)
            mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.00001, name.." gradParams2 err "..i)
         end
      end
   end

   local function test(mlp, name)
      mlp:zeroGradParameters()
      local clone = mlp:clone()
      clone:share(mlp,"weight","bias","gradWeight","gradBias") -- this actually won't work for nn.Recurrent
      
      local mlp2 = mlp:clone() -- not shared with mlp
      local clone2 = mlp2:sharedClone(true, true)
      mlp2.__test = 1
      clone2.__test = 2
      mytester:assert(mlp2.__test ~= clone2.__test)
      
      local params, gradParams = mlp:parameters()
      local params4, gradParams4 = clone:parameters()
      local params2, gradParams2 = clone2:parameters()
      local params3, gradParams3 = mlp2:parameters()
      
      mytester:assert(#params == #params2, name.." num params err")
      mytester:assert(#params3 == #params2, name.." num params err")
      mytester:assert(#gradParams == #gradParams2, name.." num gradParams err")
      mytester:assert(#gradParams == #gradParams3, name.." num gradParams err")
      
      local input = torch.randn(2,3)
      local gradOutput = torch.randn(2,4)
      
      local output = mlp:forward(input)
      local gradInput = mlp:backward(input, gradOutput)
      
      for i,param in ipairs(params) do
         mytester:assertTensorEq(param, params4[i], 0.00001, name.." params4  err "..i) 
         mytester:assertTensorEq(gradParams[i], gradParams4[i], 0.00001, name.." gradParams4 err "..i)
      end
      
      local output4 = clone:forward(input)
      local gradInput4 = clone:backward(input, gradOutput)
      
      mytester:assertTensorEq(output, output4, 0.00001, name.." updateOutput")
      mytester:assertTensorEq(gradInput, gradInput4, 0.00001, name.." updateGradInput")
      
      for i,param in ipairs(params) do
         mytester:assertTensorEq(param, params4[i], 0.00001, name.." params4  err "..i) 
         mytester:assertTensorEq(gradParams[i], gradParams4[i], 0.00001, name.." gradParams4 err "..i)
      end
      
      local output2 = clone2:forward(input)
      local gradInput2 = clone2:backward(input, gradOutput)
      
      mytester:assertTensorEq(output, output2, 0.00001, name.." updateOutput")
      mytester:assertTensorEq(gradInput, gradInput2, 0.00001, name.." updateGradInput")
      
      for i,param in ipairs(params) do
         mytester:assertTensorEq(params2[i], params3[i], 0.00001, name.." params 2 3  err "..i) 
         mytester:assertTensorEq(gradParams2[i], gradParams3[i], 0.00001, name.." gradParams 2 3 err "..i)
      end
      
      local output3 = mlp2:forward(input)
      local gradInput3 = mlp2:backward(input, gradOutput)
      
      mytester:assertTensorEq(output3, output2, 0.00001, name.." updateOutput")
      mytester:assertTensorEq(gradInput3, gradInput2, 0.00001, name.." updateGradInput")
      
      for i,param in ipairs(params) do
         mytester:assertTensorEq(params2[i], params3[i], 0.00001, name.." params 2 3  err "..i) 
         mytester:assertTensorEq(gradParams2[i], gradParams3[i], 0.00001, name.." gradParams 2 3 err "..i)
      end
      
      mlp:updateParameters(0.1)
      mlp2:updateParameters(0.1)  
      
      for i,param in ipairs(params) do
         mytester:assertTensorEq(param, params3[i], 0.00001, name.." params3 (mlp vs mlp:clone()) err "..i) -- fail
         mytester:assertTensorEq(gradParams[i], gradParams3[i], 0.00001, name.." gradParams3 err "..i) -- fail
      end
   end
   
   test(nn.Linear(3,4), 'linear')
   
   local mlp = nn.Sequential()
   mlp:add(nn.Linear(3,7))
   mlp:add(nn.Tanh())
   mlp:add(nn.Euclidean(7,4))
   mlp:add(nn.LogSoftMax())
   test(mlp, 'sequential')
   
   
   local function test2(rnn, name)
      rnn:zeroGradParameters()
      local clone = rnn:sharedClone()
      
      local input = torch.randn(2,3)
      local gradOutput = torch.randn(2,4)
      
      local output = rnn:forward(input)
      local gradInput = rnn:backward(input, gradOutput)
      local output2 = clone:forward(input)
      local gradInput2 = clone:backward(input, gradOutput)
      
      mytester:assertTensorEq(output, output2, 0.00001, name.." updateOutput")
      mytester:assertTensorEq(gradInput, gradInput2, 0.00001, name.." updateGradInput")
      
      rnn:updateParameters(0.1)
      clone:updateParameters(0.1)
      
      local params, gradParams = rnn:parameters()
      local params2, gradParams2 = clone:parameters()
      
      mytester:assert(#params == #params2, name.." num params err")
      mytester:assert(#gradParams == #gradParams2, name.." num gradParams err")
      
      for i,param in ipairs(params) do
         mytester:assertTensorEq(param, params2[i], 0.00001, name.." params (rnn vs rnn:sharedClone()) err "..i)
         mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.00001, name.." gradParams (rnn vs rnn:sharedClone()) err "..i)
      end
      
      local output = rnn:forward(input)
      local gradInput = rnn:backward(input, gradOutput)
      local output2 = clone:forward(input)
      local gradInput2 = clone:backward(input, gradOutput)
      
      mytester:assertTensorEq(output, output2, 0.00001, name.." updateOutput")
      mytester:assertTensorEq(gradInput, gradInput2, 0.00001, name.." updateGradInput")
      
      rnn:updateParameters(0.1)
      clone:updateParameters(0.1)
      
      local params, gradParams = rnn:parameters()
      local params2, gradParams2 = clone:parameters()
      
      mytester:assert(#params == #params2, name.." num params err")
      mytester:assert(#gradParams == #gradParams2, name.." num gradParams err")
      
      for i,param in ipairs(params) do
         mytester:assertTensorEq(param, params2[i], 0.00001, name.." params (rnn vs rnn:sharedClone()) err "..i)
         mytester:assertTensorEq(gradParams[i], gradParams2[i], 0.00001, name.." gradParams (rnn vs rnn:sharedClone()) err "..i)
      end
   end
   
   if pcall(function() require 'rnn' end) then
      local rnn = nn.Recurrent(4,nn.Linear(3,4),nn.Linear(4,4), nn.Sigmoid(), 999)
      testrnn(rnn, 'rnn1')
      local seq = nn.Sequential()
      seq:add(nn.Repeater(nn.Recurrent(2,nn.Linear(3,2),nn.Linear(2,2), nn.Sigmoid(), 999), 3))
      seq:add(nn.Sequencer(nn.Linear(2,4)))
      seq:add(nn.SelectTable(-1))
      test2(seq, 'rnn2')
      test2(seq, 'rnn3')
   end
   
   if pcall(function() require 'nngraph' end) then
      local lin1 = nn.Linear(10, 10)
      local p1, gp1 = lin1:getParameters() 
      
      local lin2_ = lin1:clone()
      
      local x = nn.Identity()()
      local y = lin2_(x)
      
      local lin2 = nn.gModule({x}, {y})
      
      local lin3 = lin2:sharedClone()
      
      local input = torch.randn(4, 10)
      local gradOutput = torch.randn(4, 10)
      
      lin1:zeroGradParameters()
      lin2:zeroGradParameters()
      
      local params1, gradParams1 = lin1:parameters()
      local params2, gradParams2 = lin2:parameters()
      local params3, gradParams3 = lin3:parameters()
      
      local output1 = lin1:forward(input)
      local gradInput1 = lin1:backward(input, gradOutput)
      lin1:updateParameters(0.1)
      
      local output2 = lin2:forward(input)
      local gradInput2 = lin2:backward(input, gradOutput)
      lin2:updateParameters(0.1)
      
      mytester:assertTensorEq(output1, output2, 0.000001)
      mytester:assertTensorEq(gradInput1, gradInput2, 0.000001)
      
      for i=1,#params2 do
         mytester:assertTensorEq(params2[i], params3[i], 0.000001, "sharedClone nngraph param err "..i)
         mytester:assertTensorEq(gradParams2[i], gradParams3[i], 0.000001, "sharedClone nngraph gradParam err "..i)
         mytester:assertTensorEq(params1[i], params3[i], 0.000001, "sharedClone nngraph param err "..i)
         mytester:assertTensorEq(gradParams1[i], gradParams3[i], 0.000001, "sharedClone nngraph gradParam err "..i)
      end
      
      -- ok now lets forward/backward/update lin1 and lin3 to test sharedClone
      
      local output1 = lin1:forward(input)
      local gradInput1 = lin1:backward(input, gradOutput)
      
      local output3 = lin3:forward(input)
      local gradInput3 = lin3:backward(input, gradOutput)
      
      for i=1,#params2 do
         mytester:assertTensorEq(params2[i], params3[i], 0.000001, "sharedClone nngraph param err "..i)
         mytester:assertTensorEq(gradParams2[i], gradParams3[i], 0.000001, "sharedClone nngraph gradParam err "..i)
         mytester:assertTensorEq(params1[i], params3[i], 0.000001, "sharedClone nngraph param err "..i)
         mytester:assertTensorEq(gradParams1[i], gradParams3[i], 0.000001, "sharedClone nngraph gradParam err "..i)
      end
      
      mytester:assertTensorEq(output1, output3, 0.000001)
      mytester:assertTensorEq(gradInput1, gradInput3, 0.000001)
      
      for i=1,#params2 do
         mytester:assertTensorEq(gradParams1[i], gradParams3[i], 0.000001, "sharedClone nngraph gradParam err "..i)
      end
      
   end
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
   
   if pcall(function() require 'cunn' end) then
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

function dpnntest.Module_getParameters()
   -- test that getParameters will preserve parameters sharing for hidden modules
   local lin = nn.Linear(3,4)
   local lin2 = lin:sharedClone()
   lin.sharedClone = lin2
   local params, gradParams = lin:getParameters()
   params:add(-1)
   gradParams:fill(-1)
   
   local params1, gradParams1 = lin:parameters()
   local params2, gradParams2 = lin2:parameters()
   
   for i=1,#params1 do
      mytester:assertTensorEq(params1[i], params2[i], 0.000001, "getParameters param err "..i)
      mytester:assertTensorEq(gradParams1[i], gradParams2[i], 0.000001, "getParameters gradParam err "..i)
   end
end

function dpnntest.Serial()
   function test(mlp, name)
      local input = torch.randn(4,3)
      local gradOutput = torch.randn(4,7)
      local mlp2 = mlp:clone():Serial()
      
      local output = mlp:forward(input):clone()
      local gradInput = mlp:backward(input, gradOutput):clone()
      
      local output2 = mlp2:forward(input)
      local gradInput2 = mlp2:backward(input, gradOutput)
      
      mytester:assertTensorEq(output, output2, 0.000001, name.." serial forward error")
      mytester:assertTensorEq(gradInput, gradInput2, 0.00001, name.." serial backward error")
      
      mlp2:mediumSerial()
      mlp2.tensortype = 'torch.FloatTensor'
      local mlp3 = mlp2:clone()
      
      mytester:assert(mlp3.module.output:nElement() == 0, name.." serial medium empty err")
      mytester:assert(torch.type(mlp3.module.output) == 'torch.FloatTensor', name.." serial medium type err")
      
      mlp:zeroGradParameters()
      local output = mlp:forward(input)
      local gradInput = mlp:backward(input, gradOutput)

      mlp3:zeroGradParameters()
      local output2 = mlp3:forward(input:float())
      local gradInput2 = mlp3:backward(input:float(), gradOutput:float())
      
      mytester:assertTensorEq(output:float(), output2, 0.000001, name.." serial forward error")
      mytester:assertTensorEq(gradInput:float(), gradInput2, 0.00001, name.." serial backward error")
      
      local params, gradParams = mlp:parameters()
      local params2, gradParams2 = mlp3:parameters()
      mytester:assert(#params == #params2)
      for i,param in ipairs(params) do
         mytester:assertTensorEq(param:float(), params2[i], 0.00001, name.." params err "..i)
         mytester:assertTensorEq(gradParams[i]:float(), gradParams2[i], 0.00001, name.." gradParams err "..i)
      end
   end
   
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
   
   test(mlp, 'mlp')
   
   if pcall(function() require 'rnn' end) then
      local seq = nn.Sequential()
      seq:add(nn.Repeater(nn.Recurrent(2,nn.Linear(3,2),nn.Linear(2,2), nn.Sigmoid(), 999), 3))
      seq:add(nn.Sequencer(nn.Linear(2,7)))
      seq:add(nn.SelectTable(-1))
      test(seq, 'rnn2')
   end
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

function dpnntest.ZipTableOneToMany()
   -- input : { v, {a,b,c} }
   -- output : { {v,a}, {v,b}, {v,c} }
   local z = nn.ZipTableOneToMany()
   local input = { torch.randn(3), { torch.randn(4), torch.rand(4), torch.rand(4) } }
   local output = z:forward(input)
   mytester:assert(#output == 3, "ZipTableOneToMany #output")
   mytester:assert(#(output[1]) == 2, "ZipTableOneToMany #output[1]")
   mytester:assert(#(output[2]) == 2, "ZipTableOneToMany #output[2]")
   mytester:assert(#(output[3]) == 2, "ZipTableOneToMany #output[3]")
   mytester:assertTensorEq(input[1], output[1][1], 0.000001, "ZipTableOneToMany input1 output11")
   mytester:assertTensorEq(input[1], output[2][1], 0.000001, "ZipTableOneToMany input1 output21")
   mytester:assertTensorEq(input[1], output[3][1], 0.000001, "ZipTableOneToMany input1 output31")
   mytester:assertTensorEq(input[2][1], output[1][2], 0.000001, "ZipTableOneToMany input21")
   mytester:assertTensorEq(input[2][2], output[2][2], 0.000001, "ZipTableOneToMany input22")
   mytester:assertTensorEq(input[2][3], output[3][2], 0.000001, "ZipTableOneToMany input23")
   local gradInput = z:backward(input, output)
   mytester:assert(#gradInput == 2, "ZipTableOneToMany #gradInput")
   mytester:assert(#(gradInput[2]) == 3, "ZipTableOneToMany #gradInput[2]")
   mytester:assertTensorEq(input[2][1], gradInput[2][1], 0.000001, "ZipTableOneToMany gradInput21")
   mytester:assertTensorEq(input[2][2], gradInput[2][2], 0.000001, "ZipTableOneToMany gradInput22")
   mytester:assertTensorEq(input[2][3], gradInput[2][3], 0.000001, "ZipTableOneToMany gradInput32")
   mytester:assertTensorEq(torch.mul(input[1], 3), gradInput[1], 0.000001, "ZipTableOneToMany gradInput21")
end

function dpnntest.CAddTensorTable()
   -- input : { v, {a,b,c} }
   -- output : { v+a, v+b, v+c }
   local z = nn.CAddTensorTable()
   local input = { torch.randn(3), { torch.randn(3), torch.rand(3), torch.rand(3) } }
   local output = z:forward(input)
   mytester:assert(#output == 3, "CAddTensorTable #output")
   mytester:assertTensorEq(input[1]+input[2][1], output[1], 0.00001, "CAddTensorTable input21 output1")
   mytester:assertTensorEq(input[1]+input[2][2], output[2], 0.00001, "CAddTensorTable input22 output2")
   mytester:assertTensorEq(input[1]+input[2][3], output[3], 0.00001, "CAddTensorTable input23 output3")
   local gradInput = z:backward(input, output)
   mytester:assert(#gradInput == 2, "CAddTensorTable #gradInput")
   mytester:assert(#(gradInput[2]) == 3, "CAddTensorTable #gradInput[2]")
   mytester:assertTensorEq(output[1], gradInput[2][1], 0.000001, "CAddTensorTable gradInput21")
   mytester:assertTensorEq(output[2], gradInput[2][2], 0.000001, "CAddTensorTable gradInput22")
   mytester:assertTensorEq(output[3], gradInput[2][3], 0.000001, "CAddTensorTable gradInput23")
   mytester:assertTensorEq(output[1]+output[2]+output[3], gradInput[1], 0.000001, "CAddTensorTable gradInput1")
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

   local input = torch.zeros(1, 1, 120, 120)
   local temp = input[1]:narrow(2, 30, 60):narrow(3, 30, 60)
   temp:fill(1)
   local scale = {}
   scale['min'] = 0.8
   scale['max'] = 1.2

   local layer = nn.SpatialUniformCrop(100, 100, scale)
   local o = layer:forward(input)
   gradInput = layer:backward(input, o)
   mytester:assert(gradInput:max() ~= nil, "SpatialUniformCrop scaling error.")
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
   -- test input {mean, stdev}
   local mean, stdev = torch.randn(4,10), torch.rand(4,10)
   local input = {mean, stdev}
   local rn = nn.ReinforceNormal()
   local output = rn:updateOutput(input)
   local reward = torch.randn(4)
   rn:reinforce(reward)
   local gradInput = rn:backward(input, gradOutput)
   mytester:assert(mean:isSameSizeAs(output), "ReinforceNormal forward table input - output size err")
   mytester:assert(gradInput[1]:isSameSizeAs(mean), "ReinforceNormal backward table input - mean size err")
   mytester:assert(gradInput[2]:isSameSizeAs(stdev), "ReinforceNormal backward table input - stdev size err")
   local gradStdev = output:clone():add(-1, mean):pow(2)
   local stdev2 = torch.cmul(stdev,stdev)
   gradStdev:add(-1,stdev2)
   stdev2:cmul(stdev):add(0.00000001)
   gradStdev:cdiv(stdev2)
   local reward2 = reward:view(4,1):expandAs(gradStdev)
   gradStdev:cmul(reward2):mul(-1)
   mytester:assertTensorEq(gradInput[2], gradStdev, 0.000001, "ReinforceNormal backward table input - gradStdev err")
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
   gradInput2:cdiv(input+0.00000001)
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
   
   local input2 = torch.Tensor(unpack(inputSize))
   input2:range(1,input2:nElement())
   input2:resize(unpack(inputSize))
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location2 = torch.Tensor(2):fill(0) -- center patch
   local output2 = sg:forward{input2,location2}
   mytester:assertTensorEq(output2, output[1], 0.00001, "SpatialGlimpse online output depth=1 err")
   
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
   --print('bottom-right', output2, output_:select(2, 1))
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse bottom-right 4 output depth=1 err")
   
   local glimpseSize = 5
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(1) -- bottom-right corner patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize, glimpseSize)
   local pad = torch.Tensor(batchSize, inputSize[1], inputSize[2]+glimpseSize, inputSize[3]+glimpseSize):zero()
   pad:narrow(3, (glimpseSize-1)/2, inputSize[2]):narrow(4, (glimpseSize-1)/2, inputSize[3]):copy(input)
   local output2 = pad:narrow(3,inputSize[2]-1,glimpseSize):narrow(4,inputSize[3]-1,glimpseSize)
   --print('bottom-right', output2, output_:select(2, 1))
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
   
   -- test with spatial resampling
   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   sg.module = nn.SpatialReSampling{owidth=glimpseSize,oheight=glimpseSize}
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
   sg.module = nn.SpatialReSampling{owidth=glimpseSize,oheight=glimpseSize}
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
   
   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   sg.module = nn.SpatialReSampling{owidth=glimpseSize,oheight=glimpseSize}
   local output2 = sg:forward{input[1], location[1]}
   local gradInput2 = sg:backward({input[1], location[1]}, gradOutput[1])
   mytester:assertTensorEq(gradInput[1][1], gradInput2[1], 0.000001, "SpatialGlimpse backward online img err")
   mytester:assertTensorEq(gradInput[2][1], gradInput2[2], 0.000001, "SpatialGlimpse backward online loc err")
   
   -- test with spatial avg pool
   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize, glimpseSize)
   local output2 = input:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse avgpool center 4 output depth=1 err")
   local gradOutput = output:clone()
   gradOutput:view(batchSize, 2, 2, glimpseSize, glimpseSize):select(2,1):fill(0) -- ignore first scale of glimpse
   local gradInput = sg:backward({input,location}, gradOutput)
   local srs = nn.SpatialAveragePooling(2,2,2,2)
   local gradInput2 = srs:updateGradInput(gradInput[1], output_:select(2,2))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpse avgpool backward 4 depth 2 error")
   
   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize, glimpseSize)
   local output2 = input:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse avgpool center 4 output depth=1 err")
   local gradOutput = output:clone()
   local gradInput = sg:backward({input,location}, gradOutput)
   local gradInput2 = input:clone():zero()
   gradInput2:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize):copy(output_:select(2,1))
   gradInput2:add(srs:updateGradInput(gradInput[1], output_:select(2,2)))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpse avgpool backward 4 depth 2 full error")
   
   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   local output2 = sg:forward{input[1], location[1]}
   local gradInput2 = sg:backward({input[1], location[1]}, gradOutput[1])
   mytester:assertTensorEq(gradInput[1][1], gradInput2[1], 0.000001, "SpatialGlimpse avgpool backward online img err")
   mytester:assertTensorEq(gradInput[2][1], gradInput2[2], 0.000001, "SpatialGlimpse avgpool backward online loc err")
   
   -- test avg pool with cuda
   if not pcall(function() require "cunn" end) then return end -- needs the cunn package
   local input = input:cuda()
   
   local sg = nn.SpatialGlimpse(glimpseSize, 2):cuda()
   local location = torch.CudaTensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize, glimpseSize)
   local output2 = input:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse avgpool center 4 output depth=1 err")
   local gradOutput = output:clone()
   gradOutput:view(batchSize, 2, 2, glimpseSize, glimpseSize):select(2,1):fill(0) -- ignore first scale of glimpse
   local gradInput = sg:backward({input,location}, gradOutput)
   local srs = nn.SpatialAveragePooling(2,2,2,2):cuda()
   local gradInput2 = srs:updateGradInput(gradInput[1], output_:select(2,2))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpse avgpool backward 4 depth 2 error")
   
   local sg = nn.SpatialGlimpse(glimpseSize, 2):cuda()
   local location = torch.CudaTensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize, glimpseSize)
   local output2 = input:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize)
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpse avgpool center 4 output depth=1 err")
   local gradOutput = output:clone()
   local gradInput = sg:backward({input,location}, gradOutput)
   local gradInput2 = input:clone():zero()
   gradInput2:narrow(3,3,glimpseSize):narrow(4,3,glimpseSize):copy(output_:select(2,1))
   gradInput2:add(srs:updateGradInput(gradInput[1], output_:select(2,2)))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpse avgpool backward 4 depth 2 full error")
   
   local sg = nn.SpatialGlimpse(glimpseSize, 2):cuda()
   local output2 = sg:forward{input[1], location[1]}
   local gradInput2 = sg:backward({input[1], location[1]}, gradOutput[1])
   mytester:assertTensorEq(gradInput[1][1], gradInput2[1], 0.000001, "SpatialGlimpse avgpool backward online img err")
   mytester:assertTensorEq(gradInput[2][1], gradInput2[2], 0.000001, "SpatialGlimpse avgpool backward online loc err")
   
   if false then
      -- benchmark GPU vs CPU
      local location = torch.FloatTensor(32,2):uniform(-1,1)
      local input = torch.FloatTensor(32,3,224,224):uniform(0,1)
      local gradOutput = torch.FloatTensor(32,9,32,32):uniform(0,1)
      local sg = nn.SpatialGlimpse(32, 3, 2):float()
      sg:forward{input,location}
      local a = torch.Timer()
      for i=1,5 do
         sg:forward{input,location}
      end
      local fwdCPUtime = a:time().real
      
      sg:cuda()
      location = location:cuda()
      input = input:cuda()
      gradOutput = gradOutput:cuda()
      sg:forward{input,location}
      a = torch.Timer()
      for i=1,5 do
         sg:forward{input,location}
      end
      local fwdGPUtime = a:time().real
      print(fwdGPUtime, fwdCPUtime, fwdCPUtime/fwdGPUtime)
      -- 0.13885092735291  2.0344181060791  14.651815042678
   end
end

function dpnntest.SpatialGlimpse_backwardcompat()
   -- this is ugly, but I know this verson of the module works.
   -- So we try to match the newer versions to it 
   local SG, parent = torch.class("nn.SG", "nn.Module")

   function SG:__init(size, depth, scale)
      require 'nnx'
      self.size = size -- height == width
      self.depth = depth or 3
      self.scale = scale or 2
      
      assert(torch.type(self.size) == 'number')
      assert(torch.type(self.depth) == 'number')
      assert(torch.type(self.scale) == 'number')
      parent.__init(self)
      self.gradInput = {torch.Tensor(), torch.Tensor()}
      if self.scale == 2 then
         self.module = nn.SpatialAveragePooling(2,2,2,2)
      else
         self.module = nn.SpatialReSampling{oheight=size,owidth=size}
      end
      self.modules = {self.module}
   end

   -- a bandwidth limited sensor which focuses on a location.
   -- locations index the x,y coord of the center of the output glimpse
   function SG:updateOutput(inputTable)
      assert(torch.type(inputTable) == 'table')
      assert(#inputTable >= 2)
      local input, location = unpack(inputTable)
      input, location = self:toBatch(input, 3), self:toBatch(location, 1)
      assert(input:dim() == 4 and location:dim() == 2)
      
      self.output:resize(input:size(1), self.depth, input:size(2), self.size, self.size)
      
      self._crop = self._crop or self.output.new()
      self._pad = self._pad or input.new()
      
      for sampleIdx=1,self.output:size(1) do
         local outputSample = self.output[sampleIdx]
         local inputSample = input[sampleIdx]
         local xy = location[sampleIdx]
         -- (-1,-1) top left corner, (1,1) bottom right corner of image
         local x, y = xy:select(1,1), xy:select(1,2)
         -- (0,0), (1,1)
         x, y = (x+1)/2, (y+1)/2
         
         -- for each depth of glimpse : pad, crop, downscale
         local glimpseSize = self.size
         for depth=1,self.depth do 
            local dst = outputSample[depth]
            if depth > 1 then
               glimpseSize = glimpseSize*self.scale
            end
            
            -- add zero padding (glimpse could be partially out of bounds)
            local padSize = math.floor((glimpseSize-1)/2)
            self._pad:resize(input:size(2), input:size(3)+padSize*2, input:size(4)+padSize*2):zero()
            local center = self._pad:narrow(2,padSize+1,input:size(3)):narrow(3,padSize+1,input:size(4))
            center:copy(inputSample)
            
            -- crop it
            local h, w = self._pad:size(2)-glimpseSize, self._pad:size(3)-glimpseSize
            local x, y = math.min(h,math.max(0,x*h)),  math.min(w,math.max(0,y*w))
            
            if depth == 1 then
               dst:copy(self._pad:narrow(2,x+1,glimpseSize):narrow(3,y+1,glimpseSize))
            else
               self._crop:resize(input:size(2), glimpseSize, glimpseSize)
               self._crop:copy(self._pad:narrow(2,x+1,glimpseSize):narrow(3,y+1,glimpseSize))
            
               if torch.type(self.module) == 'nn.SpatialAveragePooling' then
                  local poolSize = glimpseSize/self.size
                  assert(poolSize % 2 == 0)
                  self.module.kW = poolSize
                  self.module.kH = poolSize
                  self.module.dW = poolSize
                  self.module.dH = poolSize
               end
               dst:copy(self.module:updateOutput(self._crop))
            end
         end
      end
      
      self.output:resize(input:size(1), self.depth*input:size(2), self.size, self.size)
      self.output = self:fromBatch(self.output, 1)
      return self.output
   end

   function SG:updateGradInput(inputTable, gradOutput)
      local input, location = unpack(inputTable)
      local gradInput, gradLocation = unpack(self.gradInput)
      input, location = self:toBatch(input, 3), self:toBatch(location, 1)
      gradOutput = self:toBatch(gradOutput, 3)
      
      gradInput:resizeAs(input):zero()
      gradLocation:resizeAs(location):zero() -- no backprop through location
      
      gradOutput = gradOutput:view(input:size(1), self.depth, input:size(2), self.size, self.size)
      
      for sampleIdx=1,gradOutput:size(1) do
         local gradOutputSample = gradOutput[sampleIdx]
         local gradInputSample = gradInput[sampleIdx]
         local xy = location[sampleIdx] -- height, width
         -- (-1,-1) top left corner, (1,1) bottom right corner of image
         local x, y = xy:select(1,1), xy:select(1,2)
         -- (0,0), (1,1)
         x, y = (x+1)/2, (y+1)/2
         
         -- for each depth of glimpse : pad, crop, downscale
         local glimpseSize = self.size
         for depth=1,self.depth do 
            local src = gradOutputSample[depth]
            if depth > 1 then
               glimpseSize = glimpseSize*self.scale
            end
            
            -- add zero padding (glimpse could be partially out of bounds)
            local padSize = math.floor((glimpseSize-1)/2)
            self._pad:resize(input:size(2), input:size(3)+padSize*2, input:size(4)+padSize*2):zero()
            
            local h, w = self._pad:size(2)-glimpseSize, self._pad:size(3)-glimpseSize
            local x, y = math.min(h,math.max(0,x*h)),  math.min(w,math.max(0,y*w))
            local pad = self._pad:narrow(2, x+1, glimpseSize):narrow(3, y+1, glimpseSize)
            
            -- upscale glimpse for different depths
            if depth == 1 then
               pad:copy(src)
            else
               self._crop:resize(input:size(2), glimpseSize, glimpseSize)
               
               if torch.type(self.module) == 'nn.SpatialAveragePooling' then
                  local poolSize = glimpseSize/self.size
                  assert(poolSize % 2 == 0)
                  self.module.kW = poolSize
                  self.module.kH = poolSize
                  self.module.dW = poolSize
                  self.module.dH = poolSize
               end
               
               pad:copy(self.module:updateGradInput(self._crop, src))
            end
           
            -- copy into gradInput tensor (excluding padding)
            gradInputSample:add(self._pad:narrow(2, padSize+1, input:size(3)):narrow(3, padSize+1, input:size(4)))
         end
      end
      
      self.gradInput[1] = self:fromBatch(gradInput, 1)
      self.gradInput[2] = self:fromBatch(gradLocation, 1)
      
      return self.gradInput
   end
   
   local batchSize = 1
   local inputSize = {2,8,8}
   local glimpseSize = 4
   local input = torch.randn(batchSize, unpack(inputSize))
   input:resize(batchSize, unpack(inputSize))
   
   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   local sg2 = nn.SG(glimpseSize, 2)
   
   for i=1,10 do
      local location = torch.Tensor(batchSize, 2):uniform(-0.9,0.9)
      local output = sg:forward{input,location}
      local output2 = sg2:forward{input,location}
      mytester:assertTensorEq(output, output2, 0.0000001, "SpatialGlimpse err")
   end
   
end

-- test rectangle-shaped glimpse sampling
function dpnntest.SpatialGlimpseRect()
   if not pcall(function() require "image" end) then return end -- needs the image package
   local batchSize = 1
   local inputSize = {2,8,8}
   
   local glimpseSize = {4,2} -- {height, width}
   local input = torch.Tensor(batchSize, unpack(inputSize))
   input:range(1,input:nElement())
   input:resize(batchSize, unpack(inputSize))
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize[1], glimpseSize[2])
   local y0 = (input:size(3)-glimpseSize[1])/2 + 1
   local x0 = (input:size(4)-glimpseSize[2])/2 + 1
   local output2 = input:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect center 4 output depth=1 err")
   local outputSize = {batchSize, inputSize[1]*3, glimpseSize[1], glimpseSize[2]}
   mytester:assertTableEq(output:size():totable(), outputSize, 0.000001, "SpatialGlimpseRect output size err")
   
   local input2 = torch.Tensor(unpack(inputSize))
   input2:range(1,input2:nElement())
   input2:resize(unpack(inputSize))
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location2 = torch.Tensor(2):fill(0) -- center patch
   local output2 = sg:forward{input2,location2}
   mytester:assertTensorEq(output2, output[1], 0.00001, "SpatialGlimpseRect online output depth=1 err")
   
   local glimpseSize = {5,3}
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize[1], glimpseSize[2])
   local y0 = math.floor((input:size(3)-glimpseSize[1])/2) + 1
   local x0 = math.floor((input:size(4)-glimpseSize[2])/2) + 1
   local output2 = input:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect center 5 output depth=1 err")
   
   local glimpseSize = {4,3}
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(-1) -- top left corner patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize[1], glimpseSize[2])
   local padSize = {math.floor((glimpseSize[1]-1)/2), math.floor((glimpseSize[2]-1)/2)}
   local pad = torch.Tensor(batchSize, inputSize[1], inputSize[2]+padSize[1]*2, inputSize[3]+padSize[2]*2):zero()
   pad:narrow(3, padSize[1] + 1, inputSize[2]):narrow(4, padSize[2] + 1, inputSize[3]):copy(input)
   local output2 = pad:narrow(3,1,glimpseSize[1]):narrow(4,1,glimpseSize[2])
   --print('top-left', output2, output_:select(2, 1))
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect top-left 4 output depth=1 err")
   
   local glimpseSize = {5,4}
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(-1) -- top left corner patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize[1], glimpseSize[2])
   local pad = torch.Tensor(batchSize, inputSize[1], inputSize[2]+glimpseSize[1], inputSize[3]+glimpseSize[2]):zero()
   local y0 = math.floor((glimpseSize[1]-1)/2) + 1
   local x0 = math.floor((glimpseSize[2]-1)/2) + 1
   pad:narrow(3, y0, inputSize[2]):narrow(4, x0, inputSize[3]):copy(input)
   local output2 = pad:narrow(3,1,glimpseSize[1]):narrow(4,1,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect top-left 5 output depth=1 err")
  
   local glimpseSize = {3,4}
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(1) -- bottom-right corner patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize[1], glimpseSize[2])
   local pad = torch.Tensor(batchSize, inputSize[1], inputSize[2]+glimpseSize[1], inputSize[3]+glimpseSize[2]):zero()
   local y0 = math.floor((glimpseSize[1]-1)/2) + 1
   local x0 = math.floor((glimpseSize[2]-1)/2) + 1
   pad:narrow(3, y0, inputSize[2]):narrow(4, x0, inputSize[3]):copy(input)
   local dy = math.floor((glimpseSize[1])/2)
   local dx = math.floor((glimpseSize[2])/2)
   local output2 = pad:narrow(3,inputSize[2]-dy+1,glimpseSize[1]):narrow(4,inputSize[3]-dx+1,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect bottom-right 4 output depth=1 err")
   
   local glimpseSize = {4,5}
   local sg = nn.SpatialGlimpse(glimpseSize)
   local location = torch.Tensor(batchSize, 2):fill(1) -- bottom-right corner patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 3, inputSize[1], glimpseSize[1], glimpseSize[2])
   local pad = torch.Tensor(batchSize, inputSize[1], inputSize[2]+glimpseSize[1], inputSize[3]+glimpseSize[2]):zero()
   local y0 = math.floor((glimpseSize[1])/2)
   local x0 = math.floor((glimpseSize[2])/2)
   pad:narrow(3, y0, inputSize[2]):narrow(4, x0, inputSize[3]):copy(input)
   local dy = math.floor((glimpseSize[1])/2)
   local dx = math.floor((glimpseSize[2])/2)
   local output2 = pad:narrow(3,inputSize[2]-dy+1,glimpseSize[1]):narrow(4,inputSize[3]-dx+1,glimpseSize[2])
   --print('bottom-right', output2, output_:select(2, 1))
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect bottom-right 5 output depth=1 err")

   -- test gradients
   local glimpseSize = {4,4} -- {height, width}
   local sg = nn.SpatialGlimpse(glimpseSize, 1)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 1, inputSize[1], glimpseSize[1], glimpseSize[2])
   local y0 = math.floor((input:size(3)-glimpseSize[1])/2) + 1
   local x0 = math.floor((input:size(4)-glimpseSize[2])/2) + 1
   local output2 = input:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect center 4 output depth=1 err")
   local gradInput = sg:backward({input,location}, output)
   local gradInput2 = input:clone():zero()
   gradInput2:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2]):copy(output_:select(2,1))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpseRect backward 4 depth 1 error")

   -- test with spatial resampling
   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   sg.module = nn.SpatialReSampling{owidth=glimpseSize[2],oheight=glimpseSize[1]}
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize[1], glimpseSize[2])
   local y0 = math.floor((input:size(3)-glimpseSize[1])/2) + 1
   local x0 = math.floor((input:size(4)-glimpseSize[2])/2) + 1
   local output2 = input:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect center 4 output depth=1 err")
   local gradOutput = output:clone()
   gradOutput:view(batchSize, 2, 2, glimpseSize[1], glimpseSize[2]):select(2,1):fill(0) -- ignore first scale of glimpse
   local gradInput = sg:backward({input,location}, gradOutput)
   local srs = nn.SpatialReSampling{oheight=glimpseSize[2]*2,owidth=glimpseSize[1]*2}
   local gradInput2 = srs:updateGradInput(gradInput[1], output_:select(2,2))
   --print('SpatialReSampling', gradInput2, gradInput[1])
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpseRect backward 4 depth 2 error")
   
   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   sg.module = nn.SpatialReSampling{owidth=glimpseSize[2],oheight=glimpseSize[1]}
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize[1], glimpseSize[2])
   local output2 = input:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect center 4 output depth=1 err")
   local gradOutput = output:clone()
   local gradInput = sg:backward({input,location}, gradOutput)
   local gradInput2 = input:clone():zero()
   gradInput2:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2]):copy(output_:select(2,1))
   gradInput2:add(srs:updateGradInput(gradInput[1], output_:select(2,2)))
   --print('SpatialReSampling', gradInput2, gradInput[1])
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpseRect backward 4 depth 2 full error")
   
   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   sg.module = nn.SpatialReSampling{owidth=glimpseSize[2],oheight=glimpseSize[1]}
   local output2 = sg:forward{input[1], location[1]}
   local gradInput2 = sg:backward({input[1], location[1]}, gradOutput[1])
   mytester:assertTensorEq(gradInput[1][1], gradInput2[1], 0.000001, "SpatialGlimpseRect backward online img err")
   mytester:assertTensorEq(gradInput[2][1], gradInput2[2], 0.000001, "SpatialGlimpseRect backward online loc err")
   
   -- test with spatial avg pool
   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize[1], glimpseSize[2])
   local y0 = math.floor((input:size(3)-glimpseSize[1])/2) + 1
   local x0 = math.floor((input:size(4)-glimpseSize[2])/2) + 1
   local output2 = input:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect avgpool center 4 output depth=1 err")
   local gradOutput = output:clone()
   gradOutput:view(batchSize, 2, 2, glimpseSize[1], glimpseSize[2]):select(2,1):fill(0) -- ignore first scale of glimpse
   local gradInput = sg:backward({input,location}, gradOutput)
   local srs = nn.SpatialAveragePooling(2,2,2,2)
   local gradInput2 = srs:updateGradInput(gradInput[1], output_:select(2,2))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpseRect avgpool backward 4 depth 2 error")
   
   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   local location = torch.Tensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize[1], glimpseSize[2])
   local output2 = input:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect avgpool center 4 output depth=1 err")
   local gradOutput = output:clone()
   local gradInput = sg:backward({input,location}, gradOutput)
   local gradInput2 = input:clone():zero()
   gradInput2:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2]):copy(output_:select(2,1))
   gradInput2:add(srs:updateGradInput(gradInput[1], output_:select(2,2)))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpseRect avgpool backward 4 depth 2 full error")
   
   local sg = nn.SpatialGlimpse(glimpseSize, 2)
   local output2 = sg:forward{input[1], location[1]}
   local gradInput2 = sg:backward({input[1], location[1]}, gradOutput[1])
   mytester:assertTensorEq(gradInput[1][1], gradInput2[1], 0.000001, "SpatialGlimpseRect avgpool backward online img err")
   mytester:assertTensorEq(gradInput[2][1], gradInput2[2], 0.000001, "SpatialGlimpseRect avgpool backward online loc err")
   
   -- test avg pool with cuda
   if not pcall(function() require "cunn" end) then return end -- needs the cunn package
   local input = input:cuda()
   
   local sg = nn.SpatialGlimpse(glimpseSize, 2):cuda()
   local location = torch.CudaTensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize[1], glimpseSize[2])
   local output2 = input:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect avgpool center 4 output depth=1 err")
   local gradOutput = output:clone()
   gradOutput:view(batchSize, 2, 2, glimpseSize[1], glimpseSize[2]):select(2,1):fill(0) -- ignore first scale of glimpse
   local gradInput = sg:backward({input,location}, gradOutput)
   local srs = nn.SpatialAveragePooling(2,2,2,2):cuda()
   local gradInput2 = srs:updateGradInput(gradInput[1], output_:select(2,2))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpseRect avgpool backward 4 depth 2 error")
   
   local sg = nn.SpatialGlimpse(glimpseSize, 2):cuda()
   local location = torch.CudaTensor(batchSize, 2):fill(0) -- center patch
   local output = sg:forward{input,location}
   local output_ = output:view(batchSize, 2, inputSize[1], glimpseSize[1], glimpseSize[2])
   local output2 = input:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2])
   mytester:assertTensorEq(output2, output_:select(2, 1), 0.00001, "SpatialGlimpseRect avgpool center 4 output depth=1 err")
   local gradOutput = output:clone()
   local gradInput = sg:backward({input,location}, gradOutput)
   local gradInput2 = input:clone():zero()
   gradInput2:narrow(3,y0,glimpseSize[1]):narrow(4,x0,glimpseSize[2]):copy(output_:select(2,1))
   gradInput2:add(srs:updateGradInput(gradInput[1], output_:select(2,2)))
   mytester:assertTensorEq(gradInput[1], gradInput2, 0.000001, "SpatialGlimpseRect avgpool backward 4 depth 2 full error")
   
   local sg = nn.SpatialGlimpse(glimpseSize, 2):cuda()
   local output2 = sg:forward{input[1], location[1]}
   local gradInput2 = sg:backward({input[1], location[1]}, gradOutput[1])
   mytester:assertTensorEq(gradInput[1][1], gradInput2[1], 0.000001, "SpatialGlimpseRect avgpool backward online img err")
   mytester:assertTensorEq(gradInput[2][1], gradInput2[2], 0.000001, "SpatialGlimpseRect avgpool backward online loc err")
   
   if false then
      -- benchmark GPU vs CPU
      local location = torch.FloatTensor(32,2):uniform(-1,1)
      local input = torch.FloatTensor(32,3,224,224):uniform(0,1)
      local gradOutput = torch.FloatTensor(32,9,32,32):uniform(0,1)
      local sg = nn.SpatialGlimpse({32,24}, 3, 2):float()
      sg:forward{input,location}
      local a = torch.Timer()
      for i=1,5 do
         sg:forward{input,location}
      end
      local fwdCPUtime = a:time().real
      
      sg:cuda()
      location = location:cuda()
      input = input:cuda()
      gradOutput = gradOutput:cuda()
      sg:forward{input,location}
      a = torch.Timer()
      for i=1,5 do
         sg:forward{input,location}
      end
      local fwdGPUtime = a:time().real
      print(fwdGPUtime, fwdCPUtime, fwdCPUtime/fwdGPUtime)
      -- 
   end
end

function dpnntest.ArgMax()
   local inputSize = 5
   local batchSize = 3
   local input = torch.randn(batchSize, inputSize)
   local gradOutput = torch.randn(batchSize):long()
   local am = nn.ArgMax(1,1)
   local output = am:forward(input)
   local gradInput = am:backward(input, gradOutput)
   local val, idx = torch.max(input, 2)
   mytester:assertTensorEq(idx:select(2,1), output, 0.000001, "ArgMax output asLong err")
   mytester:assertTensorEq(gradInput, input:clone():zero(), 0.000001, "ArgMax gradInput asLong err")
   local am = nn.ArgMax(1,1,false)
   local output = am:forward(input)
   local gradInput = am:backward(input, gradOutput)
   local val, idx = torch.max(input, 2)
   mytester:assertTensorEq(idx:select(2,1):double(), output, 0.000001, "ArgMax output not asLong err")
   mytester:assertTensorEq(gradInput, input:clone():zero(), 0.000001, "ArgMax gradInput not asLong err") 
end

function dpnntest.CategoricalEntropy()
   local inputSize = 5
   local batchSize = 10
   local minEntropy = 12
   local input_ = torch.randn(batchSize, inputSize)
   local input = nn.SoftMax():updateOutput(input_)
   local gradOutput = torch.Tensor(batchSize, inputSize):zero()
   local ce = nn.CategoricalEntropy()
   local output = ce:forward(input)
   mytester:assertTensorEq(input, output, 0.0000001, "CategoricalEntropy output err")
   local gradInput = ce:backward(input, gradOutput)
   local output2 = input:sum(1)[1]
   output2:div(output2:sum())
   local log2 = torch.log(output2 + 0.000001)
   local entropy2 = -output2:cmul(log2):sum()
   mytester:assert(math.abs(ce.entropy - entropy2) < 0.000001, "CategoricalEntropy entropy err")
   local gradEntropy2 = log2:add(1) -- -1*(-1 - log(p(x))) = 1 + log(p(x))
   gradEntropy2:div(input:sum())
   local gradInput2 = gradEntropy2:div(batchSize):view(1,inputSize):expandAs(input)
   mytester:assertTensorEq(gradInput2, gradInput, 0.000001, "CategoricalEntropy gradInput err")
end

function dpnntest.TotalDropout()
   local batchSize = 4
   local inputSize = 3
   local input = torch.randn(batchSize, inputSize)
   local gradOutput = torch.randn(batchSize, inputSize)
   local td = nn.TotalDropout()
   local nOne = 0
   for i=1,10 do
      local output = td:forward(input)
      local gradInput = td:backward(input, gradOutput)
      if td.noise == 0 then
         mytester:assert(output:sum() == 0, "TotalDropout forward 0 err")
         mytester:assert(gradInput:sum() == 0, "TotalDropout backward 0 err")
      else
         mytester:assertTensorEq(output, input, 0.000001, "TotalDropout forward 1 err")
         mytester:assertTensorEq(gradInput, gradOutput, 0.000001, "TotalDropout backward 1 err")
         nOne = nOne + 1
      end
   end
   mytester:assert(nOne < 10 and nOne > 1, "TotalDropout bernoulli error")
end

function dpnnbigtest.Reinforce()
   error"this needs to be updated with new VRClassReward interface"
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

-- Unit Test SpatialBinaryLogisticRegression criterion
function dpnntest.SpatialBinaryLogisticRegression()
   local crit = nn.SpatialBinaryLogisticRegression()
   local k = 32
   local h = 28
   local w = 28

   -- Working with batch of images
   local input = torch.zeros(k, 1, h, w)
   local target = torch.zeros(k, 1, h, w)
   local inputs = {1, 0, -1}
   local targets = {1, 0, -1}
   for _,i in pairs(inputs) do
      for _,t in pairs(targets) do

      input:fill(i)
      target:fill(t)
      -- Check forward
      local loss = crit:updateOutput(input, target)
      local myLoss = math.log(1+math.exp(-1*i*t))/2
      mytester:assert( loss >= myLoss-precision and loss <= myLoss+precision,
                       "SpatialBinaryLogisticRegression cost incorrect.")

      -- Check backward
      local gradInput = crit:updateGradInput(input, target)
      local g1 = gradInput[1][1][1][1]
      local gi = (1/(1+math.exp(-1*i*t)))*math.exp(-1*i*t)*(-1*t)/(2*k*h*w)
      mytester:assert( g1 >= gi-precision and g1 <= gi+precision,
                      "SpatialBinaryLogisticRegression gradInput error.")
      end
   end

   -- Working with single image
   k = 1
   local input = torch.zeros(1, h, w)
   local target = torch.zeros(1, h, w)
   local inputs = {1, 0, -1}
   local targets = {1, 0, -1}
   for _,i in pairs(inputs) do
      for _,t in pairs(targets) do

      input:fill(i)
      target:fill(t)
      -- Check forward
      local loss = crit:updateOutput(input, target)
      local myLoss = math.log(1+math.exp(-1*i*t))/2
      mytester:assert( loss >= myLoss-precision and loss <= myLoss+precision,
                       "SpatialBinaryLogisticRegression cost incorrect.")

      -- Check backward
      local gradInput = crit:updateGradInput(input, target)
      local g1 = gradInput[1][1][1]
      local gi = (1/(1+math.exp(-1*i*t)))*math.exp(-1*i*t)*(-1*t)/(2*k*h*w)
      mytester:assert( g1 >= gi-precision and g1 <= gi+precision,
                      "SpatialBinaryLogisticRegression gradInput error.")
      end
   end
end

-- Unit Test BinaryLogisticRegression criterion
function dpnntest.BinaryLogisticRegression()
   local crit = nn.BinaryLogisticRegression()
   local k = 32

   -- Working with batch of images
   local input = torch.zeros(k, 1)
   local target = torch.zeros(k, 1)
   local inputs = {1, 0, -1}
   local targets = {1, 0, -1}
   for _,i in pairs(inputs) do
      for _,t in pairs(targets) do

      input:fill(i)
      target:fill(t)
      -- Check forward
      local loss = crit:updateOutput(input, target)
      local myLoss = math.log(1+math.exp(-1*i*t))
      mytester:assert( loss >= myLoss-precision and loss <= myLoss+precision,
                       "BinaryLogisticRegression cost incorrect.")

      -- Check backward
      local gradInput = crit:updateGradInput(input, target)
      local g1 = gradInput[1][1]
      local gi = (1/(1+math.exp(-1*i*t)))*math.exp(-1*i*t)*(-1*t)/(k)
      mytester:assert( g1 >= gi-precision and g1 <= gi+precision,
                      "BinaryLogisticRegression gradInput error.")
      end
   end

   -- Working nElements not matching.
   local input = torch.zeros(1, k)
   local target = torch.zeros(k, 1)
   local inputs = {1, 0, -1}
   local targets = {1, 0, -1}
   for _,i in pairs(inputs) do
      for _,t in pairs(targets) do

      input:fill(i)
      target:fill(t)
      -- Check forward
      local loss = crit:updateOutput(input, target)
      local myLoss = math.log(1+math.exp(-1*i*t))
      mytester:assert( loss >= myLoss-precision and loss <= myLoss+precision,
                       "BinaryLogisticRegression cost incorrect.")

      -- Check backward
      local gradInput = crit:updateGradInput(input, target)
      local g1 = gradInput[1][1]
      local gi = (1/(1+math.exp(-1*i*t)))*math.exp(-1*i*t)*(-1*t)/(k)
      mytester:assert( g1 >= gi-precision and g1 <= gi+precision,
                      "BinaryLogisticRegression gradInput error.")
      end
   end
end

-- Unit Test SpatialRegionDropout
function dpnntest.SpatialRegionDropout()
   local hasCuda = pcall(function() require 'cunn' end)
   local useCudas = {false, hasCuda}
   local p = 0.2
   local value = 2
   local model = nn.SpatialRegionDropout(p)
   local input = torch.zeros(3, 100, 100):fill(value)

   for _, useCuda in pairs(useCudas) do
      if useCuda then
         model:cuda()
         input = input:cuda()
      end
      local output = model:forward(input)
      mytester:assert( output:mean() >= value-precision and
                       output:mean() <= value+precision,
                       "SpatialRegionDropout forward mean value incorrect.")

      local gradInput = model:backward(input, input)
      mytester:assert( gradInput:mean() >= value-precision and
                       gradInput:mean() <= value+precision,
                       "SpatialRegionDropout backward mean value incorrect.")
   end
end

-- Unit Test Kmeans layer
function dpnnbigtest.Kmeans()
   local k = 10
   local dim = 5
   local batchSize = 1000
   local input = torch.rand(batchSize, dim)
   for i=1, batchSize do
      input[i]:fill(torch.random(1, k))
   end
   
   local verbose = false

   local attempts = 10
   local iter = 100
   local bestLoss = 100000000
   local bestKm = nil
   local tempLoss = 0
   local learningRate = 1

   local initTypes = {'random', 'kmeans++'}
   local hasCuda = pcall(function() require 'cunn' end)
   local useCudas = {false, hasCuda}
   for _, initType in pairs(initTypes) do
      for _, useCuda in pairs(useCudas) do 

         sys.tic()
         for j=1, attempts do
            local km = nn.Kmeans(k, dim)

            if initType == 'kmeans++' then
               km:initKmeansPlus(input)
            else
               km:initRandom(input)
            end
            
            if useCuda then km:cuda() end
            for i=1, iter do
               km:zeroGradParameters()
               
               km:forward(input)
               km:backward(input, gradOutput)

               -- Gradient descent
               km.weight:add(-learningRate, km.gradWeight)
               tempLoss = km.loss
            end
            if verbose then print("Attempt Loss " .. j ..": " .. tempLoss) end
            if tempLoss < bestLoss then
               bestLoss = tempLoss
            end
         end
         if verbose then
            print("InitType: " .. initType .. " useCuda: " .. tostring(useCuda))
            print("Best Loss: " .. bestLoss)
            print("Total time: " .. sys.toc())
         end
         if initType == 'kmeans++' then
            mytester:assert(bestLoss < 0.00001)
         else
            mytester:assert(bestLoss < 500)
         end
      end
   end
end

-- Unit Test FireModule
function dpnntest.FireModule()
   local hasCuda = pcall(function() require 'cunn' end)
   local useCudas = {false, hasCuda}
   local activations = {'ReLU', 'Tanh', 'Sigmoid'}
   local nInputPlane = 3
   local width = 32
   local height = 32
   local s1x1 = 16
   local e1x1 = 16
   local e3x3 = 16
   for _, activation in pairs(activations) do
      for _, useCuda in pairs(useCudas) do
         local model = nn.FireModule(nInputPlane, s1x1, e1x1, e3x3)
         local input = torch.rand(1, nInputPlane, height, width)
         if useCuda then
            model:cuda()
            input = input:cuda()
         end
         local output = model:forward(input)
         local gradInput = model:backward(input, output)
      end
   end
end

-- Unit Test SpatialFeatNormalization
function dpnntest.SpatialFeatNormalization()
   local hasCuda = pcall(function() require 'cunn' end)
   local useCudas = {false, hasCuda}
   local input = torch.zeros(3, 32, 32):fill(2)
   local mean = torch.zeros(3):fill(1)
   local std = torch.zeros(3):fill(0.5)
   local outputValue = 2
   local gradValue = 4
   for _, useCuda in pairs(useCudas) do
      local model = nn.SpatialFeatNormalization(mean, std)
      if useCuda then
         model:cuda()
         input = input:cuda()
      end
      local output = model:forward(input)
      local gradInput = model:backward(input, output)
      mytester:assert( output:mean() == outputValue,
                     "SpatialFeatNormalization forward mean value incorrect.")
      mytester:assert( gradInput:mean() == gradValue,
                     "SpatialFeatNormalization backward mean value incorrect.")
   end
end

function dpnntest.OneHot()
   local nClass = 10
   
   -- batch mode
   local batchSize = 3
   local input = torch.LongTensor(batchSize):random(1, nClass)
   local gradOutput = torch.randn(batchSize, nClass)
   
   local oh = nn.OneHot(nClass)

   local output = oh:forward(input)
   local output2 = torch.Tensor(batchSize, nClass):zero()
   local eye = torch.eye(nClass)
   output2:index(eye, 1, input)
   mytester:assertTensorEq(output, output2, 0.000001, "OneHot forward batch err")
   mytester:assert(output:dim() == 2)

   local gradInput = oh:backward(input, gradOutput)
   mytester:assertTensorEq(gradInput, input:double():zero(), 0.000001, "OneHot backward batch err")

   if pcall(function() require 'cunn' end) then
      oh:cuda()
      
      -- test with long input
      local output = oh:forward(input)
      mytester:assert(torch.type(output) == 'torch.CudaTensor')
      mytester:assertTensorEq(output:double(), output2, 0.000001, "OneHot forward batch long-cuda err")
      
      -- test with cuda input
      local input = input:cuda()
      gradOutput = gradOutput:cuda()
      
      local output = oh:forward(input)
      mytester:assert(torch.type(output) == 'torch.CudaTensor')
      mytester:assertTensorEq(output:double(), output2, 0.000001, "OneHot forward batch cuda err")
      
      local gradInput2 = oh:backward(input, gradOutput)
      mytester:assertTensorEq(gradInput, gradInput2:double(), 0.000001, "OneHot backward batch err")
      cutorch.synchronize()
   end
   
   -- multi-dimensional input
   local inputSize = 2
   local input = torch.LongTensor(batchSize, inputSize):random(1, nClass)
   local gradOutput = torch.randn(batchSize, inputSize, nClass)
   
   local oh = nn.OneHot(nClass, 2)
   
   local output = oh:forward(input)
   local output2 = torch.Tensor(batchSize*inputSize, nClass):zero()
   local eye = torch.eye(nClass)
   output2:index(eye, 1, input:view(-1))
   output2:resize(batchSize, inputSize, nClass)
   mytester:assertTensorEq(output, output2, 0.000001, "OneHot 2d forward batch err")
   mytester:assert(output:dim() == 3)

   local gradInput = oh:backward(input, gradOutput)
   mytester:assertTensorEq(gradInput, input:double():zero(), 0.000001, "OneHot 2d backward batch err")
   
   if pcall(function() require 'cunn' end) then
      oh:cuda()
      
      -- test with long input
      local output = oh:forward(input)
      mytester:assert(torch.type(output) == 'torch.CudaTensor')
      mytester:assertTensorEq(output:double(), output2, 0.000001, "OneHot 2d forward batch long-cuda err")
      
      -- test with cuda input
      local input = input:cuda()
      gradOutput = gradOutput:cuda()
     
      local output = oh:forward(input)
      mytester:assert(torch.type(output) == 'torch.CudaTensor')
      mytester:assertTensorEq(output:double(), output2, 0.000001, "OneHot 2d forward batch cuda err")
      
      local gradInput2 = oh:backward(input, gradOutput)
      mytester:assertTensorEq(gradInput, gradInput2:double(), 0.000001, "OneHot 2d backward batch err")
      
      local benchmark = false
      if benchmark then
         local input = torch.FloatTensor(50, 50):random(1,65):cuda()
         
         local oh = nn.OneHot(65):cuda()
         
         oh:forward(input)
         cutorch.synchronize()
         local a = torch.Timer()
         for i=1,10 do
            oh:forward(input)
         end
         cutorch.synchronize()
         local gputime = a:time().real
        
         oh:float()
         input = input:float()
         oh:forward(input)
         a = torch.Timer()
         for i=1,10 do
            oh:forward(input)
         end
         local cputime = a:time().real
         print("Onehot GPU vs CPU time", gputime, cputime)
      end
   end
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
