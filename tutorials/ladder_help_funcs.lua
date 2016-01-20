require 'csvigo'
require 'string'
require 'xlua'
require 'lfs'

-- Training function test
-- Processing a batch in one Go.
-- Has useCuda option to run on GPU [model and criterion expected in CUDA]
local conTargets, conOutputs
function model_train_multi_criterion(model, criterions, parameters,
                                     gradParameters, trainData, 
                                     optimMethod, optimState, batchSize,
                                     epoch, confusion, trainLogger,
                                     useCuda, displayProgress, classifierIndx)

   model:training()
   confusion:zero()
   local displayProgress = displayProgress or false
   local classifierIndx = classifierIndx or 1

   -- epoch tracker
   local epoch = epoch or 1

   local totalLoss = 0
   
   -- shuffle at each epoch
   local shuffle = torch.randperm(trainData.size())

   local sampleSize = trainData.data[1]:size()
   local isScalar = false
   local labelSize
   if trainData.labels:size():size() == 1 then
      isScalar = true
   else
      labelSize = trainData.labels[1]:size()
   end

   print("Doing epoch on training data:")
   print("Online epoch # " .. epoch .. " [batchSize = " .. batchSize .. "]")

   -- local variables
   local time = sys.clock()
   local inputs
   local targets
   if isScalar then
      targets = torch.Tensor(batchSize)
   else
      targets = torch.Tensor(batchSize, labelSize[1])
   end

   -- Samples
   sizeLen = sampleSize:size()
   if sizeLen == 1 then
      inputs = torch.Tensor(batchSize, sampleSize[1])
   elseif sizeLen == 2 then
      inputs = torch.Tensor(batchSize, sampleSize[1], sampleSize[2])
   elseif sizeLen == 3 then
      inputs = torch.Tensor(batchSize, sampleSize[1], sampleSize[2],
                                       sampleSize[3])
   else
      print("Invalid Sample Size")
   end

   local trainInputs = useCuda and torch.CudaTensor() or torch.FloatTensor()
   local trainTargets = useCuda and torch.CudaTensor() or torch.FloatTensor()
   local criterionTargets

   t = 1
   while t <= trainData.size() do
      if displayProgress then xlua.progress(t, trainData.size()) end
      noOfSamples = math.min(t + batchSize -1, trainData.size())
      --create mini batch
      indx = 1 
      for i=t,math.min(t+batchSize-1, trainData.size()) do
         -- Load new sample
         inputs[indx] = trainData.data[shuffle[i]]
         targets[indx] = trainData.labels[shuffle[i]]
         indx = indx + 1
      end
      indx = indx - 1

      local inputs_ = inputs[{{1,indx}}]
      trainInputs:resize(inputs_:size()):copy(inputs_)

      local targets_ = targets[{{1,indx}}]
      trainTargets:resize(targets_:size()):copy(targets_)

      criterionTargets = {trainTargets, trainInputs}

      t = t + batchSize

      -- create closure to evaluate F(X) and df/dX
      local feval = function(x)
         -- Get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = model:forward(trainInputs)
         local f = criterions:forward(outputs, criterionTargets)
         -- Total Loss
         totalLoss = totalLoss + f

         local df_do = criterions:backward(outputs, criterionTargets)
         model:backward(trainInputs, df_do)

         if useCuda then
            conOutputs = outputs[classifierIndx]:float()
            conTargets = trainTargets:float()
         else
            conOutputs = outputs[classifierIndx]
            conTargets = trainTargets
         end

         confusion:batchAdd(conOutputs, conTargets)

         -- Normalize gradients
         gradParameters:div(trainInputs:size()[1])
         f = f/trainInputs:size()[1]

         -- L1/L2 Regularization
         if optimState.coefL1 ~= 0 or optimState.coefL2 ~= 0 then
            -- locals"
            local norm, sign = torch.norm, torch.sign
         
            -- Update loss with regularizer
            f = f + optimState.coefL1 * norm(parameters, 1)
            f = f + optimState.coefL2 * norm(parameters, 2)^2/2

            -- Gradients
            gradParameters:add(sign(parameters):mul(optimState.coefL1)
                               + parameters:clone():mul(opt.coefL2))
         end

         -- return f and df/dX
         return f, gradParameters
      end

      -- optimize on current mini batch # Using SGD/adam
      optimMethod(feval, parameters, optimState)
   end

   -- time taken
   time = sys.clock() - time
   time = time/trainData.size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. "ms")  

   -- Total loss
   totalLoss = totalLoss/trainData.size()

   -- update logger
   if trainLogger ~= nil then
      trainLogger:add{["% mean class accuracy (train set)"] =
                      confusion.totalValid * 100}
   end
   return totalLoss
end

function model_test_multi_criterion(model, criterions, testData, confusion, 
                                    useCuda, classifierIndx)
   local time = sys.clock()
   model:evaluate()
   confusion:zero()
   local classifierIndx = classifierIndx or 1
   local totalLoss = 0
   local criterionTargets

   if useCuda then
      local batchSize = 64
      local inputs = torch.CudaTensor()
      local testInputs
      local cpu_targets
      local gpu_targets = torch.CudaTensor()
      local gpu_preds
      local cpu_preds
      local i = 1
      local j = 0
      while i <= testData.size() do
         j = math.min(i + batchSize -1, testData.size())
         -- Copy input and targets to cuda
         testInputs = testData.data[{{i, j}}]
         inputs:resize(testInputs:size()):copy(testInputs)
         cpu_targets = testData.labels[{{i, j}}]
         gpu_targets:resize(cpu_targets:size()):copy(cpu_targets)
         criterionTargets = {gpu_targets, inputs}

         gpu_preds = model:forward(inputs)
         totalLoss = totalLoss + criterions:forward(gpu_preds,
                                                    criterionTargets)
         cpu_preds = gpu_preds[classifierIndx]:float()
         confusion:batchAdd(cpu_preds, cpu_targets)
         i = i + batchSize
      end
   else
      local trainInputs = testData.data
      local trainTargets = testData.labels
      criterionTargets = {trainTargets, trainInputs}

      local outputs = model:forward(trainInputs)
      totalLoss = criterions:forward(outputs, criterionTargets)

      local conOutputs = outputs[classifierIndx]
      local conTargets = trainTargets
      confusion:batchAdd(conOutputs, conTargets)
   end

   -- time taken
   time = sys.clock() - time
   time = time/testData.size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. "ms")

   -- Total loss
   totalLoss = totalLoss/testData.size()

   return totalLoss
end
