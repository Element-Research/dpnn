--[[!
   Implementation of ladder as mentioned in http://arxiv.org/pdf/1504.08215.pdf
--]]

require 'nn'
require 'dp'
require 'dpnn'
require 'math'
require 'xlua'
require 'optim'
require 'nngraph'

-- Cuda
require 'cutorch'
require 'cunn'

-- Help functions
require 'ladder_help_funcs'

torch.setdefaulttensortype("torch.FloatTensor")
op = xlua.OptionParser('%prog [options]')

-- Data
op:option{'--noValidation', action='store_true', dest='noValidation',
          help='Use validation data for training as well.', default=false}
op:option{'--best', action='store_true', dest='best',
          help='Use best training or validation model.', default=false}

-- Model parameters
op:option{'--noOfClasses', action='store', dest='noOfClasses',
          help='Number of classes.', default=10} -- MNIST data
op:option{'--noiseSigma', action='store', dest='noiseSigma',
          help='Stdev for noise for denoising autoencoder (Mean is zero).',
          default=0}
op:option{'--hiddens', action='store', dest='hiddens',
          help='Hiddens units', default='{1000, 500, 250, 250, 250}'}
op:option{'--useBatchNorm', action='store_true', dest='useBatchNorm',
          help='Use batch normalization.', default=false}
op:option{'--weightTied', action='store_true', dest='weightTied',
          help='Tie weights of decoder with encoder.', default=false}

-- Criterion and learning
op:option{'--attempts', action='store', dest='attempts',
          help='Run attempts independent experiments.', default=1}
op:option{'--eta', action='store', dest='eta',
          help='If zero then only classifier cost is considered.', default=0}
op:option{'--batchSize', action='store', dest='batchSize',
          help='Batch Size.',default=32}
op:option{'--epochs', action='store', dest='epochs',
          help='Number of epochs.',default=100}
op:option{'--maxTries', action='store', dest='maxTries',
          help='Number of tries for stopping.',default=0}
op:option{'--learningRate', action='store', dest='learningRate',
          help='Learning rate',default=0.002}
op:option{'--learningRateDecay', action='store', dest='learningRateDecay',
          help='Learning rate decay',default=1e-7}
op:option{'--linearDecay', action='store_true', dest='linearDecay',
          help='Linearly reduce learning rate', default=false}
op:option{'--startEpoch', action='store', dest='startEpoch',
          help='Epoch number when to start linear decay.',default=1}
op:option{'--endLearningRate', action='store', dest='endLearningRate',
          help='Learning rate at last epoch',default=0.0}
op:option{'--momentum', action='store', dest='momentum',
          help='Learning Momemtum',default=0}
op:option{'--loss', action='store_true', dest='loss',
          help='If true use loss for early stopping else confusion matrix.',
          default=false}
op:option{'--adam', action='store_true', dest='adam',
          help='Use adaptive moment estimation optimizer.', default=false}

-- Use Cuda
op:option{'--useCuda', action='store_true', dest='useCuda', help='Use GPU',
          default=false}
op:option{'--deviceId', action='store', dest='deviceId', help='GPU device Id',
          default=2}

-- Print debug messages
op:option{'--verbose', action='store_true', dest='verbose',
          help='Print apppropriate debug messages.', default=false}

-- Command line arguments
opt = op:parse()
op:summarize()

-- Data
noValidation = opt.noValidation
best = opt.best
verbose = opt.verbose

   -- Cuda
useCuda = opt.useCuda
deviceId = tonumber(opt.deviceId)

-- MNIST Data source
ds = dp.Mnist{}

attempts = tonumber(opt.attempts)
testAccus = torch.zeros(attempts)
trData = {}
tvData = {}
tsData = {}
for attempt=1,attempts do

   local t1, t2

   trData.data, t1, t2 = ds:get('train', 'input', 'bchw', 'float')
   trData.labels, t1, t2 = ds:get('train', 'target')
   trData.size = function() return trData.data:size()[1] end

   tvData.data, t1, t2 = ds:get('valid', 'input', 'bchw', 'float')
   tvData.labels, t1, t2 = ds:get('valid', 'target')
   tvData.size = function() return tvData.data:size()[1] end

   tsData.data, t1, t2 = ds:get('test', 'input', 'bchw', 'float')
   tsData.labels, t1, t2 = ds:get('test', 'target')
   tsData.size = function() return tsData.data:size()[1] end
   collectgarbage()

   local tempSample = trData.data[1]
   local channels = tempSample:size(1)
   local width = tempSample:size(2)
   local height = tempSample:size(3)
   local linFeats = channels * height * width

   -- MNIST
   local classes = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'}
   local confusion = optim.ConfusionMatrix(classes)

   -- Model
   local noOfClasses = tonumber(opt.noOfClasses)
   local noiseSigma = tonumber(opt.noiseSigma)
   local inputHiddens = dp.returnString(opt.hiddens)
   local useBatchNorm = opt.useBatchNorm
   local weightTied = opt.weightTied


   hiddens = {linFeats}
   for i=1,#inputHiddens do
      hiddens[#hiddens+1] = inputHiddens[i]
   end
   hiddens[#hiddens+1] = noOfClasses

   -- encoder input
   local input = nil
   if noiseSigma ~= 0 then
      if verbose then print("Add noise to the samples.") end
      input = nn.WhiteNoise(0, noiseSigma)()
   else
      input = nn.Identity()()
   end

   -- encoder model
   local encoderLayers = {}
   local Zs = {}
   Zs[1] = input
   local Hs = {}
   Hs[1] = input
   for i=2,#hiddens do
      -- Zs
      encoderLayers[i] = nn.Linear(hiddens[i-1], hiddens[i])
      if useBatchNorm then
         Zs[i] = nn.BatchNormalization(hiddens[i])
                                      (encoderLayers[i](Hs[i-1]))
      else
         Zs[i] = encoderLayers[i](Hs[i-1])
      end
     
      -- Hs
      if i==#hiddens then
         Hs[i] = nn.CMul(hiddens[i])(nn.Add(hiddens[i])(Zs[i]))
      else
         Hs[i] = nn.ReLU()(nn.CMul(hiddens[i])(nn.Add(hiddens[i])(Zs[i])))
      end
   end

   -- classifier
   local classifier = nn.LogSoftMax()(Hs[#Hs])

   -- Decoder
   local decoderLayers = {}
   local Z_hats = {}
   for i=#hiddens,1,-1 do

      -- u = 0 hence no cij
      if i==#hiddens then
         z_hat1 = nn.CMul(hiddens[i])(Zs[i])
         z_hat2 = nn.CMul(hiddens[i])(Zs[i])
         z_hat3 = nn.CMul(hiddens[i])(Zs[i])
         z_hat34 = nn.Add(hiddens[i])(z_hat3)
         z_hatSigmoid34 = nn.Sigmoid()(z_hat34)
         z_hat234 = nn.CMulTable()({z_hat2, z_hatSigmoid34})
         z_hat5 = nn.CMul(hiddens[i])(Zs[i])
         Z_hats[i] = nn.CAddTable()({z_hat1, z_hat234, z_hat5})
      else
         decoderLayers[i] = nn.Linear(hiddens[i+1], hiddens[i])
         if weightTied then
            if verbose then print("Tying encoder-decoder weights.") end
            decoderLayers[i].weight:set(encoderLayers[i+1].weight:t())
            decoderLayers[i].gradWeight:set(encoderLayers[i+1].gradWeight:t())
         end

         u = decoderLayers[i](Z_hats[i+1])

         cu1 = nn.CMul(hiddens[i])(u)
         du1 = nn.Add(hiddens[i])(u)
         a1 = nn.CAddTable()({cu1, du1})
         cu2 = nn.CMul(hiddens[i])(u)
         du2 = nn.Add(hiddens[i])(u)
         a2 = nn.CAddTable()({cu2, du2})
         cu3 = nn.CMul(hiddens[i])(u)
         du3 = nn.Add(hiddens[i])(u)
         a3 = nn.CAddTable()({cu3, du3})
         cu4 = nn.CMul(hiddens[i])(u)
         du4 = nn.Add(hiddens[i])(u)
         a4 = nn.CAddTable()({cu4, du4})
         cu5 = nn.CMul(hiddens[i])(u)
         du5 = nn.Add(hiddens[i])(u)
         a5 = nn.CAddTable()({cu5, du5})

         z_hat1 = nn.CMulTable()({a1, Zs[i]})
         z_hat2 = nn.CMulTable()({a3, Zs[i]})
         z_hat3 = nn.Sigmoid()(nn.CAddTable()({z_hat2, a4}))
         z_hat4 = nn.CMulTable()({a2, z_hat3})
         Z_hats[i] = nn.CAddTable()({z_hat1, z_hat4, a5})
      end
   end
   local model = nn.gModule({input}, {classifier, Z_hats[1]--[[Decoder--]]})
   if verbose then print(model) end

   -- Criterion and learning
   -- Criterion
   local eta = tonumber(opt.eta)
   local criterions = nn.ParallelCriterion()
   local nll = nn.ClassNLLCriterion()
   local mse = nn.MSECriterion()
   criterions:add(nll)
   criterions:add(mse, eta)

   -- Learning
   local batchSize = tonumber(opt.batchSize)
   local epochs = tonumber(opt.epochs)
   local maxTries = tonumber(opt.maxTries)
   local learningRate = tonumber(opt.learningRate)
   local learningRateDecay = tonumber(opt.learningRateDecay)
   local linearDecay = opt.linearDecay
   local startEpoch = tonumber(opt.startEpoch)
   local endLearningRate = tonumber(opt.endLearningRate)
   assert(epochs > startEpoch, "startEpoch should be smaller than epochs.")   

   if linearDecay then
      if verbose then print("Using linear decay.") end
      learningRates = torch.zeros(startEpoch):fill(learningRate)
      local temp = torch.range(learningRate, endLearningRate,
                               -learningRate/(epochs-startEpoch))
      learningRates = torch.cat(learningRates, temp)
   end

   local momentum = tonumber(opt.momentum)
   local loss = opt.loss
   local adam = opt.adam

   -- Optimizer
   local optimState = {
                       coefL1 = 0,
                       coefL2 = 0,
                       learningRate = learningRate,
                       weightDecay = 0.0,
                       momentum = momentum,
                       learningRateDecay = learningRateDecay
                      }

   -- If true use Adaptive moment estimation else SGD.
   if adam then
      if verbose then print("Using Adaptive moment estimation optimizer.") end
      optimMethod = optim.adam
   else
      if verbose then print("Using Stocastic gradient descent optimizer.") end
      optimMethod = optim.sgd
   end
   if verbose then
      print(optimMethod)
      print(optimState)
   end


   if useCuda then
      if verbose then print("Using GPU: "..deviceId) end
      cutorch.setDevice(deviceId)
      if verbose then print("GPU set") end
      model:cuda()
      if verbose then print("Model copied to GPU.") end
      criterions:cuda()
      if verbose then print("Criterion copied to GPU.") end
   else
      if verbose then print("Not using GPU.") end
   end

   -- Retrieve parameters and gradients
   parameters, gradParameters = model:getParameters()

   -- Reshape samples from images to vectors
   trData.data = trData.data:reshape(trData.size(1), linFeats)
   tvData.data = tvData.data:reshape(tvData.size(1), linFeats)
   tsData.data = tsData.data:reshape(tsData.size(1), linFeats)
   collectgarbage()

   if noValidation then
      trData.data = torch.cat(trData.data, tvData.data, 1)
      trData.labels = torch.cat(trData.labels, tvData.labels, 1)
      tvData.data = nil
      tvData.labels = nil
      collectgarbage()
   end

   if verbose then
      print(trData)
      print(tvData)
      print(tsData)
   end

   -- Training
   local displayProgress = verbose
   local classifierIndx = 1
   local trainAccu = 0
   local validAccu = 0
   local bestTrainAccu = 0
   local bestValidAccu = 0
   local trainLoss = 0
   local validLoss = 0
   local bestTrainLoss = math.huge
   local bestValidLoss = math.huge
   local bestTrainModel = nn.Sequential()
   local bestValidModel = nn.Sequential()
   local earlyStopCount = 0
   for i=1, epochs do
      if linearDecay then
         optimState.learningRate = learningRates[i]
      end
      -- Training
      trainLoss = model_train_multi_criterion(model, criterions,
                                              parameters, gradParameters, trData,
                                              optimMethod, optimState, batchSize,
                                              i, confusion, trainLogger,
                                              useCuda, displayProgress,
                                              classiferIndx)
      confusion:updateValids()
      if loss then
         if verbose then
            print("Current train loss: ".. trainLoss
                     ..", best train loss: " .. bestTrainLoss)
         end
         if trainLoss < bestTrainLoss then
            bestTrainLoss = trainLoss
            bestTrainModel = model:clone()
            print(confusion)
         end
      else -- Using classification accuracy for saving best train model
         trainAccu = confusion.totalValid * 100
         if bestTrainAccu < trainAccu then
            bestTrainAccu = trainAccu
            bestTrainModel = model:clone()
            bestTrainLoss = trainLoss
         end
         if verbose then
            print("Current train accu: ".. trainAccu
                     ..", best train accu: " .. bestTrainAccu
                     ..", best train loss: " .. bestTrainLoss)
         end
      end

      -- Validating
      if not noValidation then
         validLoss = model_test_multi_criterion(model, criterions,
                                                tvData, confusion,
                                                useCuda, classifierIndx)
         confusion:updateValids()
         if loss then
            if verbose then
               print("Current valid loss: ".. validLoss
                        ..", best valid loss: " .. bestValidLoss)
            end
            if validLoss < bestValidLoss then
               earlyStopCount = 0
               bestValidLoss = validLoss
               bestValidModel = model:clone()
               print(confusion)
            else
               earlyStopCount = earlyStopCount + 1
            end
         else
            validAccu = confusion.totalValid * 100
            if bestValidAccu < validAccu then
               earlyStopCount = 0
               bestValidAccu = validAccu
               bestValidModel = model:clone()
               bestValidLoss = validLoss
            else
               earlyStopCount = earlyStopCount + 1
            end
            if verbose then
               print("Current valid accu: ".. validAccu
                     ..", best valid accu: " .. bestValidAccu
                     ..", best valid loss: " .. bestValidLoss)
            end
         end
         if verbose then
            print(noiseSigma, weightTied, useBatchNorm, eta, earlyStopCount)
         end
      end

      if maxTries ~= 0 then
         if earlyStopCount >= maxTries then
            if verbose then print("Early stopping at epoch: " .. i) end
            break
         end
      end
   end

   -- Testing
   if best then
      if noValidation then
         testLoss = model_test_multi_criterion(bestTrainModel, criterions,
                                               tsData, confusion,
                                               useCuda, classifierIndx)
      else
         testLoss = model_test_multi_criterion(bestValidModel, criterions,
                                               tsData, confusion,
                                               useCuda, classifierIndx)
      end
   else
      testLoss = model_test_multi_criterion(model, criterions,
                                            tsData, confusion,
                                            useCuda, classifierIndx)
   end
   confusion:updateValids()
   testAccu = confusion.totalValid * 100
   testAccus[attempt] = testAccu
   if verbose then
      print("Attempt: " .. tostring(attempt) .. " Test Accu: " .. testAccu)
   end
end
print("Test accuracies.")
print(testAccus)
print("Max Test Error is: " .. tostring(100 - testAccus:max()) .. "%")
