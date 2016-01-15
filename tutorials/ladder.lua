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
op:option{'-d', '--datadir', action='store', dest='datadir',
          help='path to datadir', default=""}
op:option{'--noValidation', action='store_true', dest='noValidation',
          help='Use validation data for training as well.', default=false}

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
op:option{'--eta', action='store', dest='eta',
          help='If zero then only classifier cost is considered.', default=0}
op:option{'--batchSize', action='store', dest='batchSize',
          help='Batch Size.',default=32}
op:option{'--epochs', action='store', dest='epochs',
          help='Number of epochs.',default=10000}
op:option{'--maxTries', action='store', dest='maxTries',
          help='Number of tries for stopping.',default=100}
op:option{'--learningRate', action='store', dest='learningRate',
          help='Learning rate',default=0.002}
op:option{'--learningRateDecay', action='store', dest='learningRateDecay',
          help='Learning rate decay',default=1e-7}
op:option{'--momentum', action='store', dest='momentum',
          help='Learning Momemtum',default=0}
op:option{'--loss', action='store_true', dest='loss',
          help='If true use loss for early stopping else confusion matrix.',
          default=false}
op:option{'--adam', action='store_true', dest='adam',
          help='Use adaptive moment estimation optimizer.', default=false}

-- Use Cuda
op:option{'--useCuda', action='store_true',
          dest='useCuda', help='Use GPU', default=false}
op:option{'--deviceId', action='store', dest='deviceId',
          help='GPU device Id',default=2}

-- Print debug messages
op:option{'--verbose', action='store_true', dest='verbose',
          help='Print apppropriate debug messages.', default=false}

-- Command line arguments
opt = op:parse()
op:summarize()

-- Data
datadir = opt.datadir

trDataFile = paths.concat(datadir, "trainDict.t7")
tvDataFile = paths.concat(datadir, "validDict.t7")
tsDataFile = paths.concat(datadir, "testDict.t7")

trData = torch.load(trDataFile)
trData.size = function() return trData.data:size()[1] end
tvData = torch.load(tvDataFile)
tvData.size = function() return tvData.data:size()[1] end
tsData = torch.load(tsDataFile)
tsData.size = function() return tsData.data:size()[1] end

tempSample = trData.data[1]
channels = tempSample:size(1)
width = tempSample:size(2)
height = tempSample:size(3)
linFeats = channels * height * width

-- MNIST
classes = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'}
confusion = optim.ConfusionMatrix(classes)

-- Model
noOfClasses = tonumber(opt.noOfClasses)
noiseSigma = tonumber(opt.noiseSigma)
inputHiddens = dp.returnString(opt.hiddens)
useBatchNorm = opt.useBatchNorm
weightTied = opt.weightTied

verbose = opt.verbose

hiddens = {linFeats}
for i=1,#inputHiddens do
   hiddens[#hiddens+1] = inputHiddens[i]
end
hiddens[#hiddens+1] = noOfClasses

-- encoder input
if noiseSigma ~= 0 then
   if verbose then print("Add noise to the samples.") end
   input = nn.WhiteNoise(0, noiseSigma)()
else
   input = nn.Identity()()
end

-- encoder model
encoderLayers = {}
Zs = {}
Zs[0] = input
for i=1,#hiddens-1 do
   encoderLayers[i] = nn.Linear(hiddens[i], hiddens[i+1])
   if i==1 then
      if useBatchNorm then
         Zs[i] = nn.BatchNormalization(hiddens[i+1])(encoderLayers[i](input))
      else
         Zs[i] = encoderLayers[i](input)
      end
   else
      if useBatchNorm then
         Zs[i] = nn.BatchNormalization(hiddens[i+1])
                                      (encoderLayers[i](nn.ReLU()(Zs[i-1])))
      else
         Zs[i] = encoderLayers[i](nn.ReLU()(Zs[i-1]))
      end
   end
end

-- classifier
classifier = nn.LogSoftMax()(Zs[#Zs])

-- Decoder
decoderLayers = {}
Z_hats = {}
Z_hats[#hiddens+1] = Zs[#Zs]
for i=#hiddens,2,-1 do
   decoderLayers[i] = nn.Linear(hiddens[i], hiddens[i-1])
   if weightTied then
      if verbose then print("Tying encoder-decoder weights.") end
      decoderLayers[i].weight:set(encoderLayers[i-1].weight:t())
      decoderLayers[i].gradWeight:set(encoderLayers[i-1].gradWeight:t())
   end
   if useBatchNorm then
      u = nn.ReLU()(nn.BatchNormalization(hiddens[i-1])
                                         (decoderLayers[i](Z_hats[i+1])))
   else
      u = nn.ReLU()(decoderLayers[i](Z_hats[i+1]))
   end

   cu1 = nn.CMul(hiddens[i-1])(u)
   du1 = nn.Add(hiddens[i-1])(u)
   a1 = nn.CAddTable()({cu1, du1})
   cu2 = nn.CMul(hiddens[i-1])(u)
   du2 = nn.Add(hiddens[i-1])(u)
   a2 = nn.CAddTable()({cu2, du2})
   cu3 = nn.CMul(hiddens[i-1])(u)
   du3 = nn.Add(hiddens[i-1])(u)
   a3 = nn.CAddTable()({cu3, du3})
   cu4 = nn.CMul(hiddens[i-1])(u)
   du4 = nn.Add(hiddens[i-1])(u)
   a4 = nn.CAddTable()({cu4, du4})
   cu5 = nn.CMul(hiddens[i-1])(u)
   du5 = nn.Add(hiddens[i-1])(u)
   a5 = nn.CAddTable()({cu5, du5})

   z_hat1 = nn.CMulTable()({a1, Zs[i-2]})
   z_hat2 = nn.CMulTable()({a3, Zs[i-2]})
   z_hat3 = nn.Sigmoid()(nn.CAddTable()({z_hat2, a4}))
   z_hat4 = nn.CMulTable()({a2, z_hat3})
   Z_hats[i] = nn.CAddTable()({z_hat1, z_hat4, a5})
end
model = nn.gModule({input}, {classifier, Z_hats[2]--[[Decoder--]]})
print(model)

-- Criterion and learning
-- Criterion
eta = tonumber(opt.eta)
criterions = nn.ParallelCriterion()
nll = nn.ClassNLLCriterion()
mse = nn.MSECriterion()
criterions:add(nll)
criterions:add(mse, eta)

-- Learning
batchSize = tonumber(opt.batchSize)
epochs = tonumber(opt.epochs)
maxTries = tonumber(opt.maxTries)
learningRate = tonumber(opt.learningRate)
learningRateDecay = tonumber(opt.learningRateDecay)
momentum = tonumber(opt.momentum)
loss = opt.loss
adam = opt.adam

-- Optimizer
optimState = {
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
print(optimState)

-- Cuda
useCuda = opt.useCuda
deviceId = tonumber(opt.deviceId)

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

if verbose then
   print(trData)
   print(tvData)
   print(tsData)
end

-- Training
displayProgress = verbose
classifierIndx = 1
trainAccu = 0
validAccu = 0
bestTrainAccu = 0
bestValidAccu = 0
trainLoss = 0
validLoss = 0
bestTrainLoss = math.huge
bestValidLoss = math.huge
bestTrainModel = nn.Sequential()
bestValidModel = nn.Sequential()
earlyStopCount = 0
for i=1, epochs do
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
   print(noiseSigma, weightTied, useBatchNorm, eta, earlyStopCount)

   if earlyStopCount >= maxTries then
      if verbose then print("Early stopping at epoch: " .. i) end
      break
   end
end

-- Testing
testLoss = model_test_multi_criterion(bestValidModel, criterions,
                                      tsData, confusion,
                                      useCuda, classifierIndx)
print("Testing Loss: " .. testLoss)
confusion:updateValids()
testAccu = confusion.totalValid * 100
print("Testing confusion using best validation model.")
print(confusion)
print(noiseSigma, weightTied, useBatchNorm, eta)
