require 'nn'
require 'cunn'
require 'torch'
require 'cutorch'
require 'optim'

npy4th = require 'npy4th'
model = require 'model'
data = require 'data'


-- Load data
--
train_start = npy4th.loadnpy('../../Data/all/train_dataset_start.npy')
train_end = npy4th.loadnpy('../../Data/all/train_dataset_end.npy')

valid_start = npy4th.loadnpy('../../Data/all/valid_dataset_start.npy')
valid_end = npy4th.loadnpy('../../Data/all/valid_dataset_end.npy')

train_size = train_start:size()[1]
valid_size = valid_start:size()[1]

-- Parameters 
--
filt_size = 11
pad = torch.floor((filt_size -1)/2)

epochs = 100
patch_size = 64

local optimState = {learningRate = 0.005}

-- Create model 
--
mlp = model.create_model(filt_size, pad)
print(mlp)

criterion = nn.AbsCriterion()

-- Transfer to gpu 
--
criterion = criterion:cuda()
mlp = mlp:cuda()

-- Training 
--
params, gradParams = mlp:getParameters()

-- One step (for one batch of batch_size images)
step = function(batch_size)
  local current_loss = 0
  local count = 0
  batch_size = batch_size or 4
  --local shuffle = torch.randperm(train_size)

  for t = 0, train_size - 1, batch_size do
    batchInputs, batchOutputs = data.create_batch_tensor(train_start, train_end, batch_size, t)

    local feval = function(params)
      gradParams:zero()

      -- Perform gradient descent
      local outputs = mlp:forward(batchInputs)
      local loss = criterion:forward(outputs, batchOutputs)
      local dloss_doutputs = criterion:backward(outputs, batchOutputs)
      mlp:backward(batchInputs, dloss_doutputs)

      return loss, gradParams
    end
    
    _, fs = optim.sgd(feval, params, optimState)
    count = count + 1
    current_loss = current_loss + fs[1]
  end 
 
  -- Normalize loss
  return current_loss/count
end


-- Compute accuracy
--
eval = function(data_start, data_end, batch_size)
  local acc = 0
  local data_size = data_start:size()[1]

  for i = 0, data_size-1, batch_size  do
    validInputs, validOutputs = data.create_batch_tensor(data_start, data_end, batch_size, i)     
 
    local outputs = mlp:forward(validInputs)
    local sum_true = torch.lt(torch.abs(torch.add(outputs, -validOutputs)), 1e-2):sum()
    acc = acc + sum_true/(patch_size*patch_size)
  end
 
  return acc/data_size
end


 do
  local last_accuracy = 0
  local decreasing = 0
  local threshold = 9

  for e = 1, epochs, 1 do
    local time = sys.clock()
    local loss = step()
    print('Epoch: ' .. e .. ', Current loss : ' .. loss)
    local accuracy = eval(valid_start, valid_end, 1)

    local green = string.char(27) .. '[32m'
    print('Accuracy on the validation dataset : ' .. green  .. accuracy)
    if accuracy < last_accuracy then
      if decreasing > threshold then break end
        decreasing = decreasing + 1
      else
        decreasing = 0
      end
      last_accuracy = accuracy
    end
end


torch.save('../../Network/optim.bin', mlp)
print('network optim.bin correctly saved ')

