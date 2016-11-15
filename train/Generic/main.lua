require 'nn'
require 'cunn'
require 'torch'
require 'cutorch'
require 'optim'
require '../../lib/function'

npy4th = require 'npy4th'
model = require 'model'
model_2 = require 'model_2'
data = require 'data'
settings = require 'settings'

-- Parser
--
defaults = {
  epochs = 100,
  lr = 0.005,
  filt_size = 11,
  batch_size = 4,
  patch_size = 64,
  crit = 'Abs',
  act_function = 'ReLU',
  nb_layers = 3,
  nstates = '1x32x64x128x2048',
  fc = 'true',
  save_net = '../../Network/generic.bin',
  dropout = 0.1,
  nb_conv = 3
}

cmd = torch.CmdLine()

cmd:option('-epochs', defaults.epochs, 'number of epochs')
cmd:option('-lr', defaults.lr, 'learning rate')
cmd:option('-filt_size', defaults.filt_size, 'filter size')
cmd:option('-batch_size', defaults.batch_size, 'batch size')
cmd:option('-patch_size', defaults.patch_size, 'patch size')
cmd:option('-crit', defaults.crit, 'type of criterion for the training (MSE, Abs, Smooth, Dist)')
cmd:option('-act_function', defaults.act_function, 'type of activation function for the network (ReLU, ReLU6, PReLU, RReLU, ELU, LeakyReLU)')
cmd:option('-nb_layers', defaults.nb_layers, 'nb layers of the network')
cmd:option('-nstates', defaults.nstates, 'differents states at each layer')
cmd:option('-fc', defaults.fc, 'fully connected layers, true or false')
cmd:option('-save_net', defaults.save_net, 'Folder to save network')
cmd:option('-dropout', defaults.dropout, 'probability of dropout')
cmd:option('-nb_conv', defaults.nb_conv, 'nb conv per layer')


opt = cmd:parse(arg)

print(opt)


-- Parameters 
--
epochs = opt.epochs
lr = tonumber(opt.lr)
filt_size = opt.filt_size
batch_size = opt.batch_size
patch_size = opt.patch_size
crit = opt.crit
act_function = opt.act_function
nb_layers = opt.nb_layers
dropout = opt.dropout
nb_conv = opt.nb_conv

nstates = opt.nstates
nstates = split(nstates, "x")
for k,v in ipairs(nstates) do
  nstates[k] = tonumber(v)
end
      
fc = opt.fc
if fc == 'false' then fc = false
else fc = true end

save_net = opt.save_net

pad = torch.floor((filt_size -1)/2)
optimState = {
   learningRate = lr,
   learningRateDecay = 1e-4,
   weightDecay = 1e-3,
   momentum = 1e-4
}


-- Load data
--
train_start = npy4th.loadnpy('../../Data/all/train_dataset_start.npy')
train_end = npy4th.loadnpy('../../Data/all/train_dataset_end.npy')

valid_start = npy4th.loadnpy('../../Data/all/valid_dataset_start.npy')
valid_end = npy4th.loadnpy('../../Data/all/valid_dataset_end.npy')

train_size = train_start:size()[1]
valid_size = valid_start:size()[1]


-- Create model 
--
if nb_layers == 3 then 
  mlp = model.create_model(filt_size, pad, nstates, act_function, fc, dropout)
else 
  mlp = model_2.create_model(filt_size, pad, nstates, nb_layers, act_function, fc, patch_size, dropout, nb_conv)
end

print(mlp)

criterion = settings.get_crit(crit)


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
  local shuffle = torch.randperm(train_size)

  for t = 0, train_size - 1, batch_size do
    batchInputs, batchOutputs = data.create_batch_tensor(train_start, train_end, batch_size, t, shuffle, false)

    local feval = function(x)
      collectgarbage()
      if x ~= params then
        params:copy(x)
      end
      gradParams:zero()

      -- Perform gradient descent
      local outputs = mlp:forward(batchInputs)
      local loss = criterion:forward(outputs, batchOutputs)
      local dloss_doutputs = criterion:backward(outputs, batchOutputs)
      mlp:backward(batchInputs, dloss_doutputs)

      return loss, gradParams
    end
    
    _, fs = optim.sgd(feval, params, optimState)
    --_, fs = optim.asgd(feval, params)
    --_, fs = optim.cg(feval, params)
    --_, fs = optim.rmsprop(feval, params, optimState)
    
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
  local shuffle = torch.randperm(data_size)

  for i = 0, data_size-1, batch_size  do
    validInputs, validOutputs = data.create_batch_tensor(data_start, data_end, batch_size, i, shuffle, false)     
 
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
    local accuracy = eval(valid_start, valid_end, 1, false)

    green = string.char(27) .. '[32m'
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


torch.save(save_net, mlp)
print(green .. 'network ' .. save_net .. ' correctly saved ')

