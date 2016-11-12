npy4th = require 'npy4th'
require 'nn'
require 'cunn'
require 'torch'
require 'cutorch'

-- Load data
train_start = npy4th.loadnpy('../../Data/all/train_dataset_start.npy')
train_end = npy4th.loadnpy('../../Data/all/train_dataset_end.npy')


-- Dataset
dataset={};
function dataset:size() return 48 end -- 48 examples
for i=1,dataset:size() do 
  local input = train_start[i]
  local output = train_end[i]

  input = input:cuda()
  output = output:cuda()
  dataset[i] = {input, output}
end

print(dataset:size())



-- Parameters
--
filt_size = 5
pad = torch.floor((filt_size -1)/2)


-- Model 
--
mlp = nn.Sequential();
 
mlp:add(nn.SpatialConvolutionMM(1, 32, filt_size, filt_size, 1, 1, pad, pad))
mlp:add(nn.ReLU())
mlp:add(nn.SpatialConvolutionMM(32, 32, filt_size, filt_size, 1, 1, pad, pad))
mlp:add(nn.ReLU())
mlp:add(nn.SpatialConvolutionMM(32, 32, filt_size, filt_size, 1, 1, pad, pad))
mlp:add(nn.ReLU())

local pool = nn.SpatialMaxPooling(2,2,2,2)
mlp:add(pool)

mlp:add(nn.SpatialConvolutionMM(32, 64, filt_size, filt_size, 1, 1, pad, pad))
mlp:add(nn.ReLU())
mlp:add(nn.SpatialConvolutionMM(64, 64, filt_size, filt_size, 1, 1, pad, pad))
mlp:add(nn.ReLU())
mlp:add(nn.SpatialConvolutionMM(64, 64, filt_size, filt_size, 1, 1, pad, pad))
mlp:add(nn.ReLU())

local pool_2 = nn.SpatialMaxPooling(2,2,2,2)
mlp:add(pool_2)

mlp:add(nn.SpatialConvolutionMM(64, 128, filt_size, filt_size, 1, 1, pad, pad))
mlp:add(nn.ReLU())
mlp:add(nn.SpatialConvolutionMM(128, 128, filt_size, filt_size, 1, 1, pad, pad))
mlp:add(nn.ReLU())
mlp:add(nn.SpatialConvolutionMM(128, 128, filt_size, filt_size, 1, 1, pad, pad))
mlp:add(nn.ReLU())

local pool_3 = nn.SpatialMaxPooling(2,2,2,2)
mlp:add(pool_3)

mlp:add(nn.Reshape(128*8*8))
mlp:add(nn.Dropout(0.5))
mlp:add(nn.Linear(128*8*8, 2048))
mlp:add(nn.Linear(2048, 128*8*8))
mlp:add(nn.Reshape(128, 8, 8))


mlp:add(nn.SpatialMaxUnpooling(pool_3))
mlp:add(nn.SpatialConvolutionMM(128, 128, filt_size, filt_size, 1, 1, pad, pad))
mlp:add(nn.ReLU())
mlp:add(nn.SpatialConvolutionMM(128, 128, filt_size, filt_size, 1, 1, pad, pad))
mlp:add(nn.ReLU())
mlp:add(nn.SpatialConvolutionMM(128, 64, filt_size, filt_size, 1, 1, pad, pad))
mlp:add(nn.ReLU())

mlp:add(nn.SpatialMaxUnpooling(pool_2))
mlp:add(nn.SpatialConvolutionMM(64, 64, filt_size, filt_size, 1, 1, pad, pad))
mlp:add(nn.ReLU())
mlp:add(nn.SpatialConvolutionMM(64, 64, filt_size, filt_size, 1, 1, pad, pad))
mlp:add(nn.ReLU())
mlp:add(nn.SpatialConvolutionMM(64, 32, filt_size, filt_size, 1, 1, pad, pad))
mlp:add(nn.ReLU())

mlp:add(nn.SpatialMaxUnpooling(pool))
mlp:add(nn.SpatialConvolutionMM(32, 32, filt_size, filt_size, 1, 1, pad, pad))
mlp:add(nn.ReLU())
mlp:add(nn.SpatialConvolutionMM(32, 32, filt_size, filt_size, 1, 1, pad, pad))
mlp:add(nn.ReLU())
mlp:add(nn.SpatialConvolutionMM(32, 1, filt_size, filt_size, 1, 1, pad, pad))

criterion = nn.MSECriterion()  

-- Transfer to gpu 
--
criterion = criterion:cuda()
mlp = mlp:cuda()


-- Training
--
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 50
trainer:train(dataset)

torch.save('../../Network/SGD.bin', mlp)
print('network SGD.bin correctly saved ')
