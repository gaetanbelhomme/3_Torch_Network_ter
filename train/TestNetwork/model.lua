require 'nn'

model = {}

function model.create_model(filt_size, pad)
  mlp = nn.Sequential();

  mlp:add(nn.SpatialConvolutionMM(1, 1, filt_size, filt_size, 1,1, pad, pad))
  mlp:add(nn.ReLU())

  return mlp

end

return model 
