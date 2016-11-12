require 'nn'

model = {}

function model.create_model(filt_size, pad)
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
  mlp:add(nn.Dropout(0.1))
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

  return mlp
end

return model 
