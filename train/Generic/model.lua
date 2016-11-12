require 'nn'

settings = require 'settings'

model = {}

function model.create_model(filt_size, pad, nstates, act, fc, dropout)
  mlp = nn.Sequential();
  
  mlp:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filt_size, filt_size, 1, 1, pad, pad))
  mlp:add(settings.get_act(act))
  mlp:add(nn.SpatialConvolutionMM(nstates[2], nstates[2], filt_size, filt_size, 1, 1, pad, pad))
  mlp:add(settings.get_act(act))
  mlp:add(nn.SpatialConvolutionMM(nstates[2], nstates[2], filt_size, filt_size, 1, 1, pad, pad))
  mlp:add(settings.get_act(act))
  
  local pool = nn.SpatialMaxPooling(2,2,2,2)
  mlp:add(pool)
  
  mlp:add(nn.SpatialConvolutionMM(nstates[2], nstates[3], filt_size, filt_size, 1, 1, pad, pad))
  mlp:add(settings.get_act(act))
  mlp:add(nn.SpatialConvolutionMM(nstates[3], nstates[3], filt_size, filt_size, 1, 1, pad, pad))
  mlp:add(settings.get_act(act))
  mlp:add(nn.SpatialConvolutionMM(nstates[3], nstates[3], filt_size, filt_size, 1, 1, pad, pad))
  mlp:add(settings.get_act(act))
      
  local pool_2 = nn.SpatialMaxPooling(2,2,2,2)
  mlp:add(pool_2)
      
  mlp:add(nn.SpatialConvolutionMM(nstates[3], nstates[4], filt_size, filt_size, 1, 1, pad, pad))
  mlp:add(settings.get_act(act))
  mlp:add(nn.SpatialConvolutionMM(nstates[4], nstates[4], filt_size, filt_size, 1, 1, pad, pad))
  mlp:add(settings.get_act(act))
  mlp:add(nn.SpatialConvolutionMM(nstates[4], nstates[4], filt_size, filt_size, 1, 1, pad, pad))
  mlp:add(settings.get_act(act))
  
  local pool_3 = nn.SpatialMaxPooling(2,2,2,2)
  mlp:add(pool_3)
  
  if fc then 
    mlp:add(nn.Reshape(nstates[4]*8*8))
    mlp:add(nn.Dropout(dropout))
    mlp:add(nn.Linear(nstates[4]*8*8, nstates[5]))
    mlp:add(nn.Linear(nstates[5], nstates[4]*8*8))
    mlp:add(nn.Reshape(nstates[4], 8, 8))
  end
  
  
  mlp:add(nn.SpatialMaxUnpooling(pool_3))
  mlp:add(nn.SpatialConvolutionMM(nstates[4], nstates[4], filt_size, filt_size, 1, 1, pad, pad))
  mlp:add(settings.get_act(act))
  mlp:add(nn.SpatialConvolutionMM(nstates[4], nstates[4], filt_size, filt_size, 1, 1, pad, pad))
  mlp:add(settings.get_act(act))
  mlp:add(nn.SpatialConvolutionMM(nstates[4], nstates[3], filt_size, filt_size, 1, 1, pad, pad))
  mlp:add(settings.get_act(act))
  
  mlp:add(nn.SpatialMaxUnpooling(pool_2))
  mlp:add(nn.SpatialConvolutionMM(nstates[3], nstates[3], filt_size, filt_size, 1, 1, pad, pad))
  mlp:add(settings.get_act(act))
  mlp:add(nn.SpatialConvolutionMM(nstates[3], nstates[3], filt_size, filt_size, 1, 1, pad, pad))
  mlp:add(settings.get_act(act))
  mlp:add(nn.SpatialConvolutionMM(nstates[3], nstates[2], filt_size, filt_size, 1, 1, pad, pad))
  mlp:add(settings.get_act(act))
  
  mlp:add(nn.SpatialMaxUnpooling(pool))
  mlp:add(nn.SpatialConvolutionMM(nstates[2], nstates[2], filt_size, filt_size, 1, 1, pad, pad))
  mlp:add(settings.get_act(act))
  mlp:add(nn.SpatialConvolutionMM(nstates[2], nstates[2], filt_size, filt_size, 1, 1, pad, pad))
  mlp:add(settings.get_act(act))
  mlp:add(nn.SpatialConvolutionMM(nstates[2], nstates[1], filt_size, filt_size, 1, 1, pad, pad))

  collectgarbage()
  return mlp
end

return model 
