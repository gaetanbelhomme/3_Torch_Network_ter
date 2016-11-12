require 'nn'
require 'os'

settings = require 'settings'

model_2 = {}

function model_2.create_model(filt_size, pad, nstates, nb_layers, act, fc, patch_size, dropout, nb_conv)
  -- nb layer max 
  local nb_max = 0
  local i = patch_size

  while i % 2 == 0 do
    i = i / 2
    nb_max = nb_max + 1
  end

  if nb_layers > nb_max then
    print('Error, nb layers (',nb_layers,') > nb_max(',nb_max,')')
    os.exit()
  end

  local final_image_dim = patch_size / (2 ^ nb_layers)

  if fc then true_size_states = 2 + nb_layers
  else true_size_states = 1 + nb_layers end
 
  if #nstates > true_size_states then
    print('Error, nstates size (',#nstates,') != nb required (',true_size_states,')')
    os.exit()
  end

  -- pool layer
  local poolsize = 2

  local pool_layers = {}
  for i = 1, nb_layers, 1
  do
    pool_layers[i] = nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize)
  end
 
  mlp = nn.Sequential();

  for i = 1, nb_layers, 1
  do
    for j = 1, nb_conv, 1
    do 
      if j == 1 then index = i
      else index = i + 1 end
      mlp:add(nn.SpatialConvolutionMM(nstates[index], nstates[i + 1], filt_size, filt_size, 1, 1, pad, pad))
      mlp:add(settings.get_act(act))
    end

    mlp:add(pool_layers[i])
  end


  if fc then
    mlp:add(nn.Reshape(nstates[#nstates - 1]*final_image_dim*final_image_dim))
    mlp:add(nn.Dropout(dropout))
    mlp:add(nn.Linear(nstates[#nstates - 1]*final_image_dim*final_image_dim, nstates[#nstates]))
    mlp:add(nn.Linear(nstates[#nstates], nstates[#nstates - 1]*final_image_dim*final_image_dim))
    mlp:add(nn.Reshape(nstates[#nstates - 1], final_image_dim, final_image_dim))
  end

  for i = nb_layers, 1, -1
  do 
    mlp:add(nn.SpatialMaxUnpooling(pool_layers[i]))
    
    for j = nb_conv, 1, -1
    do
      if j == 1 then index = i
      else index = i + 1 end
      mlp:add(nn.SpatialConvolutionMM(nstates[i + 1], nstates[index], filt_size, filt_size, 1, 1, pad, pad))
      if (i ~= 1 or j ~= 1) then mlp:add(settings.get_act(act)) end
    end

  end


  collectgarbage()
  return mlp
end

return model_2


