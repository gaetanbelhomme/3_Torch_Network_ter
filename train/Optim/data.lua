require 'cunn'
require 'cutorch'

data = {}

function data.create_batch_table(train_start, train_end, nb_elements)
  -- Dataset
  dataset={};
  function dataset:size() return nb_elements end
  for i=1,dataset:size() do
    local input = train_start[i]
    local output = train_end[i]

    input = input:cuda()
    output = output:cuda()
    dataset[i] = {input, output}
  end

  print(dataset:size())
  return dataset
end

function data.create_batch_tensor(train_start, train_end, nb_elements, offset)
  -- Dataset
  batchInputs = torch.CudaTensor(nb_elements, 1, 64, 64)
  batchOutputs = torch.CudaTensor(nb_elements, 1, 64, 64)
  
  for i=1,nb_elements do
    local input = train_start[i + offset]
    local output = train_end[i + offset]
    
    batchInputs[i]:copy(input)
    batchOutputs[i]:copy(output)
  end

  return batchInputs, batchOutputs
end

return data



