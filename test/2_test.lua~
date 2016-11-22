require 'torch'
require 'nn'
require 'cunn'
require 'cutorch'
npy4th = require 'npy4th'
require '../../3_Torch_Network/src_lua_gpu/function'

-- parser :
cmd = torch.CmdLine()
cmd:argument('net_path', 'network to load')
cmd:argument('input', 'input image')

options = cmd:parse(arg)

input = options.input
net = options.net_path

-- load model :
model = torch.load(net)

-- get data :
dataset = {}
function dataset:size() return 1 end
dataset = npy4th.loadnpy(input)
dataset = dataset:cuda()

-- get name :
name = split(input, '.')[1]

-- get output :
output = model:forward(dataset)

-- save output :
npy4th.savenpy('..' .. name .. '_prediction.npy', output)	


