commands :
	To compute the real network, with the choice of components
		cd train/Generic/
		th main.lua (--help)

	To compute one convolution only
		cd train/TestNetwork/
		th main.lua

	To test the network given one .nrrd image
		cd test/
		python 1_NrrdtoNpy -i input.nrrd -o path
		th 2_test.lua network.bin input.npy
		python 3_NpytoNrrd -i input_prediction.npy -header input_header.pickle



Data/all : all the compressed data + headers
lib/function : a useful function (split string)
test/ : 
	1NrrdtoNpy : convert nrrd into npy with its header
	-i inputimage -o outputpath (type --help)

        2test : compute the network given an input
        pathNetwork inputImage (type --help)

	3NpytoNrrd : convert npy into nrrd given an input and a header
        -i inputimage -header header_image
train/ :
	Generic : contain a generic network:  
		main.lua : take the data, create a model, and train (SGD optimization (manual)), 
		model_lua x2 : given parameters, create a model, 
		data.lua : given a batch size, allow to create batch of data, 
		settings.lua : take care of the activation function and the criterion choice

 	type "th main.lua --help" to see choices. The training data should be in the directory ../../Data/all

	
	Optim : contain an network according to one model [(C,R)*3,maxP]*3 Fc Fc [maxP,(C,R)*3]*2 maxP,(C,R)*2,C]
	        use the SGD optimization (manual)
		contain main.lua, model.lua, data.lua


	SGDMethod : compute a network according to one model  [(C,R)*3,maxP]*3 [maxP,(C,R)*3]*2 maxP,(C,R)*2,C] (without FullyConnected)
		use the SGD optimization (torch function)
		contain main.lua


	TestNetwork : compute a network with only one convolution and one pooling layer
		contain main.lua, model.lua, data.lua
