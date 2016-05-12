--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "initenv"

function create_network(args)
    local net = nn.Sequential()
    net:add(nn.Reshape(unpack(args.input_dims)))

    --- first convolutional layer
    local convLayer = nn.SpatialConvolution

    net:add(convLayer( args.input_dims[1], args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],1))
    net:add(args.nl())

    -- Add convolutional layers
    for i=1,(#args.n_units-1) do
        -- second convolutional layer
        net:add(convLayer(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1]))
        net:add(args.nl())
    end


    local nel
    if args.gpu >= 0 then
        nel = net:cuda():forward(torch.zeros(1,unpack(args.input_dims))
                :cuda()):nElement()
    else
        nel = net:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
    end
    net:add(nn.Reshape(nel))

    local full_net = nn.Sequential()
    -- reshape all feature planes into a vector per example
    full_net:add(net)

    full_net:add(nn.Replicate(2)) -- one goes to reward and other to SR
    full_net:add(nn.SplitTable(1))
    local fork = nn.ParallelTable()
        -- qnet
        local qnet = nn.Sequential()
        -- fully connected layer    
        qnet:add(nn.Linear(nel, args.n_hid[1]))
        qnet:add(args.nl())
        local last_layer_size = args.n_hid[1]
        for i=1,(#args.n_hid-1) do
            -- add Linear layer
            last_layer_size = args.n_hid[i+1]
            qnet:add(nn.Linear(args.n_hid[i], last_layer_size))
            qnet:add(args.nl())
        end
        -- add the last fully connected layer (to actions)
        qnet:add(nn.Linear(last_layer_size, args.n_actions))
    fork:add(qnet)

        -- 1/0 goal : halting network
        local clf = nn.Sequential()
        clf:add(nn.Linear(nel, 256))
        clf:add(nn.ReLU())
        clf:add(nn.Linear(256, 2))
        clf:add(nn.LogSoftMax())

    fork:add(clf)

    full_net:add(fork)

    if args.gpu >=0 then
        full_net:cuda()
    end
    if args.verbose >= 2 then
        print(full_net)
        print('Convolutional layers flattened output size:', nel)
    end
    return full_net
end
