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

    net:add(convLayer(args.hist_len*args.ncols, args.n_units[1],
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

    -- reshape all feature planes into a vector per example
    net:add(nn.Reshape(nel))

    -- join vectors

    local subgoal_proc = nn.Sequential()
                            :add(nn.Linear(args.subgoal_dims, args.subgoal_nhid))
                            :add(nn.Sigmoid())
                            :add(nn.Linear(args.subgoal_nhid,args.subgoal_nhid))
                            :add(nn.Sigmoid())

    local net_parallel = nn.ParallelTable(2)
    net_parallel:add(net)
    net_parallel:add(subgoal_proc)

    local full_net = nn.Sequential()
    full_net:add(net_parallel)
    full_net:add(nn.JoinTable(2))


    -- fully connected layer    
    full_net:add(nn.Linear(nel+args.subgoal_nhid*9, args.n_hid[1]))
    full_net:add(args.nl())
    local last_layer_size = args.n_hid[1]

    for i=1,(#args.n_hid-1) do
        -- add Linear layer
        last_layer_size = args.n_hid[i+1]
        full_net:add(nn.Linear(args.n_hid[i], last_layer_size))
        full_net:add(args.nl())
    end



    -- add the last fully connected layer (to actions)
    full_net:add(nn.Linear(last_layer_size, args.n_actions))

    if args.gpu >=0 then
        full_net:cuda()
    end
    if args.verbose >= 2 then
        print(full_net)
        print('Convolutional layers flattened output size:', nel)
    end
    return full_net
end
