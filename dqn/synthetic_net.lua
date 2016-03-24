require "initenv"
-- require 'nn'

function create_network(args)
    local net1 = nn.Sequential()
    
    local parallel2 = nn.ParallelTable()
    histLen = 1
    for i=1, histLen do
        parallel2:add(nn.LookupTable(100, args.n_hid))
    end

    net1:add(parallel2)
    net1:add(nn.JoinTable(2))
    net1:add(nn.Linear(args.n_hid*histLen, args.n_hid))
    net1:add(args.nl())

    local net2 = nn.Sequential()
    net2:add(nn.LookupTable(10, args.n_hid))
    net2:add(nn.Linear(args.n_hid, args.n_hid))
    net2:add(args.nl())

    local net_parallel = nn.ParallelTable()
    net_parallel:add(net1)
    net_parallel:add(net2)

    local net = nn.Sequential()

    net:add(net_parallel)
    net:add(nn.JoinTable(2))

    net:add(nn.Linear(args.n_hid*2, args.n_actions))

    if args.gpu >=0 then
        net:cuda()
    end

    return net
end

-- net = create_network({n_hid=10, n_actions=2, gpu=-1, nl=nn.ReLU})
-- net:forward({nn.SplitTable(2):forward(torch.ones(20, 4)), torch.ones(20)})
-- net:backward({nn.SplitTable(2):forward(torch.ones(20, 4)), torch.ones(20)}, torch.ones(20,2))