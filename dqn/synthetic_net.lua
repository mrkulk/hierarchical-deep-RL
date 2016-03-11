require "initenv"

function create_network(args)
    local net = nn.Sequential()
    net:add(nn.LookupTable(100, args.n_hid))
    net:add(nn.Linear(args.n_hid, args.n_hid))
    net:add(args.nl())

    net:add(nn.Linear(args.n_hid, args.n_actions))

    if args.verbose >= 2 then
        print(net)
        print('Convolutional layers flattened output size:', nel)
    end
    return net
end
