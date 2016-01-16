dqn = {}
require 'TransitionTable_spriority'
require 'cutorch'

seed = torch.random(1,10000)
torch.manualSeed(seed)
print('seed:', seed)

local function trans_table()
	local args = {
	    stateDim = 10 , numActions = 5,
	    histLen = 4, gpu = 1,
	    maxSize = 32, histType = "linear",
	    histSpacing = 1, nonTermProb = 1,
	    bufferSize = 16,
	    subgoal_dims = 1
	}

	local transitions = dqn.TransitionTable(args)
	for ii=1,100 do

		for i =1,36, 4 do
			for j=i,i+5 do
				transitions:add(torch.rand(args.stateDim), 2, torch.Tensor({0, 0}), false, 1)
			end
			transitions:add(torch.rand(args.stateDim), 2, torch.Tensor({1, 0}), true, 1)
		end
		-- print('table # -> ', transitions:size())
        -- print('\n--------------------')
        -- print('END:', transitions.end_ptrs)
        -- print('Before sample DYN:', transitions.dyn_ptrs)
        -- print('R:', transitions.trace_indxs_with_reward)

        for kk=1,5 do
			s, a, r, s2, t, sg, sg2= transitions:sample(8)
			-- print(r)
        	-- print('After sample DYN:', transitions.dyn_ptrs)
    	end
	end
	print('[Transition Table with Prioritization] Success!')
end

trans_table() -- testing transition table
