--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]
require 'image'
require 'torch'
if not dqn then
    require 'initenv'
end

local nql = torch.class('dqn.NeuralQLearner')

------ ZMQ server -------
local json = require ("dkjson")
local zmq = require "lzmq"
ctx = zmq.context()
skt = ctx:socket{zmq.REQ,
    linger = 0, rcvtimeo = 1000;
    connect = "tcp://127.0.0.1:" .. ZMQ_PORT;
}

function nql:__init(args)
    self.state_dim  = args.state_dim -- State dimensionality.
    self.actions    = args.actions
    self.n_actions  = #self.actions
    self.verbose    = args.verbose
    self.best       = args.best

    self.subgoal_dims = args.subgoal_dims
    self.subgoal_nhid = args.subgoal_nhid

    -- to keep track of stats
    self.subgoal_success = {}
    self.subgoal_total = {}

    -- to keep track of dying position
    self.deathPosition = nil
    self.DEATH_THRESHOLD = 15

    --- epsilon annealing
    self.ep_start   = args.ep or 1
    self.ep         = self.ep_start -- Exploration probability.
    self.ep_end     = args.ep_end or self.ep
    self.ep_endt    = args.ep_endt or 1000000

    ---- learning rate annealing
    self.lr_start       = args.lr or 0.01 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 1000000
    self.wc             = args.wc or 0  -- L2 weight cost.
    self.minibatch_size = args.minibatch_size or 1
    self.valid_size     = args.valid_size or 500

    --- Q-learning parameters
    self.dynamic_discount = 0
    self.discount       = args.discount or 0.99 --Discount factor.
    self.discount_internal       = args.discount_internal --Discount factor for internal rewards
    self.update_freq    = args.update_freq or 1
    -- Number of points to replay per learning step.
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.learn_start    = args.learn_start or 0
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory or 1000000
    self.hist_len       = args.hist_len or 1
    self.rescale_r      = args.rescale_r
    self.max_reward     = args.max_reward
    self.min_reward     = args.min_reward
    self.clip_delta     = args.clip_delta
    self.target_q       = args.target_q
    self.bestq          = 0

    self.gpu            = args.gpu

    self.ncols          = args.ncols or 1  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.hist_len*self.ncols, 84, 84}
    self.preproc        = args.preproc  -- name of preprocessing network
    self.histType       = args.histType or "linear"  -- history type to use
    self.histSpacing    = args.histSpacing or 1
    self.nonTermProb    = args.nonTermProb or 1
    self.bufferSize     = args.bufferSize or 512

    self.transition_params = args.transition_params or {}

    self.network    = args.network or self:createNetwork()
    -- check whether there is a network file
    local network_function
    if not (type(self.network) == 'string') then
        error("The type of the network provided in NeuralQLearner" ..
              " is not a string!")
    end

    local msg, err = pcall(require, self.network)
    if not msg then
        -- try to load saved agent
        local err_msg, exp = pcall(torch.load, self.network)
        if not err_msg then
            error("Could not find network file ")
        end
        if self.best and exp.best_model then
            self.network = exp.best_model
        else
            self.network = exp.model
        end
    else
        print('Creating Agent Network from ' .. self.network)
        self.network = err
        self.network = self:network()
    end

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
    else
        self.network:float()
    end

    -- Load preprocessing network.
    if not (type(self.preproc == 'string')) then
        error('The preprocessing is not a string')
    end
    msg, err = pcall(require, self.preproc)
    if not msg then
        error("Error loading preprocessing net")
    end
    self.preproc = err
    self.preproc = self:preproc()
    self.preproc:float()

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.network:float()
        self.tensor_type = torch.FloatTensor
    end

    -- Create transition table.
    ---- assuming the transition table always gets floating point input
    ---- (Foat or Cuda tensors) and always returns one of the two, as required
    ---- internally it always uses ByteTensors for states, scaling and
    ---- converting accordingly
    local transition_args = {
        stateDim = self.state_dim, numActions = self.n_actions,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize,
        subgoal_dims = args.subgoal_dims
    }

    self.transitions = dqn.TransitionTable(transition_args)

    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
    self.lastAction = nil
    self.lastSubgoal = nil
    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.

    self.q_max = 1
    self.r_max = 1

    self.network_real = self.network:clone()

    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    self.deltas = self.dw:clone():fill(0)
    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)

    self.w_real, self.dw_real = self.network_real:getParameters()
    self.dw_real:zero()
    self.deltas_real = self.dw_real:clone():fill(0)
    self.tmp_real= self.dw_real:clone():fill(0)
    self.g_real  = self.dw_real:clone():fill(0)
    self.g2_real = self.dw_real:clone():fill(0)


    if self.target_q then
        self.target_network = self.network:clone()
        self.target_network_real = self.network_real:clone()
    end
end


function nql:reset(state)
    if not state then
        return
    end
    self.best_network = state.best_network
    self.best_network_real = state.best_network_real
    
    self.network = state.model
    self.network_real = state.model_real
    
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()

    self.w_real, self.dw_real = self.network_real:getParameters()
    self.dw_real:zero()
    
    self.numSteps = 0
    print("RESET STATE SUCCESFULLY")
end


function nql:preprocess(rawstate)
    if self.preproc then
        return self.preproc:forward(rawstate:float())
                    :clone():reshape(self.state_dim)
    end

    return rawstate
end


function nql:getQUpdate(args, external_r)
    local s, a, r, s2, term, delta
    local q, q2, q2_max

    s = args.s
    a = args.a
    r = args.r
    s2 = args.s2
    subgoals2 = args.subgoals2
    subgoals = args.subgoals
    term = args.term

    -- The order of calls to forward is a bit odd in order
    -- to avoid unnecessary calls (we only need 2).

    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
    term = term:clone():float():mul(-1):add(1)

    local target_q_net
    if self.target_q then
        target_q_net = args.target_network
    else
        target_q_net = args.network
    end

    -- Compute max_a Q(s_2, a).
    -- print(s2:size(), subgoals2:size())
    q2_max = target_q_net:forward({s2, subgoals2}):float():max(2)

    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)

    local discount
    if external_r then
        discount = math.max(self.dynamic_discount, self.discount) -- for real network
    else
        discount = math.max(self.dynamic_discount, self.discount_internal) -- for subgoal network
    end

    q2 = q2_max:clone():mul(discount):cmul(term)

    delta = r:clone():float()

    -- TODO: removed scaling. check later
    -- if self.rescale_r then
    --     delta:div(self.r_max)
    -- end

    delta:add(q2)

    -- q = Q(s,a)
    local q_all = args.network:forward({s, subgoals}):float()
    q = torch.FloatTensor(q_all:size(1))
    for i=1,q_all:size(1) do
        q[i] = q_all[i][a[i]]
    end
    delta:add(-1, q)

    if self.clip_delta then
        delta[delta:ge(self.clip_delta)] = self.clip_delta
        delta[delta:le(-self.clip_delta)] = -self.clip_delta
    end

    local targets = torch.zeros(self.minibatch_size, self.n_actions):float()
    for i=1,math.min(self.minibatch_size,a:size(1)) do
        targets[i][a[i]] = delta[i]
    end

    if self.gpu >= 0 then targets = targets:cuda() end

    return targets, delta, q2_max
end


function nql:qLearnMinibatch(network, target_network, dw, w, g, g2, tmp, deltas, external_r)
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    assert(self.transitions:size() > self.minibatch_size)

    local s, a, r, s2, term, subgoals, subgoals2 = self.transitions:sample(self.minibatch_size)

    if external_r then
        r = r[{{},1}] --extract external reward
        subgoals[{{},{1,self.subgoal_dims}}] = 0
        subgoals2[{{},{1,self.subgoal_dims}}] = 0
    else
        r = r[{{},2}] --external + intrinsic reward 
    end

    local targets, delta, q2_max = self:getQUpdate({s=s, a=a, r=r, s2=s2,
        term=term, subgoals = subgoals, subgoals2=subgoals2, network = network, update_qmax=true, target_network = target_network}, external_r)

    -- zero gradients of parameters
    dw:zero()

    -- get new gradient
    -- print(subgoals)
    network:backward({s, subgoals}, targets)

    -- add weight cost to gradient
    dw:add(-self.wc, w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

     --grad normalization
    -- local max_norm = 1000
    -- local grad_norm = dw:norm()
    -- if grad_norm > max_norm then
    --   local scale_factor = max_norm/grad_norm
    --   dw:mul(scale_factor)
    --   if false and grad_norm > 1000 then
    --       print("Scaling down gradients. Norm:", grad_norm)
    --   end
    -- end

    -- use gradients (original)
    -- g:mul(0.95):add(0.05, dw)
    -- tmp:cmul(dw, dw)
    -- g2:mul(0.95):add(0.05, tmp)
    -- tmp:cmul(g, g)
    -- tmp:mul(-1)
    -- tmp:add(g2)
    -- tmp:add(0.01)
    -- tmp:sqrt()

    --rmsprop
    local smoothing_value = 1e-8
    tmp:cmul(dw, dw)
    g:mul(0.9):add(0.1, tmp)
    tmp = torch.sqrt(g)
    tmp:add(smoothing_value)  --negative learning rate

    -- accumulate update
    deltas:mul(0):addcdiv(self.lr, dw, tmp)
    w:add(deltas)
end


function nql:sample_validation_data()
    local s, a, r, s2, term, subgoals, subgoals2 = self.transitions:sample(self.valid_size)
    self.valid_s    = s:clone()
    self.valid_a    = a:clone()
    self.valid_r    = r:clone()
    self.valid_s2   = s2:clone()
    self.valid_term = term:clone()
    self.valid_subgoals = subgoals:clone()
    self.valid_subgoals2 = subgoals2:clone()
end


function nql:compute_validation_statistics()
    local targets, delta, q2_max = self:getQUpdate{s=self.valid_s,
        a=self.valid_a, r=self.valid_r[{{},1}], s2=self.valid_s2, term=self.valid_term, subgoals = self.valid_subgoals,
         subgoals2 = self.valid_subgoals2, network = self.network_real, target_network = self.target_network_real}

    self.v_avg = self.q_max * q2_max:mean()
    self.tderr_avg = delta:clone():abs():mean()
end


function process_pystr(msg)
    loadstring(msg)()
    for i = 1, #objlist do
        objlist[i] = torch.Tensor(objlist[i])
    end
    return objlist
end

-- returns a table of num_objects x vectorized object reps
function nql:get_objects(rawstate)
    image.save('tmp_' .. ZMQ_PORT .. '.png', rawstate[1])
    skt:send("")
    msg = skt:recv()
    while msg == nil do
        msg = skt:recv()
    end
    local object_list = process_pystr(msg)
    return object_list --nn.SplitTable(1):forward(torch.rand(4, self.subgoal_dims))  
end

function nql:pick_subgoal(rawstate, oid)
    local objects = self:get_objects(rawstate)
    local indxs = oid or torch.random(2, #objects) -- first is the agent
    while objects[indxs]:sum() == 0 do -- object absent
        indxs = torch.random(2, #objects) -- first is the agent
    end
    -- concatenate subgoal with objects (input into network)
    local subg = objects[indxs]
    local ftrvec = torch.zeros(#objects*self.subgoal_dims)
    for i = 1,#objects do
        ftrvec[{{(i-1)*self.subgoal_dims + 1, i*self.subgoal_dims}}] = objects[i]
    end

    -- add stats
    self.subgoal_total[subg:sum()] = self.subgoal_total[subg:sum()] or 0
    self.subgoal_total[subg:sum()] = self.subgoal_total[subg:sum()] + 1

    return torch.cat(subg, ftrvec)
end

function nql:isGoalReached(subgoal, objects)
    local agent = objects[1]

    -- IMP: remember that subgoal includes both subgoal and all objects
    local dist = math.sqrt((subgoal[1] - agent[1])^2 + (subgoal[2]-agent[2])^2)
    if dist < 13 then --just a small threshold to indicate when agent meets subgoal (euc dist)
        print('subgoal reached!')
        -- local indexTensor = subgoal[{{3, self.subgoal_dims}}]:byte()
        -- print(subgoal, indexTensor)
        local subg = subgoal[{{1, self.subgoal_dims}}]
        self.subgoal_success[subg:sum()] = self.subgoal_success[subg:sum()] or 0
        self.subgoal_success[subg:sum()] = self.subgoal_success[subg:sum()] + 1
        return true
    else
        return false
    end
end

function nql:intrinsic_reward(subgoal, objects)
    -- return reward based on distance or 0/1 towards sub-goal
    local agent = objects[1]
    local reward
    -- if self.lastSubgoal then
    --     print("last subgoal", self.lastSubgoal[{{1,7}}])
    -- end
    -- print("current subgoal", subgoal[{{1,7}}])
    if self.lastSubgoal and (self.lastSubgoal[{{3,self.subgoal_dims}}] - subgoal[{{3, self.subgoal_dims}}]):abs():sum() == 0 then
        local dist1 = math.sqrt((subgoal[1] - agent[1])^2 + (subgoal[2]-agent[2])^2)
        local dist2 = math.sqrt((self.lastSubgoal[1] - self.lastobjects[1][1])^2 + (self.lastSubgoal[2]-self.lastobjects[1][2])^2)
        reward = dist2 - dist1
    else
        reward = 0
    end

    return reward * 0.1
end


function nql:perceive(subgoal, reward, rawstate, terminal, testing, testing_ep)
    -- Preprocess state (will be set to nil if terminal)
    if terminal then
        reward = -200
    end

    local state = self:preprocess(rawstate):float()
    local objects = self:get_objects(rawstate)

    if terminal then
        self.deathPosition = objects[1][{{1,2}}] --just store the x and y coords of the agent
    end 

    local goal_reached = self:isGoalReached(subgoal, objects)
    local intrinsic_reward = self:intrinsic_reward(subgoal, objects)
    reward = reward - 0.01 -- penalize for just standing
    if goal_reached then
        intrinsic_reward = intrinsic_reward + 50
    end

    local curState

    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end
    if self.rescale_r then
        self.r_max = math.max(self.r_max, reward)
    end

    --print(reward, intrinsic_reward)

    self.transitions:add_recent_state(state, terminal, subgoal)  

    -- local currentFullState = self.transitions:get_recent()

    --Store transition s, a, r, s'
    if self.lastState and not testing and self.lastSubgoal then
        self.transitions:add(self.lastState, self.lastAction, torch.Tensor({reward, reward + intrinsic_reward}),
                            self.lastTerminal, self.lastSubgoal, priority)
    end

    if self.numSteps == self.learn_start+1 and not testing then
        self:sample_validation_data()
    end

    curState, subgoal = self.transitions:get_recent()
    curState = curState:resize(1, unpack(self.input_dims))

    -- Select action
    local actionIndex = 1
    local qfunc
    if not terminal then
        actionIndex, qfunc = self:eGreedy(curState, testing_ep, subgoal)
    end

    self.transitions:add_recent_action(actionIndex) 

    --Do some Q-learning updates
    if self.numSteps > self.learn_start and not testing and
        self.numSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:qLearnMinibatch(self.network, self.target_network, self.dw, self.w, self.g, self.g2, self.tmp, self.deltas, false)
            self:qLearnMinibatch(self.network_real,  self.target_network_real, self.dw_real, self.w_real, self.g_real, self.g2_real, self.tmp_real, self.deltas_real, true) 
        end
    end

    if not testing then
        self.numSteps = self.numSteps + 1
    end

    self.lastState = state:clone()
    self.lastAction = actionIndex
    self.lastTerminal = terminal
    if not terminal then
        -- print("Getting subgoal")
        self.lastSubgoal = subgoal
        --check if the game is still in the stages right after the agent dies
        if self.deathPosition then
            currentPosition = objects[1][{{1,2}}]
            -- print("Positions:", currentPosition, self.deathPosition)
            if math.sqrt((currentPosition[1]-self.deathPosition[1])^2 + (currentPosition[2]-self.deathPosition[2])^2) < self.DEATH_THRESHOLD then
                self.lastSubgoal = nil
                -- print("death overruling")
            else
                -- print("Removing death position", self.deathPosition)
                self.deathPosition = nil
            end

        end

    else
        self.lastSubgoal = nil
    end



    self.lastobjects = objects

    if self.target_q and self.numSteps % self.target_q == 1 then
        self.target_network = self.network:clone()
        self.target_network_real = self.network_real:clone() 
    end

    if not terminal then
        return actionIndex, goal_reached, reward, reward+intrinsic_reward, qfunc
    else
        return 0, goal_reached, reward, reward+intrinsic_reward, qfunc
    end
end


function nql:eGreedy(state, testing_ep, subgoal)
    self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))

    -- if testing, zero out subgoals
    if testing_ep then
	subgoal = subgoal:clone()
        subgoal[{{1,self.subgoal_dims}}] = 0
    end
       

    -- Epsilon greedy
    if torch.uniform() < self.ep then
        return torch.random(1, self.n_actions)
    else
        return self:greedy(state, subgoal)
    end
end


function nql:greedy(state, subgoal)
    -- Turn single state into minibatch.  Needed for convolutional nets.
    if state:dim() == 2 then
        assert(false, 'Input must be at least 3D')
        state = state:resize(1, state:size(1), state:size(2))
    end
    subgoal = torch.reshape(subgoal, 1, self.subgoal_dims*9)
    if self.gpu >= 0 then
        state = state:cuda()
        subgoal = subgoal:cuda()
    end
    local q = self.network:forward({state, subgoal}):float():squeeze()
    local maxq = q[1]
    local besta = {1}
    -- print("Q Value:", q)
    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, self.n_actions do
        if q[a] > maxq then
            besta = { a }
            maxq = q[a]
        elseif q[a] == maxq then
            besta[#besta+1] = a
        end
    end
    self.bestq = maxq

    local r = torch.random(1, #besta)

    self.lastAction = besta[r]

    return besta[r], q
end


function nql:createNetwork()
    local n_hid = 128
    local mlp = nn.Sequential()
    mlp:add(nn.Reshape(self.hist_len*self.ncols*self.state_dim))
    mlp:add(nn.Linear(self.hist_len*self.ncols*self.state_dim, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))

    return mlp
end


function nql:report()
    print("Subgoal Network\n---------------------")
    print(get_weight_norms(self.network))
    print(get_grad_norms(self.network))
    print(" Real Network\n---------------------")
    print(get_weight_norms(self.network_real))
    print(get_grad_norms(self.network_real))
    

    -- print stats on subgoal success rates
    for subg, val in pairs(self.subgoal_total) do
        if self.subgoal_success[subg] then 
            print("Subgoal " , subg , ' : ',  self.subgoal_success[subg]/val, self.subgoal_success[subg] .. '/' .. val)
        else
            print("Subgoal " , subg ,  ' : ')
        end
    end
    self.subgoal_success = {}
    self.subgoal_total = {}
end
