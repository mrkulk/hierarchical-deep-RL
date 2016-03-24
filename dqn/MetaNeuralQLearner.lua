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

    -- run subgoal specific experiments
    self.use_distance = args.use_distance -- if we want to use the distance as the reward

    -- to keep track of stats
    self.subgoal_success = {}
    self.subgoal_total = {}

    self.global_subgoal_success = {}
    self.global_subgoal_total = {}

    self.subgoal_seq = {}
    self.global_subgoal_seq = {}


    -- to keep track of dying position
    self.deathPosition = nil
    self.DEATH_THRESHOLD = 15
    self.ignoreState = nil
    self.metaignoreState = nil

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
    self.lr_meta        = args.lr_meta
    self.wc             = args.wc or 0  -- L2 weight cost.
    self.minibatch_size = args.minibatch_size or 1
    self.valid_size     = args.valid_size or 32
    

    --- Q-learning parameters
    self.dynamic_discount = args.dynamic_discount
    self.discount       = args.discount or 0.99 --Discount factor.
    self.discount_internal       = args.discount_internal --Discount factor for internal rewards
    self.update_freq    = args.update_freq or 1
    -- Number of points to replay per learning step.
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.learn_start    = args.learn_start or 0
    self.meta_learn_start    = args.meta_learn_start or 0
    
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory or 1000000
    self.hist_len       = args.hist_len or 1
    self.rescale_r      = args.rescale_r
    self.max_reward     = args.max_reward
    self.min_reward     = args.min_reward
    self.clip_delta     = args.clip_delta
    self.target_q       = args.target_q

    self.gpu            = args.gpu

    self.ncols          = args.ncols or 1  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.hist_len} -- {self.hist_len*self.ncols, 84, 84}
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
        if self.best and exp.best_model then --best_model_real and model_rel if testing on non-subgoal network  
            self.network = exp.best_model
        else
            self.network = exp.model
        end

        if exp.model_meta then
            self.network_meta = exp.model_meta
        end
    else
        print('Creating Agent Network from ' .. self.network)
        self.network = err
        -- print("NETWORK", self.network)
        self.network = self:network()    
    end

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        if self.network_meta then
            self.network_meta:cuda()
        end
    else
        self.network:float()
        if self.network_meta then
            self.network_meta:float()
        end
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
        maxSize = 200000, --self.replay_memory, 
        histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize,
        subgoal_dims = args.subgoal_dims
    }
    self.transitions = dqn.TransitionTable(transition_args)

    --- meta table
    local meta_transition_args = {
        stateDim = self.state_dim, numActions = args.max_objects,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = 50000, --self.replay_memory, 
        histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = 512,
        subgoal_dims = args.subgoal_dims
    }
    -- self.meta_transitions = dqn.TransitionTable_priority(meta_transition_args)
    self.meta_transitions = dqn.TransitionTable(meta_transition_args)

    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
    self.lastAction = nil
    self.lastSubgoal = nil

    self.metanumSteps = 0
    self.metalastState = nil
    self.metalastAction = nil
    self.metalastSubgoal = nil
    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.

    self.q_max = 1
    self.r_max = 1

    -- TODO: also save this into file and read
    local meta_args = table_clone(args)
    meta_args.n_hid          = 50
    meta_args.nl             = nn.Rectifier
    meta_args.n_actions = 5
    meta_args.input_dims = self.input_dims
    self.meta_args = meta_args

    -- create a meta network from scratch if not read in from saved file
    if not self.network_meta then
        print("Creating new Meta Network.....")
        require 'synthetic'
       
        self.network_meta = create_network(meta_args)
    end

    -- copy the lower level weights from lower network
    --print(self.network.modules)
    --for i=1, #self.network.modules-1 do
    --   print(self.network.modules[i])
    --    if i==1 then
    --        for j=1, #self.network.modules[1].modules do
    --        if self.network.modules[1].modules[j].bias then
    --           self.network_meta.modules[1].modules[j].bias = self.network.modules[1].modules[j].bias:clone()
    --        end
    --        if self.network.modules[1].modules[j].weights then
    --            self.network_meta.modules[1].modules[j].weights = self.network.modules[1].modules[j].weights:clone()
    --        end
    --        end
    --    end
    --    if self.network.modules[i].bias then 
    --        self.network_meta.modules[i].bias = self.network.modules[i].bias:clone()
    --    end
    --    if self.network.modules[i].weights then 
    --    self.network_meta.modules[i].weights = self.network.modules[i].weights:clone()
    --    end
    --end

    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    self.deltas = self.dw:clone():fill(0)
    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)

    self.w_meta, self.dw_meta = self.network_meta:getParameters()
    self.dw_meta:zero()
    self.deltas_meta = self.dw_meta:clone():fill(0)
    self.tmp_meta= self.dw_meta:clone():fill(0)
    self.g_meta  = self.dw_meta:clone():fill(0)
    self.g2_meta = self.dw_meta:clone():fill(0)


    if self.target_q then
        self.target_network = self.network:clone()
        self.target_network_meta = self.network_meta:clone()
        self.w_target, self.dw_target = self.target_network:getParameters()
        self.w_meta_target, self.dw_meta_target = self.target_network_meta:getParameters()
    end

end


function nql:reset(state)
    if not state then
        return
    end
    self.best_network = state.best_network
    self.best_network_meta = state.best_network_meta
    
    self.network = state.model
    self.network_meta = state.model_meta
    
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()

    self.w_meta, self.dw_meta = self.network_meta:getParameters()
    self.dw_meta:zero()
    
    self.numSteps = 0
    print("RESET STATE SUCCESFULLY")
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
    s2[torch.le(s2, 0)] = 99
    subgoals2[torch.le(subgoals2, 0)] = 99
    -- print("s2", s2:size(),subgoals2:size(), '----------------')  
    q2_max = target_q_net:forward({nn.SplitTable(2):forward(s2), subgoals2:resize(subgoals2:size(1))}):float():max(2)

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
    s[torch.le(s, 0)] = 99
    subgoals[torch.le(subgoals, 0)] = 99
    local q_all = args.network:forward({nn.SplitTable(2):forward(s), subgoals:resize(subgoals:size(1))}):float()
    q = torch.FloatTensor(q_all:size(1))
    for i=1,q_all:size(1) do
        q[i] = q_all[i][a[i]]
    end
    delta:add(-1, q)

    if self.clip_delta then
        delta[delta:ge(self.clip_delta)] = self.clip_delta
        delta[delta:le(-self.clip_delta)] = -self.clip_delta
    end

    local targets = torch.zeros(self.minibatch_size, args.n_actions):float()
    for i=1,math.min(self.minibatch_size,a:size(1)) do
        targets[i][a[i]] = delta[i]
    end

    if self.gpu >= 0 then targets = targets:cuda() end

    return targets, delta, q2_max
end


function nql:qLearnMinibatch(network, target_network, tran_table, dw, w, g, g2, tmp, deltas, external_r, nactions, metaFlag)
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    assert(tran_table:size() > self.minibatch_size)

    local s, a, r, s2, term, subgoals, subgoals2 = tran_table:sample(self.minibatch_size)
    -- print(r, s:sum(2))
    if external_r then
        r = r[{{},1}] --extract external reward
    else    
        r = r[{{},2}] --external + intrinsic reward 
    end

    local targets, delta, q2_max = self:getQUpdate({s=s, a=a, r=r, s2=s2, n_actions = nactions,
        term=term, subgoals = subgoals, subgoals2=subgoals2, network = network, update_qmax=true, target_network = target_network}, external_r)

    -- zero gradients of parameters
    dw:zero()

    -- get new gradient
    -- print(subgoals)
    s[torch.le(s, 0)] = 99
    subgoals[torch.le(subgoals, 0)] = 99
    -- print(s:size(), subgoals:size(), targets:size())
    network:backward({nn.SplitTable(2):forward(s), subgoals:resize(subgoals:size(1))}, targets)

    -- add weight cost to gradient
    dw:add(-self.wc, w)

    -- compute linearly annealed learning rate
    if metaFlag then
        learn_start = self.meta_learn_start
    else
        learn_start = self.learn_start
    end

    local t = math.max(0, self.numSteps - learn_start)
    lr_start = self.lr_start
    if metaFlag then
        lr_start = self.lr_meta
    end
    self.lr = (lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
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
    g:mul(0.95):add(0.05, dw)
    tmp:cmul(dw, dw)
    g2:mul(0.95):add(0.05, tmp)
    tmp:cmul(g, g)
    tmp:mul(-1)
    tmp:add(g2)
    tmp:add(0.01)
    tmp:sqrt()

    --rmsprop
    -- local smoothing_value = 1e-8
    -- tmp:cmul(dw, dw)
    -- g:mul(0.9):add(0.1, tmp)
    -- tmp = torch.sqrt(g)
    -- tmp:add(smoothing_value)  --negative learning rate

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
    local targets, delta, q2_max = self:getQUpdate{s=self.valid_s, n_actions = self.n_actions,
        a=self.valid_a, r=self.valid_r[{{},1}], s2=self.valid_s2, term=self.valid_term, subgoals = self.valid_subgoals,
         subgoals2 = self.valid_subgoals2, network = self.network, target_network = self.target_network}

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


function nql:pick_subgoal(rawstate, metareward, terminal, testing, testing_ep)
    local ftrvec = torch.Tensor({0})
    local state = rawstate:clone()
    local subgoal_list = {1,3,4,5,6}

    -- print("XXXXX", state, ftrvec)
    
    self.meta_transitions:add_recent_state(state, terminal, ftrvec)  

    --Store transition s, a, r, s'

    if self.metalastState and not testing then
        -- print(self.metalastState, self.metalastAction, torch.Tensor({metareward, metareward + 0}),
                        -- self.metalastTerminal, ftrvec, priority)
        self.meta_transitions:add(self.metalastState, self.metalastAction, torch.Tensor({metareward, metareward + 0}),
                        self.metalastTerminal, ftrvec, priority)
        -- if metareward ~=0 then
        --     print("Metareward", metareward)
        -- end
    end


    curState, subgoal = self.meta_transitions:get_recent()
    curState = curState:resize(1, unpack(self.input_dims))

    -- Select action
    local actionIndex = 1
    local qfunc
    if not terminal then
        actionIndex, qfunc = self:eGreedy('meta', self.network_meta, curState, testing_ep, subgoal, self.metalastAction)
    end

    -- UNCOMMENT if you want to choose the subgoals
    -- print(qfunc)
    -- print("Action chosen:", actionIndex)
    -- actionIndex = io.read("*number")

    self.meta_transitions:add_recent_action(actionIndex) 

    --Do some Q-learning updates
    if self.metanumSteps > self.meta_learn_start and not testing and
        self.metanumSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:qLearnMinibatch(self.network_meta, self.target_network_meta, self.meta_transitions,
             self.dw_meta, self.w_meta, self.g_meta, self.g2_meta, self.tmp_meta, self.deltas_meta, true, self.meta_args.n_actions, true)
        end
    end


    if not testing then
        self.metanumSteps = self.metanumSteps + 1
    end

    self.metalastState = state:clone()
    self.metalastAction = actionIndex
    self.metalastTerminal = terminal


    local alpha = 0.999
    self.w_meta_target:mul(alpha):add(self.w_meta * (1-alpha))

    local subg = subgoal_list[actionIndex]

    if not terminal then
        self.subgoal_total[subg] = self.subgoal_total[subg] or 0
        self.subgoal_total[subg] = self.subgoal_total[subg] + 1

        self.global_subgoal_total[subg] = self.global_subgoal_total[subg] or 0
        self.global_subgoal_total[subg] = self.global_subgoal_total[subg] + 1
    end


    -- keep track of subgoal sequences
    if terminal then
        table.insert(self.global_subgoal_seq, self.subgoal_seq)
        self.subgoal_seq = {}
    else
        table.insert(self.subgoal_seq, subg)
    end

    -- Return subgoal    
    return torch.Tensor({subg}) --for hierarchical
    -- return torch.Tensor({1}) -- for basic DQN
end

function nql:isGoalReached(subgoal, rawstate)
    -- print("rawstate", rawstate[1], subgoal[1])
    if subgoal[1] == rawstate[1] then
        -- print("rawstate", rawstate[1], subgoal[1])
        self.subgoal_success[subgoal[1]] = (self.subgoal_success[subgoal[1]] or 0) + 1
        return true
    else
        return false
    end
end

function nql:intrinsic_reward(subgoal, rawstate)
    local reward
    -- print("siubgoals", self.lastSubgoal, subgoal)
    if self.lastSubgoal and self.lastSubgoal[1] == subgoal[1] then
        local dist1 = math.abs(subgoal[1] - rawstate[1])
        local dist2 = math.abs(self.lastSubgoal[1] - self.lastState[1])
        reward = dist2 - dist1
    else
        reward = 0
    end
    -- print("intrinsic_reward", reward, subgoal[1], rawstate[1])
    return reward
end


function nql:perceive(subgoal, reward, rawstate, terminal, testing, testing_ep)
    -- Preprocess state (will be set to nil if terminal)
    
    local state = rawstate:clone()

    -- if terminal then
        -- print(subgoal[1], rawstate[1], terminal)
        -- io.read()
    -- end

    local goal_reached = self:isGoalReached(subgoal, state)
    local intrinsic_reward = 0
    if self.use_distance then
        intrinsic_reward = self:intrinsic_reward(subgoal, state)
    end

    if goal_reached then
        -- print("GOAL reached")
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

    intrinsic_reward = intrinsic_reward - 0.01 -- step penalty

    -- print("intrinsic_reward", intrinsic_reward)

    -- intrinsic_reward = 0 -- for basic DQN

    self.transitions:add_recent_state(state, terminal, subgoal)  

    --Store transition s, a, r, s'
    if self.lastState and not testing and self.lastSubgoal then
        if self.ignoreState then
            self.ignoreState = nil
            -- print("reward would have been", intrinsic_reward)
        else
            if self.lastTerminal then
                        -- print("died3")
                    end
            -- print("Intrinsic Reward:", intrinsic_reward)

            self.transitions:add(self.lastState, self.lastAction, torch.Tensor({reward,  intrinsic_reward}),
                            self.lastTerminal, self.lastSubgoal, priority)
        end
        -- print("STORING PREV TRANSITION", self.lastState:sum(), self.lastAction, torch.Tensor({reward, reward + intrinsic_reward}),
                            -- self.lastTerminal, self.lastSubgoal:sum(), priority)
    end

    if self.numSteps == self.learn_start+1 and not testing then
        -- self:sample_validation_data()
    end

    curState, subgoal = self.transitions:get_recent()
    curState = curState:resize(1, unpack(self.input_dims))

    -- Select action
    local actionIndex = 1
    local qfunc
    if not terminal then
        actionIndex, qfunc = self:eGreedy('lower', self.network, curState, testing_ep, subgoal)
        --self:eGreedy(curState, testing_ep, subgoal)
    end

    self.transitions:add_recent_action(actionIndex) 

    --Do some Q-learning updates
    if self.numSteps > self.learn_start and not testing and
        self.numSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:qLearnMinibatch(self.network, self.target_network, self.transitions,
                self.dw, self.w, self.g, self.g2, self.tmp, self.deltas, false, self.n_actions, false)

            -- TODO: learning for Real network
            -- self:qLearnMinibatch(self.network_real,  self.target_network_real, self.dw_real, self.w_real, self.g_real, self.g2_real, self.tmp_real, self.deltas_real, true) 
        end
    end

    if not testing then
        self.numSteps = self.numSteps + 1
    end

    self.lastState = state:clone()
    self.lastAction = actionIndex
    self.lastTerminal = terminal
    if self.lastTerminal then
                -- print("died4")
    end

    if not terminal then
        self.lastSubgoal = subgoal
    else
        self.lastSubgoal = nil
    end

    -- target q copy
    if false then -- deprecated
        if self.target_q and self.numSteps % self.target_q == 1 then
            self.target_network = self.network:clone()
            self.target_network_real = self.network_real:clone() 
        end
    else --smooth average
        local alpha = 0.999
        self.w_target:mul(0.999):add(self.w * (1-alpha))
        -- self.w_real_target:mul(0.999):add(self.w_real * (1-alpha))
    end

    -- print("Reward", reward, intrinsic_reward)

    if not terminal then
        return actionIndex, goal_reached, reward, reward+intrinsic_reward, qfunc
    else
        return 0, goal_reached, reward, reward+intrinsic_reward, qfunc
    end
end


function nql:eGreedy(mode, network, state, testing_ep, subgoal, lastsubgoal)
    -- handle learn_start
    if mode == 'meta' then
        learn_start = self.meta_learn_start
    else
        learn_start = self.learn_start
    end
    self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - learn_start))/self.ep_endt))
   

    local n_actions = nil
    if mode == 'meta' then
        n_actions = self.meta_args.n_actions
    else
        n_actions = self.n_actions
    end

    -- Epsilon greedy
    if torch.uniform() < self.ep then
        if mode == 'meta' then
            local chosen_act = torch.random(1,n_actions)
            while chosen_act == lastsubgoal do
                chosen_act = torch.random(1,n_actions)
            end
            return chosen_act
        else
            return torch.random(1, n_actions)
        end
    else
        -- if mode == 'meta' then
        --     print("inputs to dqn", state, subgoal[1], lastsubgoal)
        --     io.read()
        -- end
        return self:greedy(network, n_actions, state, subgoal, lastsubgoal)
    end
end


function nql:greedy(network, n_actions,  state, subgoal, lastsubgoal)
    -- Turn single state into minibatch.  Needed for convolutional nets.
    -- if state:dim() == 2 then
    --     assert(false, 'Input must be at least 3D')
    --     state = state:resize(1, state:size(1), state:size(2))
    -- end
    if self.gpu >= 0 then
        state = state:cuda()
        subgoal = subgoal:cuda()
    end
    state[torch.le(state, 0)] = 99
    subgoal[torch.le(subgoal, 0)] = 99
    local q = network:forward({nn.SplitTable(2):forward(state), subgoal:resize(subgoal:size(1))}):float():squeeze()
    local maxq = q[1]
    local besta = { 1 }

    if lastsubgoal == 1 then
	    maxq = q[2]
	    besta = { 2 }
    end
    -- print("Q Value:", q)
    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, n_actions do
	    if a ~= lastsubgoal then
            if q[a] > maxq then
                besta = { a }
                maxq = q[a]
            elseif q[a] == maxq then
                besta[#besta+1] = a
            end
	    end
    end

    local r = torch.random(1, #besta)

    return besta[r], q
end


function nql:report(filename)
    print("Subgoal Network\n---------------------")
    print(get_weight_norms(self.network))
    print(get_grad_norms(self.network))
    -- print(" Real Network\n---------------------")
    -- print(get_weight_norms(self.network_real))
    -- print(get_grad_norms(self.network_real))
    

    -- print stats on subgoal success rates
    for subg, val in pairs(self.subgoal_total) do
        if self.subgoal_success[subg] then 
            print("Subgoal ID :" , subg , ' : ',  self.subgoal_success[subg]/val, self.subgoal_success[subg] .. '/' .. val)
        else
            print("Subgoal ID :" , subg ,  ' : ')
        end
    end

    torch.save(filename , {self.subgoal_success, self.subgoal_total, self.global_subgoal_seq})
    self.subgoal_success = {}
    self.subgoal_total = {}
    self.global_subgoal_seq = {}
end
