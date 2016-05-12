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
    self.total_subgoals = args.total_subgoals

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
    self.input_dims     = args.input_dims or { (self.hist_len+1)*self.ncols, 84, 84} -- +1 for goal
    self.preproc        = args.preproc  -- name of preprocessing network
    self.histType       = args.histType or "linear"  -- history type to use
    self.histSpacing    = args.histSpacing or 1
    self.nonTermProb    = args.nonTermProb or 1
    self.bufferSize     = args.bufferSize or 512

    self.frames_per_subgoal = 4

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
        stateDim = self.state_dim, numActions = args.total_subgoals,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = 50000, --self.replay_memory, 
        histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = 512,
        subgoal_dims = args.subgoal_dims
    }
    self.meta_transitions = dqn.TransitionTable(meta_transition_args)
    --self.meta_transitions = dqn.TransitionTable(meta_transition_args)

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
    meta_args.n_units        = {32, 64, 64}
    meta_args.filter_size    = {8, 4, 3}
    meta_args.filter_stride  = {4, 2, 1}
    meta_args.n_hid          = {512}
    meta_args.nl             = nn.Rectifier
    meta_args.n_actions = args.total_subgoals
    meta_args.input_dims = self.input_dims
    self.meta_args = meta_args
    self.confusion = optim.ConfusionMatrix({"0","1"})


    -- create a meta network from scratch if not read in from saved file
    if not self.network_meta then
        print("Creating new Meta Network.....")
        require 'convnet_atari3'
       
        self.network_meta = create_network(meta_args)
    end

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

    self.criterion = nn.ClassNLLCriterion()
    -- self.criterion.sizeAverage = false

    if self.gpu and self.gpu >= 0 then
        self.criterion:cuda()
    end

    if self.target_q then
        self.target_network = self.network:clone()
        self.target_network_meta = self.network_meta:clone()
        self.w_target, self.dw_target = self.target_network:getParameters()
        self.w_meta_target, self.dw_meta_target = self.target_network_meta:getParameters()
    end

    -- load expert images
    self.expert_data = torch.zeros(self.total_subgoals*self.frames_per_subgoal, unpack(self.input_dims)) 
    self.expert_data_raw = torch.zeros(self.total_subgoals*self.frames_per_subgoal, 1, 1, self.input_dims[2], self.input_dims[3])
    for i=1,self.total_subgoals do
        for j=1,self.frames_per_subgoal do 
            expert_data = image.load('expert/' .. i .. '/' .. j .. '.png')
            expert_data = self:preprocess(expert_data):float()
            expert_data:resize(1, 1, self.input_dims[2], self.input_dims[3])
            self.expert_data_raw[(i-1)*self.frames_per_subgoal + j] = expert_data:clone()
            self.expert_data[(i-1)*self.frames_per_subgoal + j] = torch.cat(expert_data, torch.zeros(1, self.input_dims[1]-1, self.input_dims[2],self.input_dims[3]),2)
            if self.gpu >= 0 then
                self.expert_data=self.expert_data:cuda()
                -- self.expert_data_raw=self.expert_data_raw:cuda()
            end
        end
    end
    
    -- self.expert_data = torch.Tensor(self.expert_data)
    -- self.expert_data_raw = torch.Tensor(self.expert_data_raw)
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
    q2_max = target_q_net:forward(s2)[1]:float():max(2)

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
    local q_all, predicted_label = unpack(args.network:forward(s))
    q_all = q_all:float()
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

    return targets, delta, q2_max, predicted_label
end


-- function nql:get_residual(s)
--     local scaling = 1.0 --  10.0 --3
--     local s_reshaped = s:reshape(self.minibatch_size, self.input_dims[1], self.input_dims[2], self.input_dims[3])
--     local residual = s_reshaped:clone():zero()

--     for i=1,self.hist_len do
--         if i == 1 then
--             residual[{{},i,{},{}}] = s_reshaped[{{}, i,{},{}}] - s_reshaped[{{}, i+1,{},{}}]
--         elseif i == self.hist_len then
--             residual[{{},i,{},{}}] = s_reshaped[{{}, i,{},{}}] - s_reshaped[{{}, i-1,{},{}}]
--         else
--             local tmp1 = (s_reshaped[{{}, i,{},{}}] - s_reshaped[{{}, i-1,{},{}}]):abs()
--             local tmp2 = (s_reshaped[{{}, i,{},{}}] - s_reshaped[{{}, i+1,{},{}}]):abs()
--             residual[{{},i,{},{}}] = tmp1 + tmp2
--         end
--         residual[{{},i,{},{}}] = residual[{{},i,{},{}}]:abs()
--         residual[{{},i,{},{}}] = torch.ge(residual[{{},i,{},{}}], 0.001) --* (scaling-1)
--         -- residual[{{},i,{},{}}] = residual[{{},i,{},{}}] -- + 1.0
--     end
--     -- local res = residual:clone()
--     -- res = res[{{1, 4}}]:reshape(4, self.hist_len , 84,84); res = res[{{},1,{},{}}]
--     -- disp.image(res, {win=5, title='gradient scaling'})

--     -- residual = residual:reshape(self.minibatch_size*self.input_dims[1]*self.input_dims[2]*self.input_dims[3])
    
--     -- grads = grads:cmul(residual)
--     return residual
-- end




function nql:qLearnMinibatch(network, target_network, tran_table, dw, w, g, g2, tmp, deltas, external_r, nactions, metaFlag)
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    assert(tran_table:size() > self.minibatch_size)

    local s, a, r, s2, term, subgoals, subgoals2 = tran_table:sample(self.minibatch_size)
    subgoals = subgoals:reshape(subgoals:size(1))
    subgoals2 = subgoals2:reshape(subgoals2:size(1))
 

    --multiply by number of frames per subgoal and add random offset to sample a frame for each subgoal
    subgoals:add(-1):mul(self.frames_per_subgoal)
    subgoals2:add(-1):mul(self.frames_per_subgoal)

    sub_copy = subgoals:clone()
    sub_copy2 = subgoals2:clone()


    local rand_offset = torch.Tensor(subgoals:size(1))
    rand_offset:random(self.frames_per_subgoal):cuda()

    subgoals:add(rand_offset)
    subgoals2:add(rand_offset)

    expert_frames = self.expert_data_raw:index(1, subgoals:long())
    expert_frames2 = self.expert_data_raw:index(1, subgoals2:long())
    expert_frames = torch.squeeze(expert_frames, 2)
    expert_frames2  = torch.squeeze(expert_frames2, 2)
    if self.gpu >=0 then
        expert_frames = expert_frames:cuda()
        expert_frames2 = expert_frames2:cuda()
    end
    -- print(expert_frames:size())

  

    if metaFlag then        
        expert_frames:zero()
        expert_frames2:zero()

        s:resize(self.minibatch_size, self.input_dims[1]-1, self.input_dims[2], self.input_dims[3])
        s2:resize(self.minibatch_size, self.input_dims[1]-1, self.input_dims[2], self.input_dims[3])
        
        s = torch.cat(s, expert_frames, 2)
        s2 = torch.cat(s2, expert_frames, 2)

    else
        s:resize(self.minibatch_size, self.input_dims[1]-1, self.input_dims[2], self.input_dims[3])
        s2:resize(self.minibatch_size, self.input_dims[1]-1, self.input_dims[2], self.input_dims[3])
        
        s = torch.cat(s, expert_frames, 2)
        s2 = torch.cat(s2, expert_frames, 2)

    end

    --replace half the states with positive examples
    local rand_offset2 = torch.Tensor(self.minibatch_size/2, self.hist_len):random(self.frames_per_subgoal):cuda()
    sub_copy = sub_copy:repeatTensor(1, 2)
    local positive_frames = self.expert_data_raw:index(1, sub_copy:long())
    s[{{1, self.minibatch/2},{1,2}}] = positive_frames



    -- print(r, s:sum(2))
    if external_r then
        r = r[{{},1}] --extract external reward
        subgoals[{{},{1,self.subgoal_dims}}] = 0
        subgoals2[{{},{1,self.subgoal_dims}}] = 0
        if SUBGOAL_SCREEN then
            -- TODO
        end

    else    
        r = r[{{},2}] --external + intrinsic reward 
    end

    local targets, delta, q2_max, predicted_label = self:getQUpdate({s=s, a=a, r=r, s2=s2, n_actions = nactions,
        term=term, subgoals = subgoals, subgoals2=subgoals2, network = network, update_qmax=true, target_network = target_network}, external_r)

    --zero out half the Q-value targets
    targets[{{1, self.minibatch_size/2}}]:zero()

    -- zero gradients of parameters
    dw:zero()

    gold_label = torch.zeros(self.minibatch_size)
    gold_label[{{1, self.minibatch_size/2}}]:add(1)

    -- local s_residual = self:get_residual(s_flatten)
    local err = self.criterion:forward(predicted_label, gold_label)
    -- print(mse_err)
    local gradInput = self.criterion:backward(predicted_label, gold_label)

    if metaFlag then
        gradInput:zero()
    else
        -- reconsCriterion = self:motionScaling(s_flatten, reconsCriterion)
        -- disp.image(s[{{1,4},{4},{},{}}], {win=3, title='observed'}) 
        for i = 1,self.minibatch_size do
            confusion:add(predicted_label[i], gold_label[i])
         end
        -- disp.image(reconstruction[{{1,4},{4},{},{}}], {win=4, title='predictions'}) 
        -- disp.image(s_residual[{{1,4},{4},{},{}}], {win=5, title='observed-residual'})
    end

    -- get new gradient
    -- print(subgoals)
    network:backward(s, {targets, gradInput})

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

-- returns a table of num_objects x vectorized object reps
function nql:get_objects(rawstate)
    image.save('tmp_' .. ZMQ_PORT .. '.png', rawstate[1])
    skt:send("")
    msg = skt:recv()
    while msg == nil do
        msg = skt:recv()
    end
    local object_list = process_pystr(msg)
    self.objects = object_list
    return object_list --nn.SplitTable(1):forward(torch.rand(4, self.subgoal_dims))  
end

function nql:pick_subgoal(rawstate, metareward, terminal, testing, testing_ep)
    if false then
    local objects = self:get_objects(rawstate)

    local state = self:preprocess(rawstate):float()

    self.meta_transitions:add_recent_state(state, terminal, torch.Tensor({0}))  

    --Store transition s, a, r, s'

    if self.metalastState and not testing then
        self.meta_transitions:add(self.metalastState, self.metalastAction, torch.Tensor({metareward, metareward + 0}),
                        self.metalastTerminal, torch.Tensor({0}), priority)
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
        actionIndex, qfunc = self:eGreedy('meta', self.network_meta, curState, testing_ep, self.metalastAction)
    end

    self.meta_transitions:add_recent_action(actionIndex) 

    --Do some Q-learning updates
    if self.metanumSteps > self.meta_learn_start and not testing and
        self.metanumSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:qLearnMinibatch(self.network_meta, self.target_network_meta, self.meta_transitions,
             self.dw_meta, self.w_meta, self.g_meta, self.g2_meta, self.tmp_meta, self.deltas_meta, false, self.meta_args.n_actions, true)
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

    if not terminal then
        self.subgoal_total[actionIndex] = self.subgoal_total[actionIndex] or 0
        self.subgoal_total[actionIndex] = self.subgoal_total[actionIndex] + 1

        self.global_subgoal_total[actionIndex] = self.global_subgoal_total[actionIndex] or 0
        self.global_subgoal_total[actionIndex] = self.global_subgoal_total[actionIndex] + 1
    end

    -- keep track of subgoal sequences
    if terminal then
        table.insert(self.global_subgoal_seq, self.subgoal_seq)
        self.subgoal_seq = {}
    else
        table.insert(self.subgoal_seq, actionIndex)
    end
    -- return subgoal    
    return actionIndex
    end
    return torch.random(2)
end

function nql:isGoalReached(subgoal, state)
    local s = state:clone()
    s:resize(1, self.input_dims[1]-1, self.input_dims[2], self.input_dims[3])
    local expert_frame = self.expert_data_raw[(subgoal-1)*self.frames_per_subgoal + torch.random(self.frames_per_subgoal)]
    
    s = torch.cat(s, expert_frame,2)
    
    -- get features of state
    if self.gpu >= 0 then s=s:cuda() end

    _, pred_label = unpack(self.target_network:forward(s))
    pred_label = pred_label:squeeze()
    pred_label = pred_label:max(1)[1]
    pred_label = pred_label - 1
    if pred_label == 1 then
        return true
    else
        return false
    end
end


function nql:perceive(subgoal, reward, rawstate, terminal, testing, testing_ep)
    -- Preprocess state (will be set to nil if terminal)
    
    local state = self:preprocess(rawstate):float()
    local objects = self:get_objects(rawstate)

    if terminal then
        -- this deathPosition business is hacky as hell but it fixes bug in ALE engine for MZ
        self.deathPosition = objects[1][{{1,2}}] --just store the x and y coords of the agent
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

    local subgoal_vec = torch.ones(1)
    subgoal_vec[1] = subgoal

    self.transitions:add_recent_state(state, terminal, subgoal_vec)  
    curState, subgoal = self.transitions:get_recent()    
    subgoal = subgoal[1]

    local intrinsic_reward = 0
    local goal_reached = self:isGoalReached(subgoal,curState)
    if self.numSteps < 2 * self.learn_start then -- make sure classifier is stable before relying on goal predictions
        goal_reached = false
    end
    if goal_reached then
        intrinsic_reward = intrinsic_reward + 50  --binary reward for reaching the goal
    end
    if terminal then
        intrinsic_reward = intrinsic_reward - 200
    end
    intrinsic_reward = intrinsic_reward - 0.1


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


    curState:resize(1, self.input_dims[1]-1, self.input_dims[2], self.input_dims[3])
    -- curState = torch.cat(curState, torch.zeros(1, 1, self.input_dims[2],self.input_dims[3]),2)
    curState = torch.cat(curState, self.expert_data_raw[subgoal], 2)

    -- Select action
    local actionIndex = 1
    local qfunc
    if not terminal then
        actionIndex, qfunc = self:eGreedy('lower', self.network, curState, testing_ep)
        --self:eGreedy(curState, testing_ep, subgoal)
    end

    self.transitions:add_recent_action(actionIndex) 

    --Do some Q-learning updates
    if self.numSteps > self.learn_start and not testing and
        self.numSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:qLearnMinibatch(self.network, self.target_network, self.transitions,
                self.dw, self.w, self.g, self.g2, self.tmp, self.deltas, false, self.n_actions)

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
        -- print("Getting subgoal")
        self.lastSubgoal = subgoal
        -- print("lastsubgoal", self.lastSubgoal)
        --check if the game is still in the stages right after the agent dies
        if self.deathPosition then
            -- this deathPosition business is hacky as hell but it fixes bug in ALE engine for MZ
            currentPosition = objects[1][{{1,2}}]
            -- print("Positions:", currentPosition, self.deathPosition)
            if math.sqrt((currentPosition[1]-self.deathPosition[1])^2 + (currentPosition[2]-self.deathPosition[2])^2) < self.DEATH_THRESHOLD then
                self.lastSubgoal = nil
                -- print("death overruling")
            else
                -- print("Removing death position", self.deathPosition)
                self.deathPosition = nil
                self.ignoreState = 1
            end
        end
    else
        -- print("LAST SUBGOAL is now NIL")
        -- self.lastSubgoal = nil
    end

    self.lastobjects = objects

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

    if not terminal then
        return actionIndex, goal_reached, reward, reward+intrinsic_reward, qfunc
    else
        return 0, goal_reached, reward, reward+intrinsic_reward, qfunc
    end
end


function nql:eGreedy(mode, network, state, testing_ep, lastsubgoal)
    -- handle learn_start
    if mode == 'meta' then
        learn_start = self.meta_learn_start
    else
        learn_start = self.learn_start
    end
    self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - learn_start))/self.ep_endt))
   
    -- local subgoal_id = subgoal[#subgoal]
    -- if mode ~= 'meta' and  subgoal_id ~= 6 and subgoal_id ~= 8 then -- TODO: properly update later using running hit rate
    --     self.ep = 0.1
    -- end

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
        return self:greedy(network, n_actions, state, lastsubgoal)
    end
end


function nql:greedy(network, n_actions,  state, lastsubgoal)
    -- Turn single state into minibatch.  Needed for convolutional nets.
    if state:dim() == 2 then
        assert(false, 'Input must be at least 3D')
        state = state:resize(1, state:size(1), state:size(2))
    end
    -- subgoal = torch.reshape(subgoal, 1, self.subgoal_dims*9)
    if self.gpu >= 0 then
        state = state:cuda()
        -- subgoal = subgoal:cuda()
    end
    local q, unused = unpack(network:forward(state))
    q = q:float():squeeze()
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
            print("Subgoal ID (8-key, 6/7-bottom ladders):" , subg , ' : ',  self.subgoal_success[subg]/val, self.subgoal_success[subg] .. '/' .. val)
        else
            print("Subgoal ID (8-key, 6/7-bottom ladders):" , subg ,  ' : ')
        end
    end

    torch.save(filename , {self.subgoal_success, self.subgoal_total, self.global_subgoal_seq})
    self.subgoal_success = {}
    self.subgoal_total = {}
    self.global_subgoal_seq = {}
end
