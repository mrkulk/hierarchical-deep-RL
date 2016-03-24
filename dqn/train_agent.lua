--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]
require 'xlua'
require 'optim'

-- require 'signal'
-- signal.signal("SIGPIPE", function() print("raised") end)

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-subgoal_index', 12, 'the index of the subgoal that we want to reach. used for slurm multiple runs')
cmd:option('-max_subgoal_index', 12, 'used as an index to run with all the subgoals instead of only one specific one')

cmd:option('-exp_folder', '', 'name of folder where current exp state is being stored')
cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', true,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')

cmd:option('-subgoal_dims', 1, 'dimensions of subgoals')
cmd:option('-subgoal_nhid', 50, '')
cmd:option('-display_game', false, 'option to display game')
cmd:option('-port', 5550, 'Port for zmq connection')
cmd:option('-stepthrough', false, 'Stepthrough')
cmd:option('-subgoal_screen', true, 'overlay subgoal on screen')

cmd:option('-max_steps_episode', 5000, 'Max steps per episode')

cmd:option('-meta_agent', true, 'hierarchical training')
cmd:option('-max_objects', 6, 'max number of objects in scene that are parsed and used as subgoals')
cmd:option('-gif', false, 'gif on/off')




cmd:text()

local opt = cmd:parse(arg)
ZMQ_PORT = opt.port
SUBGOAL_SCREEN = opt.subgoal_screen
META_AGENT = opt.meta_agent

if not dqn then
    require "initenv"
end

print("NETWORK", opt.network)

print(opt.env_params)
print(opt.seed)

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local step = 0
time_history[1] = 0

local total_reward
local nrewards
local nepisodes
local episode_reward

print("Calling NEWGAME ---------------------")
local screen, reward, terminal = game_env:newGame()


print("Iteration ..", step)
local win = nil

local subgoal

if META_AGENT then
    subgoal = agent:pick_subgoal(screen, 0, terminal, false)
else
    if opt.subgoal_index < opt.max_subgoal_index then 
        subgoal = agent:pick_subgoal(screen, opt.subgoal_index)
    else
        subgoal = agent:pick_subgoal(screen)
    end
end


death_counter = 0 --to handle a bug in MZ atari

episode_step_counter = 0
metareward = 0
SAVE_NET_EXIT = false
cum_metareward = 0 
numepisodes = 0

test_avg_R = test_avg_R or optim.Logger(paths.concat(opt.exp_folder , 'test_avgR.log'))

while step < opt.steps do
    xlua.progress(step, opt.steps)

    step = step + 1    

    local action_index, isGoalReached, reward_ext, reward_tot,  qfunc = agent:perceive(subgoal, reward, screen, terminal)
    metareward = metareward + reward_ext

    if opt.stepthrough then
        print("Reward Ext", reward_ext)
        print("Reward Tot", reward_tot)
        print("Q-func")
        if qfunc then
            for i=1, #action_list do
                print(action_list[i], qfunc[i])
            end
        end

        -- print("Action", action_index, action_list[action_index])
        -- io.read()
    end

    -- game over? get next game!
    if not terminal and  episode_step_counter < opt.max_steps_episode then

        screen, reward, terminal = game_env:step(game_actions[action_index], true)

        episode_step_counter = episode_step_counter + 1
        prev_Q = qfunc 
    else
        -- print("TERMINAL ENCOUNTERED")
        if META_AGENT then
            -- Note: this screen is the death screen (terminal)
            subgoal = agent:pick_subgoal(screen, metareward, true, false)
            cum_metareward = cum_metareward + metareward
            metareward = 0
        end
    
        screen, reward, terminal = game_env:newGame()

        new_game = true
        isGoalReached = true --new game so reset goal
        episode_step_counter = 0
        numepisodes = numepisodes+1
    end
  
    if isGoalReached then



        if META_AGENT then
            if metareward > 0 then 
                -- print("METAREWARD: ", metareward, "| subgoal:", subgoal[-1])
                -- io.read()
            end
            subgoal = agent:pick_subgoal(screen, metareward, terminal, false)
            -- print("Picked new subgoal", subgoal)
            cum_metareward = cum_metareward + metareward
            metareward = 0
        else        
            subgoal = agent:pick_subgoal(screen)            
        end

        isGoalReached = false
    end


    if step % opt.prog_freq == 0 then
        assert(step==agent.numSteps, 'trainer step: ' .. step ..
                ' & agent.numSteps: ' .. agent.numSteps)
        print("Steps: ", step)
        
        agent:report(paths.concat(opt.exp_folder , 'subgoal_statistics_' .. step .. '.t7'))
        collectgarbage()
    end


    if step%1000 == 0 then collectgarbage() end

    -- evaluation
    if step % opt.eval_freq == 0 and step > learn_start then
        cum_metareward = cum_metareward / math.max(1,numepisodes)
        test_avg_R:add{['% Average Meta Reward'] = cum_metareward}
        test_avg_R:style{['% Average Meta Reward'] = '-'}; test_avg_R:plot()
        numepisodes = 0
        cum_metareward = 0
    end


    if SAVE_NET_EXIT or (step % opt.save_freq == 0 or step == opt.steps) then
        local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
            agent.valid_s2, agent.valid_term
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = nil, nil, nil, nil, nil, nil, nil
        local w_meta, dw_meta, g_meta, g2_meta, delta, delta2, deltas, deltas_meta, tmp_meta = agent.w_meta, agent.dw_meta,
            agent.g_meta, agent.g2_meta, agent.delta, agent.delta2, agent.deltas, agent.deltas_meta, agent.tmp_meta
        agent.w_meta, agent.dw_meta, agent.g_meta, agent.g2_meta, agent.delta, agent.delta2, agent.deltas, 
            agent.deltas_meta, agent.tmp_meta = nil, nil, nil, nil, nil, nil, nil, nil, nil

        local filename = opt.name
        if opt.save_versions > 0 then
            filename = filename .. "_" .. math.floor(step / opt.save_versions)
        end
        filename = filename
        torch.save(filename .. ".t7", {agent = agent,
                                model = agent.network,
                                best_model = agent.best_network,
                                model_meta = agent.network_meta, 
                                best_model_meta = agent.best_network_meta,
                                reward_history = reward_history,
                                reward_counts = reward_counts,
                                episode_counts = episode_counts,
                                time_history = time_history,
                                v_history = v_history,
                                td_history = td_history,
                                qmax_history = qmax_history,
                                arguments=opt})
        if opt.saveNetworkParams then
            local nets = {network=w_meta:clone():float()}
            torch.save(filename..'.params.t7', nets, 'ascii')
        end
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = s, a, r, s2, term
        agent.w_meta, agent.dw_meta, agent.g_meta, agent.g2_meta, agent.delta, agent.delta2, agent.deltas,
            agent.deltas_meta, agent.tmp_meta = w_meta, dw_meta, g_meta, g2_meta, delta, delta2, deltas, deltas_meta, tmp_meta
        print('Saved:', filename .. '.t7')
        io.flush()
        collectgarbage()

        if SAVE_NET_EXIT then
            io.read()
            print('Halted ..... ')
            SAVE_NET_EXIT = false
        end
    end
end
