--[[
Copyright (c) 2014 Google Inc.
See LICENSE file for full terms of limited license.
]]

gd = require "gd"
require 'torch'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

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
cmd:option('-seed', torch.random(0,10000), 'fixed input seed for repeatable experiments')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
cmd:option('-gif_file', '', 'GIF path to write session screens')
cmd:option('-csv_file', '', 'CSV path to write session data')
cmd:option('-subgoal_dims', 7, 'dimensions of subgoals')
cmd:option('-subgoal_nhid', 50, '')
cmd:option('-port', 5550, 'Port for zmq connection')
cmd:option('-stepthrough', false, 'Stepthrough')
cmd:option('-human_input', false, 'Human input action')
cmd:option('-subgoal_screen', false, 'overlay subgoal on screen')



cmd:text()

local opt = cmd:parse(arg)
ZMQ_PORT = opt.port


if not dqn then
    require "initenv"
end

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

-- file names from command line
local gif_filename = opt.gif_file

-- start a new game
local screen, reward, terminal = game_env:newGame()

-- compress screen to JPEG with 100% quality
local jpg = image.compressJPG(screen:squeeze(), 100)
-- create gd image from JPEG string
local im = gd.createFromJpegStr(jpg:storage():string())
-- convert truecolor to palette
im:trueColorToPalette(false, 256)

-- write GIF header, use global palette and infinite looping
im:gifAnimBegin(gif_filename, true, 0)
-- write first frame
im:gifAnimAdd(gif_filename, false, 0, 0, 7, gd.DISPOSAL_NONE)

-- remember the image and show it first
local previm = im
local win = image.display({image=screen})

print("Started playing...")

subgoal = agent:pick_subgoal(screen, 6)
--print('Subgoal:', subgoal)


local action_list = {'no-op', 'fire', 'up', 'right', 'left', 'down', 'up-right','up-left','down-right','down-left',
                    'up-fire', 'right-fire','left-fire', 'down-fire','up-right-fire','up-left-fire',
                    'down-right-fire', 'down-left-fire'}

-- play one episode (game)
while true or not terminal do
    -- if action was chosen randomly, Q-value is 0
    agent.bestq = 0

    if opt.subgoal_screen then
        screen[{1,{}, {30+subgoal[1]-5, 30+subgoal[1]+5}, {subgoal[2]-5,subgoal[2]+5} }] = 1
        win = image.display({image=screen, win=win})
    end
    
    -- choose the best action
    local action_index, isGoalReached, reward_ext, reward_tot, qfunc 
    = agent:perceive(subgoal, reward, screen, terminal, true, 0.0)

    local tmp2

    if opt.stepthrough then
        print("Reward Ext", reward_ext)
        print("Reward Tot", reward_tot)
        print("Q-func")
        if qfunc then
            for i=1, #action_list do
                print(string.format("%s %.4f", action_list[i], qfunc[i]))
            end
        end
        print("Action", action_index, action_list[action_index])
        tmp2 = io.read()
    end

    --human input of action
    if tmp2=='y' or opt.human_input then
        print("Enter action")
        local tmp = io.read()
        if tmp then
            action_index = tonumber(tmp)
        end

    end

    -- play game in test mode (episodes don't end when losing a life if false below)
    screen, reward, terminal = game_env:step(game_actions[action_index], false)
        -- screen, reward, terminal = game_env:step(game_actions[1], false) --no-op

    -- screen, reward, terminal = game_env:step(game_actions[action_index])




    if isGoalReached then
        subgoal = agent:pick_subgoal(screen)
    end


    if not opt.subgoal_screen then
        screen_cropped = screen:clone()
        screen_cropped = screen_cropped[{{},{},{30,210},{1,160}}]
        screen_cropped[{1,{}, {subgoal[1]-5, subgoal[1]+5}, {subgoal[2]-5,subgoal[2]+5} }] = 1
        
        -- display screen
        image.display({image=screen_cropped, win=win})
    end

    -- create gd image from tensor
    jpg = image.compressJPG(screen:squeeze(), 100)
    im = gd.createFromJpegStr(jpg:storage():string())
    
    -- use palette from previous (first) image
    im:trueColorToPalette(false, 256)
    im:paletteCopy(previm)

    -- write new GIF frame, no local palette, starting from left-top, 7ms delay
    im:gifAnimAdd(gif_filename, false, 0, 0, 7, gd.DISPOSAL_NONE)
    -- remember previous screen for optimal compression
    previm = im

end

-- end GIF animation and close CSV file
gd.gifAnimEnd(gif_filename)

print("Finished playing, close window to exit!")
