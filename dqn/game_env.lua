

local env = torch.class('GameEnv')

local json = require ("dkjson")
local zmq = require "lzmq"

if pcall(require, 'signal') then
    signal.signal("SIGPIPE", function() print("raised") end)
else
    print("No signal module found. Assuming SIGPIPE is okay.")
end

function env:__init(args)
    -- for agent
    self.ctx = zmq.context()
    self.skt = self.ctx:socket{zmq.REQ,
        linger = 0, rcvtimeo = 10000;
        connect = "tcp://127.0.0.1:" .. args.zmq_port;
    }

    -- for exp
    self.ctx2 = zmq.context()
    self.skt2 = self.ctx2:socket{zmq.REQ,
        linger = 0, rcvtimeo = 10000;
        connect = "tcp://127.0.0.1:" .. tostring(args.zmq_port+1);
    }
end

function env:process_msg(msg)    
    -- screen, reward, terminal
    -- print("MESSAGE:", msg)
    loadstring(msg)()
    -- if reward ~= 0 then
    --     print('non-zero reward', reward)
    -- end    
    return torch.Tensor(state), reward, terminal
end

function env:newGame()
    self.skt2:send("newGame")
    msg = self.skt2:recv()
    while msg == nil do
        msg = self.skt:recv()
    end
    return self:process_msg(msg)
end


function env:step(action, ignore_flag)
    assert(action==1 or action==0, "Action " .. tostring(action))
    self.skt:send(tostring(action))
    msg = self.skt:recv()
    while msg == nil do
        msg = self.skt:recv()
    end
    return self:process_msg(msg)
end

function env:evalStart()
    self.skt:send("evalStart")
    msg = self.skt:recv()
    assert(msg == 'done', msg)
end

function env:evalEnd()
    self.skt:send("evalEnd")
    msg = self.skt:recv()
    assert(msg == 'done', msg)
end


function env:getActions()   
    return {0,1} -- actions 
end
