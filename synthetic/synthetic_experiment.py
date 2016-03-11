# 
# Copyright (C) 2008, Brian Tanner
# 
#http://rl-glue-ext.googlecode.com/
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys, argparser, zmq

import rlglue.RLGlue as RLGlue

whichEpisode=0

argparser = argparse.ArgumentParser(sys.argv[0])
argparser.add_argument("--port",
    type = int,
    default = 5051,
    help = "port for server")

args = argparser.parse_args()

def runEpisode(stepLimit):
	global whichEpisode
	terminal=RLGlue.RL_episode(stepLimit)
	totalSteps=RLGlue.RL_num_steps()
	totalReward=RLGlue.RL_return()
	
	print "Episode "+str(whichEpisode)+"\t "+str(totalSteps)+ " steps \t" + str(totalReward) + " total reward\t " + str(terminal) + " natural end"
	
	whichEpisode=whichEpisode+1

#Main Program starts here

print "\n\nExperiment starting up!"
taskSpec = RLGlue.RL_init()
print "RL_init called, the environment sent task spec: " + taskSpec

print "\n\n----------Sending some sample messages----------"

#Talk to the agent and environment a bit...*/
responseMessage = RLGlue.RL_agent_message("what is your name?")
print "Agent responded to \"what is your name?\" with: " + responseMessage

responseMessage = RLGlue.RL_agent_message("If at first you don't succeed; call it version 1.0")
print "Agent responded to \"If at first you don't succeed; call it version 1.0  \" with: " + responseMessage + "\n"

responseMessage = RLGlue.RL_env_message("what is your name?")
print "Environment responded to \"what is your name?\" with: " + responseMessage
responseMessage = RLGlue.RL_env_message("If at first you don't succeed; call it version 1.0")
print "Environment responded to \"If at first you don't succeed; call it version 1.0  \" with: " + responseMessage

print "\n\n----------Running a few episodes----------"

while True:
    message = socket.recv()
    if message == "newGame":
        runEpisode(100)
    else:
        print "Invalid message!!"
        break

RLGlue.RL_cleanup()



