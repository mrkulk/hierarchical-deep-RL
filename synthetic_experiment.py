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

import sys

import rlglue.RLGlue as RLGlue

whichEpisode=0

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
runEpisode(100)
runEpisode(100)
runEpisode(100)
runEpisode(100)
runEpisode(100)
runEpisode(1)
# Remember that stepLimit of 0 means there is no limit at all!*/
runEpisode(0)
RLGlue.RL_cleanup()

print "\n\n----------Stepping through an episode----------"
#We could also start over and do another experiment */
taskSpec = RLGlue.RL_init()

#We could run one step at a time instead of one episode at a time */
#Start the episode */
startResponse = RLGlue.RL_start()

firstObservation = startResponse.o.intArray[0]
firstAction = startResponse.a.intArray[0]
print "First observation and action were: " + str(firstObservation) + " and: " + str(firstAction)

#Run one step */
stepResponse = RLGlue.RL_step()

#Run until the episode ends*/
while (stepResponse.terminal != 1):
    stepResponse = RLGlue.RL_step()
    #if (stepResponse.terminal != 1) 
        #Could optionally print state,action pairs */
        #printf("(%d,%d) ",stepResponse.o.intArray[0],stepResponse.a.intArray[0])*/

print "\n\n----------Summary----------"

totalSteps = RLGlue.RL_num_steps()
totalReward = RLGlue.RL_return()
print "It ran for " + str(totalSteps) + " steps, total reward was: " + str(totalReward)
RLGlue.RL_cleanup()



