#
# Copyright (C) 2007, Mark Lee
#
#http://rl-glue-ext.googlecode.com/
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
import numpy as np
import random
import sys
from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal
# /**
#  *  This is a very simple environment with discrete observations corresponding to states labeled {0,1,...,19,20}
#     The starting state is 10.
#
#     There are 2 actions = {0,1}.  0 decrements the state, 1 increments the state.
#
#     The problem is episodic, ending when state 0 or 20 is reached, giving reward -1 or +1, respectively.  The reward is 0 on
#     all other steps.
#  * @author Brian Tanner
#  */

class skeleton_environment(Environment):
  

    def env_init(self):
        self.LEFT = 1
        self.RIGHT = 2
        return "VERSION RL-Glue-3.0 PROBLEMTYPE episodic DISCOUNTFACTOR 1.0 OBSERVATIONS INTS (0 5)  ACTIONS INTS (0 1)  REWARDS (1/100 1.0)."

    def env_start(self):
        self.currentState=2
        self.met_subgoal = False
        returnObs=Observation()
        returnObs.intArray=[self.currentState]
        self.path_list = [str(self.currentState)]
        return returnObs

    def env_step(self,thisAction):
        episodeOver=0
        theReward=0

        if thisAction.intArray[0]==self.LEFT:
            self.currentState=self.currentState-1
        if thisAction.intArray[0]==self.RIGHT:
            sampled_action =np.random.choice([self.LEFT,self.RIGHT])
            # sampled_action =np.random.choice([self.RIGHT])
            if sampled_action == self.LEFT:
                self.currentState=self.currentState-1
            else:
                self.currentState=self.currentState+1



        if self.currentState <= 1:
            self.currentState=1
            if not self.met_subgoal:
                theReward= 1/100
                episodeOver=1
            else:
                theReward= 1
                episodeOver=1

        if self.currentState >= 6:
            self.currentState=6
            self.met_subgoal = True
            #theReward=1.
            #episodeOver=1

        self.path_list.append(str(self.currentState))
        if episodeOver == 1:
            tempfile = open('results_synth_paths.txt', 'ab')
            tempfile.write(''.join(self.path_list))
            tempfile.write('\n')

        theObs=Observation()
        theObs.intArray=[self.currentState]

        returnRO=Reward_observation_terminal()
        returnRO.r=theReward
        returnRO.o=theObs
        returnRO.terminal=episodeOver
        # print 'reward', returnRO.r
        # print 'observation1', np.array(list(returnRO.o.intArray))
        # print 'terminal', returnRO.terminal
        # print ''
        return returnRO

    def env_cleanup(self):
        pass

    def env_message(self,inMessage):
        if inMessage=="what is your name?":
            return "my name is synthetic_environment, Python edition!";
        else:
            return "I don't know how to respond to your message";


if __name__=="__main__":
    EnvironmentLoader.loadEnvironment(skeleton_environment())