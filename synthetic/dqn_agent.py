import random
import sys
import pdb
import copy
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation

from random import Random
import zmq
import numpy as np


socket = None #will be initialized in main


def sendMsg(state, reward, terminal):
    global socket
    if terminal:
        terminal = 'true'
    else:
        terminal = 'false'
    outMsg = 'state, reward, terminal = ' + str(state) + ',' + str(reward)+','+terminal
    # print "DQN out:", outMsg
    socket.send(outMsg.replace('[', '{').replace(']', '}'))
    # print "Done sending"
    

class skeleton_agent(Agent):
    randGenerator=Random()
    lastAction=Action()
    lastObservation=Observation()

    def agent_init(self,taskSpec):
        #See the sample_sarsa_agent in the mines-sarsa-example project for how to parse the task spec
        self.lastAction=Action()
        self.lastObservation=Observation()

    def agent_start(self,observation):
        # print "Calling start"
        dummy_msg = socket.recv()  #since zmq server-client pattern expects server to receive before sending
        assert dummy_msg == 'dummy'
        #Generate random action, 0 or 1
        # thisIntAction=self.randGenerator.randint(0,1)
        
        

        # lastAction=copy.deepcopy(returnAction)
        # lastObservation=copy.deepcopy(observation)


        #pass observation via zmq to lua agent and then return action
        state = np.array(list(observation.intArray))
        reward = 0.
        terminal = False

        sendMsg(state, reward, terminal)

        # print "Waiting to receive...."
        returnAction=Action()
        returnAction.intArray=[int(socket.recv())]

        self.lastObservation = copy.deepcopy(observation)


        return returnAction

    def agent_step(self,reward, observation):

        # print "Calling step"
     
        state = np.array(list(observation.intArray))
        reward = reward
        terminal = False

        sendMsg(state, reward, terminal)
        # print "Waiting to receive...."
        returnAction=Action()
        returnAction.intArray=[int(socket.recv())]
        # print "--------------------------- DQN received action:", returnAction

        self.lastObservation = copy.deepcopy(observation)
        # print "lastobs", self.lastObservation.intArray

        return returnAction


    def agent_end(self,reward):
        state = np.array(list(self.lastObservation.intArray))
        if state[0] == 2: 
            state[0] = 1
        elif state[0] == 5: 
            state[0] = 6
        reward = reward
        terminal = True
        # print "Calling end function", state, reward, terminal
        sendMsg(state, reward, terminal)
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self,inMessage):
        if inMessage=="what is your name?":
            return "my name is skeleton_agent, Python edition!";
        else:
            return "I don't know how to respond to your message";


if __name__=="__main__":

    port = int(sys.argv[1])

    #-------------------------------------------------

    #server setup
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)
    print "Started server on port", port

    AgentLoader.loadAgent(skeleton_agent())