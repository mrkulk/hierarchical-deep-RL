import random
import sys
import copy
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation

from random import Random
import zmq, argparse
import numpy as np


socket = None #will be initialized in main


def sendMsg(state, reward, terminal):
    global socket
    if terminal:
        terminal = 'true'
    else:
        terminal = 'false'
    outMsg = 'state, reward, terminal = ' + str(state) + ',' + str(reward)+','+terminal
    socket.send(outMsg.replace('[', '{').replace(']', '}'))
    

class skeleton_agent(Agent):
    randGenerator=Random()
    lastAction=Action()
    lastObservation=Observation()

    def agent_init(self,taskSpec):
        #See the sample_sarsa_agent in the mines-sarsa-example project for how to parse the task spec
        self.lastAction=Action()
        self.lastObservation=Observation()

    def agent_start(self,observation):
        #Generate random action, 0 or 1
        # thisIntAction=self.randGenerator.randint(0,1)
        # returnAction=Action()
        # returnAction.intArray=[thisIntAction]

        # lastAction=copy.deepcopy(returnAction)
        # lastObservation=copy.deepcopy(observation)


        #pass observation via zmq to lua agent and then return action
        state = np.array(list(observation.intArray))
        reward = 0.
        terminal = False

        sendMsg(state, reward, terminal)
        returnAction = int(socket.recv())

        self.lastObservation = copy.deepcopy(observation)


        return returnAction

    def agent_step(self,reward, observation):

     
        state = np.array(list(observation.intArray))
        reward = reward
        terminal = False

        sendMsg(state, reward, terminal)
        returnAction = int(socket.recv())

        self.lastObservation = copy.deepcopy(observation)

        return returnAction


    def agent_end(self,reward):
        state = np.array(list(self.lastObservation.intArray))
        reward = reward
        terminal = True

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

    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--port",
        type = int,
        default = 5050,
        help = "port for server")

    args = argparser.parse_args()

    #-------------------------------------------------

    #server setup
    port = args.port
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)
    print "Started server on port", port

    AgentLoader.loadAgent(skeleton_agent())