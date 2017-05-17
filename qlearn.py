#!/usr/bin/env python
# from __future__ import print_function

import argparse
import sys
# sys.path.append("game/")
import main as game
import random
import numpy as np
from collections import deque

import json
# from keras import initializations
# from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf
import keras

GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 4 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 0. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

# img_rows , img_cols = 80, 80
#Convert image into Black and white
# img_channels = 4 #We stack 4 frames
class QLearn:
    def __init__(self, layers, activation_fns, model_file=None, bias = False, name = ''):
        self.D = deque()
        print("Now we build the model")
        self.model = Sequential()
        w1 = keras.initializers.RandomNormal(mean=0, stddev=1, seed=None)
        w2 = keras.initializers.RandomNormal(mean=0, stddev=1, seed=None)
        self.model.add(Dense(layers[0], input_dim=2, init=w1, activation=activation_fns[0],  use_bias=False ))
        self.model.add(Dense(layers[1], init=w2, activation=activation_fns[0], use_bias=False))
        self.model.add(Dense(layers[2]))
        self.model.add(Activation(activation_fns[1]))
        # self.model.compile(loss='mean_squared_error', optimizer='rmsprop')
            
        adam = Adam(lr=LEARNING_RATE)
        self.model.compile(loss='mse',optimizer=adam)
        print("We finished building the model")

    def get_all_weights(self):
        # need weights of layers 1 & 2 - everything except first and last
        w = []
        # # print "No of layers: ", len(self.model.layers)
        # w.append(self.w1.eval(session = self.session))
        # w.append(self.w2.eval(session = self.session))
        for i in xrange(1,len(self.model.layers)-1):
            # print self.model.layers[i].get_weights()
            w.append(self.model.layers[i].get_weights())
        return w

    def set_all_weights(self, weights):
        # pass
        print weights
        # self.session.run(tf.assign(self.w1, weights[0]))
        # self.session.run(tf.assign(self.w2, weights[1]))
        for i in xrange(1,len(self.model.layers)-1):
            w = np.array(weights[i-1])
            self.model.layers[i].set_weights(weights[i-1])


    # needs to call population to get environment of each bot
    # or pop needs to call this for each bot
    # each bot calls this
    def output(self, input_, reward, dead):
        # open up a game state to communicate with emulator
        # game_state = game.Population()

        # store the previous observations in replay memory

        # get the first state by doing nothing
        # do_nothing = np.zeros(ACTIONS)
        # set default action to MOVE_FORWARD
        # do_nothing[0] = 1
        # current state, reward and terminal from the environment after applying default action
        s_t, r_0, terminal = input_, reward, dead
        a_t = np.zeros([ACTIONS])
        epsilon = INITIAL_EPSILON

        if random.random() <= epsilon:
            print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            # q = self.model.predict(s_t)       #input a stack of 4 images, get the prediction
            q = self.model.predict(np.array([np.array(s_t)]).reshape((1,2)))[0]
            max_Q = np.argmax(q)
            action_index = max_Q
            a_t[max_Q] = 1
        # print "action: ",a_t
        return a_t

    def train_network(self, input_, action_index, reward, dead, next_state, args='train'):

        if args == 'run':
            OBSERVE = 999999999    #We keep observe, never train
            epsilon = FINAL_EPSILON
            print ("Now we load weight")
            model.load_weights("model.h5")
            adam = Adam(lr=LEARNING_RATE)
            model.compile(loss='mse',optimizer=adam)
            print ("Weight load successfully")    
        else:                       #We go to training mode
            OBSERVE = OBSERVATION
            epsilon = INITIAL_EPSILON

        t = 0
        # while (True):
        #     print "Time: ",t
        loss = 0
        Q_sa = 0
        # action_index = 0
        r_t = 0
        # a_t = np.zeros([ACTIONS])
        # #choose an action epsilon greedy
        # if t % FRAME_PER_ACTION == 0:
        #     if random.random() <= epsilon:
        #         print("----------Random Action----------")
        #         action_index = random.randrange(ACTIONS)
        #         a_t[action_index] = 1
        #     else:
        #         q = self.model.predict(s_t)       #input a stack of 4 images, get the prediction
        #         max_Q = np.argmax(q)
        #         action_index = max_Q
        #         a_t[max_Q] = 1
            #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        s_t1, r_t, terminal = next_state, reward, dead

            
        # store the transition in D
        self.D.append((input_, action_index, r_t, s_t1, terminal))
        if len(self.D) > REPLAY_MEMORY:
            self.D.popleft()


        #only train if done observing
        if len(self.D) >= BATCH:
            #sample a minibatch to train on
            minibatch = random.sample(self.D, BATCH)
            print "minibatch: ", minibatch
            inputs = np.zeros((len(minibatch), 2))   #32, 80, 80, 4
            print (inputs.shape)
            targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

            #Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]

                print state_t, action_t, reward_t, state_t1, terminal
                # raw_input()
                # if terminated, only equals reward

                inputs[i:i + 1] = np.array(state_t)    #I saved down s_t

                targets[i] = self.model.predict(np.array([np.array(state_t)]).reshape((1,2)))[0]  # Hitting each buttom probability
                Q_sa = self.model.predict(np.array([np.array(state_t1)]).reshape((1,2)))[0]

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            # targets2 = normalize(targets)
            loss += self.model.train_on_batch(inputs, targets)
            

        #     s_t = s_t1
        #     t = t + 1

        #     # save progress every 10000 iterations
        #     if t % 1000 == 0:
        #         print("Now we save model")
        #         self.model.save_weights("model.h5", overwrite=True)
        #         with open("model.json", "w") as outfile:
        #             json.dump(self.model.to_json(), outfile)

        #     # print info
        #     state = ""
        #     if t <= OBSERVE:
        #         state = "observe"
        #     elif t > OBSERVE and t <= OBSERVE + EXPLORE:
        #         state = "explore"
        #     else:
        #         state = "train"

        #     print("TIMESTEP", t, "/ STATE", state, \
        #         "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
        #         "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

        # print("Episode finished!")
        # print("************************")

def playGame(args):
    model = buildmodel()
    trainNetwork(model,args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()


