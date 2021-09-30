# The example function below keeps track of the opponent"s history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Flatten

from operator import ne, sub
import re
import numpy as np
from tensorflow.python.ops.gen_array_ops import broadcast_gradient_args_eager_fallback
import operator
import random
# from sklearn import hmm

state = {
        "RP" : 1,
        "RS" : 1,
        "RR" : 1,
        "PR" : 1,
        "PS" : 1,
        "PP" : 1,
        "SR" : 1,
        "SP" : 1,
        "SS" : 1
      }

class MarkovChain:
    global state

    def __init__(self):
      self.beat = {'R': 'P', 'P': 'S', 'S': 'R'}
      self.currentOdd = {}

    def beat_result(self, input):
      return self.beat[input]

    # First entry will be ''
    # Second entry will set the first val
    # Third entry onward will have a pattern
    def update_state(self, input):
      if len(input) > 2:
        lastTwo = input[-2]+input[-1]
        state[lastTwo] += 1

    def find_latest_state(self, lastPlay):
      keyToCheck = []
      for i in state.keys():
        if i[0] == lastPlay:
          keyToCheck.append(i)
      return keyToCheck


    def check_highest_val(self, listOfPotential):
      subsetDict = {key: state[key] for key in listOfPotential}
      return max(subsetDict, key = subsetDict.get)[1]

    def updateOdd(self):

      self.currentOdd = { "R": [
                          state["RP"]/(state['RP']+state['RS']+state['RR']),
                          state["RS"]/(state['RP']+state['RS']+state['RR']),
                          state["RR"]/(state['RP']+state['RS']+state['RR'])
                        ],
                        "P": [
                          state["PR"]/(state['PR']+state['PS']+state['PP']),
                          state["PS"]/(state['PR']+state['PS']+state['PP']),
                          state["PP"]/(state['PR']+state['PS']+state['PP'])
                        ],
                         "S": [
                          state["SR"]/(state['SR']+state['SP']+state['SS']),
                          state["SP"]/(state['SR']+state['SP']+state['SS']),
                          state["SS"]/(state['SR']+state['SP']+state['SS'])
                        ]}

    def predict(self, last_play):
      self.update_state(last_play)
      self.updateOdd()
      predictedPlay = random.choices(self.find_latest_state(last_play[-1]), self.currentOdd[last_play[-1]])[0][-1]
      return self.beat[predictedPlay]

    def getCurrentOdd(self):
      print(self.currentOdd)
    
historicalPlay = []

def player(prev_play, opponent_history=[], winningVal=[]):
    if prev_play == '':
      return 'R'

    historicalPlay.append(prev_play)

    model = MarkovChain()
    output = model.predict(historicalPlay)

    return output