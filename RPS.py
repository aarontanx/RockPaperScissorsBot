# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.


def player(prev_play, opponent_history=[], winningVal=[]):
    opponent_history.append(prev_play)

    if prev_play == 'R':
      winningVal.append('P') 
    elif prev_play == 'P':
      winningVal.append('S') 
    else: 
      winningVal.append('S') 

    train_X = np.asarray(opponent_history)
    train_y = np.asarray(winningVal)

    train = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    genericModel(train, prev_play)
    # guess = "R"
    # if len(opponent_history) > 2:
    #     guess = opponent_history[-2]

    return guess


def genericModel(train, currentPlay):
    model = Sequential()
    # model.add(base)
    # model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(3,activation='softmax'))
    model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['acc'])

    model.fit(train)
    result = model.predict(currentPlay)
    return result[-1]