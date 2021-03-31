/**code with comments to show what each line is doing*/



/** the def load_data() function loads machine learning data in python */
def load_data():

    /**each of these functions basically just creates and defines the paths where all of the information that will be read,
    validated, and tested will go with the use of the functions below */

    train_path = os.path.join(data_path, "ptb.train.txt") #
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    /** the functions listed below takes inputed text and converts it into integers
    Each letter is not specified a unique integer identifier though, but each sentence
    is split up by the call of each function and each word will then be given a unique number identifier.
    splitting it into seperte words and then giving them number identifiers allows all of the text
    to be put into  the neural network
    */

    word_to_id = build_vocab(train_path) #each unique word in the corpus is assaigned unique integer index
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    print(train_data[:5])
    print(word_to_id)
    print(vocabulary)
    print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary


/**this calls on the paths created above and splits the information, after each word has been given
a unique number identifier, into test data, train data and validate data. I imagine
tht it does so similar to other nearual networks and keeps doing this until we get higher accuracy.
each data listed is given in the same order.*/

    train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()

/**this is the Python iterator function. Handles data extraction, executing the model, logging
metrics, and executing calbacks*/

    while True:
    #do some things to create a batch of data (x, y)
   yield x, y


/**code above allows the data to be extracted from the object created below.
The reason for this object is that it allows the data to be taken as the first argument.  */

   class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, vocabulary, skip_step=5):
        self.data = data

        /**defines the number of words that will be fed into the time
        distributed layer of the neural network. It is the scond argument after KerasBatchGenerator.
        Basically, this is the set of words, that have been given numbers, tht the model will learn (train) on
        to figure out the word coming after.*/
        self.num_steps = num_steps

        /** this determines the number of training examples in one forward/backwards pass*/
        self.batch_size = batch_size

        /**equal to 10,000
        self.vocabulary = vocabulary
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        /*it is the number of words to be skipped before the next batch of data is taken */
        self.skip_step = skip_step

/** this generates the prediction of one word based on 10,000 possible categories
it also has a function that check if a reset is needed for the current index pointer. finally, the keres_to_categorial
function is used to convert each of the target words in each sample into the one-hot or categorical representation  */

        def generate(self):
    x = np.zeros((self.batch_size, self.num_steps))
    y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))

    /**creates a for loop of size_batch to populate all the data into the branch */
    while True:
        for i in range(self.batch_size):
            if self.current_idx + self.num_steps >= len(self.data):
                # reset the index back to the start of the data set
                self.current_idx = 0
            x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps]
            temp_y = self.data[self.current_idx + 1:self.current_idx + self.num_steps + 1]
            # convert all of temp_y into a one hot representation
            y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary)
            self.current_idx += self.skip_step
        yield x, y

        /** the generator class has been created above and now the following code will create
        instances of this class */

        train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)


/**this is the begnning of actually creating the keras LTSM network.
allows you to easily stack layers into your network without worrying
too much about all the tensors (and their shapes) flowing through the model*/

                                           model = Sequential() /**first step converts our words into meaningful vectors*/

/**takes the size of the vocabulary as its first argument,
then the size of the resultant embedding vector that you want as the next argument*/
model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))

model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
if use_dropout:
    model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(vocabulary)))
model.add(Activation('softmax'))

/**to specify a layer, you have to specify the number of nodes in the hidden layer*/

/**after a model has been created, you use the following function call to run it */
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

/** this function uses the adam optimizer because because it is adaptive to many different conditions*/


/**this is the callback function. It saves the model after each epoch,
which can be handy for when you are running long-term training*/

checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)

/**this is the final step in creating th KEras LTSM model. the functions in the first argument
make sure that whole data set is run through the model in each epoch.

model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs, /**first argument*/
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])

                        /**an LTSM model such as this will have to be run over many epochs to get
                        the accuracy desired*/


                        /** the code below tests the trained Keras LSTM model,
                        and compares the predicted word outputs against what the actual word sequences
                        are in the training and test data set. first step is it reloads the model from the trained data.
                        Then, it creates a KerasBAtchGenerator to specify the batch size. Then, it creates a loop of dummy
                        data extractions. then it has a num_predict function to predict what will happen. */


                        model = load_model(data_path + "\model-40.hdf5")
dummy_iters = 40
example_training_generator = KerasBatchGenerator(train_data, num_steps, 1, vocabulary,
                                                     skip_step=1)
print("Training data:")
for i in range(dummy_iters):
    dummy = next(example_training_generator.generate())
num_predict = 10
true_print_out = "Actual words: "
pred_print_out = "Predicted words: "
for i in range(num_predict):
    data = next(example_training_generator.generate())
    prediction = model.predict(data[0])
    predict_word = np.argmax(prediction[:, num_steps-1, :])

/**translates into an actual English word*/
    true_print_out += reversed_dictionary[train_data[num_steps + dummy_iters + i]] + " "
    pred_print_out += reversed_dictionary[predict_word] + " "


print(true_print_out)
print(pred_print_out)