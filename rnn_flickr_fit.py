import os
import sys
import json
import numpy as np
from numpy import array
from pickle import load, dump
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# load a pre-defined list of photo identifiers
def load_list(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return dataset


def concat_features_to_bboxes(features, bboxes):
    results = []
    for ind in range(len(bboxes)):
        a = bboxes[ind]
        # the fifth colum has the score
        indexes = np.argsort(-a[:, 5])
        sorted_a = a[indexes, :]
        # getting only 20 bboxes instead of default 100, to avoid so many null arrays
        final_bboxes = sorted_a[:20]
        results.append(np.append(features[ind], final_bboxes.flatten()))
    return results


# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            descriptions[image_id].append(desc)
    return descriptions


# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    # The parameter 'photo' contains the characteristics vector and photo's detections
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)


# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length, imgsIds, vocab_size):
    # loop for ever over images
    while 1:
        for ind in range(len(imgsIds)):
            # retrieve the photo feature
            photo = photos[ind]
            key = imgsIds[ind]
            desc_list = descriptions[str(key)]
            in_img, in_seq, out_word = create_sequences(
                tokenizer, max_length, desc_list, photo, vocab_size)
            yield [[in_img, in_seq], out_word]


# define the captioning model
def define_model(vocab_size, max_length, curr_shape):
    # curr_shape is a tuple
    inputs1 = Input(shape=curr_shape)
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    # model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)
    return model


def main(args):
    images_ids = args[1] if len(args) > 1 else '../ImageDescriptionModel/Flickr8k_text/Flickr_8k.trainImages.txt'
    # load names of images as id
    train = load_list(images_ids)  
    print('Dataset: %d' % len(train))
    # descriptions
    descriptions_file = args[2] if len(args) > 2 else 'descriptions.txt'
    train_descriptions = load_clean_descriptions(descriptions_file, train)
    print('train_descriptions: %d' % len(train_descriptions))
    # photo features
    features = args[3] if len(args) > 3 else 'features_flickr_train.pkl'
    train_features = load(open(features, 'rb'))
    print('Features: %d' % len(train_features))
    print(type(train_features))
    print(type(train_features[0]))
    print(train_features[0].shape)  # (4032,) nasnet and (4051,) for resnet50
    # photo bounding boxes (generated by a detection model)
    bboxes = args[4] if len(args) > 4 else input('Type the path for the detection features (train) or press 1 to continue: ') # flickr_detection_train.pkl
    if bboxes == '1':
        # do not concat features and bboxes
        train_rnn_input = train_features
    else:
        train_bboxes = load(open(bboxes, 'rb'))
        print('BBoxes: %d' % len(train_bboxes))
        print(type(train_bboxes))
        print(type(train_bboxes[0]))
        # (100,6) needs to be converted to 1D array with shape (20,6) - 20 bboxes with 6 data: x1,y1,x2,y2,class_id, bbox_score
        print(train_bboxes[0].shape)
        # concat features and bboxes into 1D array
        train_rnn_input = concat_features_to_bboxes(train_features, train_bboxes)
     
    print('Input RNN: %d' % len(train_rnn_input))
    print(type(train_rnn_input))
    print(type(train_rnn_input[0]))
    print(train_rnn_input[0].shape)  # (4152,)

    # prepare tokenizer
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    # determine the maximum sequence length
    max_len = max_length(train_descriptions)
    print('Description Length: %d' % max_len)

    # dev dataset
    # load val/dev set
    val_ids = args[5] if len(args) > 5 else '../ImageDescriptionModel/Flickr8k_text/Flickr_8k.devImages.txt'
    val = load_list(val_ids)
    print('Dataset: %d' % len(val))
    # descriptions
    val_descriptions = load_clean_descriptions(descriptions_file, val)
    print('val_descriptions: val=%d' % len(val_descriptions))
    # photo features
    features_file = args[6] if len(args) > 6 else 'features_flickr_dev.pkl'
    val_features = load(open(features_file, 'rb'))
    print('Photos: val=%d' % len(val_features))
    print(type(val_features))
    print(type(val_features[0]))
    print(val_features[0].shape)  # (4032,)
    # photo bboxes (generated by Mask R-CNN) - x1,y1,x2,y2,class_id, bbox_score
    if bboxes == '1':
        # do not concat features and bboxes
        val_rnn_input = val_features
    else:
        val_bboxes_file = args[7] if len(args) > 7 else input('Type the path for the detection features (dev): ') # flickr_detection_dev.pkl
        val_bboxes = load(open(val_bboxes_file, 'rb'))
        print('BBoxes: %d' % len(val_bboxes))
        print(type(val_bboxes))
        print(type(val_bboxes[0]))
        # (100,6) needs to be converted to 1D array with shape (20,6) - 20 bboxes with 6 data: x1,y1,x2,y2,class_id, bbox_score
        print(val_bboxes[0].shape)
        # concat features and bboxes into 1D array
        val_rnn_input =  concat_features_to_bboxes(val_features, val_bboxes)

    print('Input RNN: %d' % len(val_rnn_input))
    print(type(val_rnn_input))
    print(type(val_rnn_input[0]))
    print(val_rnn_input[0].shape)  # (4152,)

    # fit model
    # define the model
    model = define_model(vocab_size, max_len, train_rnn_input[0].shape)
    # define checkpoint callback
    filepath = 'checkpoint/model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    epochs = 10
    train_steps = len(train_descriptions)
    val_steps = len(val_descriptions)

    # create the data generator
    train_generator = data_generator(train_descriptions, train_rnn_input, tokenizer, max_len, train, vocab_size)
    val_generator = data_generator(val_descriptions, val_rnn_input, tokenizer, max_len, val, vocab_size)
    # fit for train dataset (6K) and validation dataset (1K)
    history = model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=train_steps, verbose=1, callbacks=[checkpoint], validation_data=val_generator, validation_steps=val_steps)

    with open('checkpoint/history_epochs.pkl', 'wb') as f:
        dump(history.history, f)

if __name__ == '__main__':
    main(sys.argv)