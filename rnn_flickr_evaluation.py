import os
import sys
import json
import numpy as np
from numpy import argmax
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.models import load_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import sys


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


# covert a dictionary of clean descriptions to a list of descriptions
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
def create_sequences(tokenizer, max_length, desc_list, photo):
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


# define the captioning model
def define_model(vocab_size, max_length):
    # 4632 features + detections
    inputs1 = Input(shape=(4152,))
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


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length, model_type=""):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        ph = photo.reshape((1, photo.shape[1])) if model_type == "resnet50" else photo.reshape((1, photo.shape[0]))
        # predict next word
        yhat = model.predict([ph, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            print("NO WORD IN VOCAB FOR THIS DESCRIPTION")
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, max_length, imgsIds_val, printCaption=False):
    actual, predicted = list(), list()
    for ind in range(len(imgsIds_val)):
            # generate description
        yhat = generate_desc(model, tokenizer, photos[ind], max_length)
        key = imgsIds_val[ind]
        desc_list = descriptions[str(key)]
        # store actual and predicted
        references = [d.split() for d in desc_list]
        if printCaption:
            print("predicted: ", yhat.split())
            print("references: ", references)
        actual.append(references)
        predicted.append(yhat.split())
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


FILENAME = '../ImageDescriptionModel/Flickr8k_text/Flickr_8k.testImages.txt'
FILENAME_TRAIN = '../ImageDescriptionModel/Flickr8k_text/Flickr_8k.trainImages.txt'

def full_eval():    
    print("Running full evaluation. This might take a few minutes...")
    # load training dataset (+/- 41K)
    imgsIds = load_list(FILENAME_TRAIN)
    print('Dataset: %d' % len(imgsIds))
    imgsIds_test = load_list(FILENAME)
    print('Dataset: %d' % len(imgsIds_test))
    # descriptions
    descriptions_train = load_clean_descriptions('descriptions.txt', imgsIds)
    print('Descriptions: %d' % len(descriptions_train))
    descriptions_test = load_clean_descriptions('descriptions.txt', imgsIds_test)
    print('Descriptions: %d' % len(descriptions_test))
    # photo features (extracted by NasNet)
    features_test = load(
        open('features_flickr_test.pkl', 'rb'))
    print('Features: %d' % len(features_test))
    print(features_test[0].shape)
    # photo bboxes (generated by Mask R-CNN)
    bboxes_test = load(open('flickr_detection_test.pkl', 'rb'))
    print('BBoxes: %d' % len(bboxes_test))
    print(bboxes_test[0].shape)
    # concat features and bboxes into 1D array
    rnn_input_test = concat_features_to_bboxes(features_test, bboxes_test)
    # do not concat features and bboxes
    # rnn_input_test = features_test
    print('Input RNN: %d' % len(rnn_input_test))
    print(rnn_input_test[0].shape)
    # prepare tokenizer
    tokenizer = create_tokenizer(descriptions_train)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    # determine the maximum sequence length
    max_len = max_length(descriptions_train)
    print('Description Length: %d' % max_len)
    # load the model to evaluate it
    filename = 'model-ep004-loss3.367-val_loss3.777.h5'
    model = load_model(filename)
    # evaluate model
    evaluate_model(model, descriptions_test, rnn_input_test,
                   tokenizer, max_len, imgsIds_test)

def single_eval(args):
    print("Running single evaluation. This should be fast!")
    # load training dataset (+/- 41K)
    imgsIds = load_list(FILENAME_TRAIN)
    imgsIds_test = [args[1]]
    # descriptions
    descriptions_train = load_clean_descriptions('descriptions.txt', imgsIds)
    descriptions_test = load_clean_descriptions('descriptions.txt', imgsIds_test)
    # photo features (extracted by NasNet)
    features_test = load(
        open(args[2], 'rb'))
    if len(args) > 3 and args[3] == "bbox":
        print("BBOX ON")
        # photo bboxes (generated by Mask R-CNN)
        bboxes_test = load(open('flickr_detection_test.pkl', 'rb'))
        print('BBoxes: %d' % len(bboxes_test))
        print(bboxes_test[0].shape)
        # concat features and bboxes into 1D array
        rnn_input_test = concat_features_to_bboxes(features_test, bboxes_test)
        # do not concat features and bboxes
        # rnn_input_test = features_test
    else:
        print("NO BBOX")
        rnn_input_test = features_test
    # prepare tokenizer
    tokenizer = create_tokenizer(descriptions_train)
    vocab_size = len(tokenizer.word_index) + 1
    # determine the maximum sequence length
    max_len = max_length(descriptions_train)
    # load the model to evaluate it
    filename = args[4]
    model = load_model(filename)
    # evaluate model
    evaluate_model(model, descriptions_test, rnn_input_test,
                   tokenizer, max_len, imgsIds_test, True)

            
def main(args):
    return single_eval(args) if len(args) > 1 else full_eval()
               
if __name__ == '__main__':
    main(sys.argv)