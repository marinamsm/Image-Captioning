''' We can call this function to prepare the photo data for testing our models, then save the resulting dictionary to a file named ‘features.pkl‘. '''
import os
import sys
import numpy as np
from pickle import dump
from keras.applications.nasnet import NASNetLarge
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.nasnet import preprocess_input
from keras.utils import plot_model
from keras.models import Model

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

# extract features from each photo in the directory
def extract_features(directory, ids, model):
    if int(model) == 1:
        print("1")
        # load ResNet50 model
        model = ResNet50()
        input_size = 224
    else:
        print("2")
        # load NASNetLarge model
        model = NASNetLarge(input_shape=(331, 331, 3), include_top=True, weights='imagenet', input_tensor=None, pooling=None)
        input_size = 331
    # pops the last layer to get the features
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # model characteristics
    plot_model(model, to_file='model.png')
    imgs = load_list(ids)
    print('Dataset: %d' % len(imgs))
    N = len(imgs)
    print(N)
    results = []
    i = 0
    batch_size = 1 #this can be 8 for a GTX 1080 Ti and 32G of RAM
    while i < N:
        if i % 1024 == 0:
            print('{} from {} images.'.format(i, N))
        batch = imgs[i:i + batch_size]
        i += batch_size
        images = [
            load_img(
                os.path.join(directory, img + ".jpg"),
                target_size=(input_size, input_size)
            )
            for img in batch
        ]
        images = [preprocess_input(img_to_array(img)) for img in images]
        images = np.stack(images)
        r = model.predict(images)
        for ind in range(batch_size):
            results.append(r[ind])
    return results


def main(args): 
    # extract features from all images in database
    imgs_directory = args[1] if len(args) > 1 else input('Type the images path: ')  #'../ImageDescriptionModel/Flickr8k_Dataset/Flicker8k_Dataset'
    imgs_ids = args[2] if len(args) > 2 else input('Type the document path (.txt) with image ids: ')  #'../ImageDescriptionModel/Flickr8k_text/test'
    model = args[3] if len(args) > 3 else input('Type 1 to use Resnet50 or 2 for NasnetLarge: ')
    features = extract_features(imgs_directory, imgs_ids, model)
    print('ok')
    print('Extracted Features: %d' % len(features))
    # save to file
    dump(features, open('features.pkl', 'wb'))
    
    
if __name__ == '__main__':
    main(sys.argv)