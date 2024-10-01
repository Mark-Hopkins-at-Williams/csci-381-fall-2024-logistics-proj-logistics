import torch
import torchvision
from tqdm import tqdm
import pandas as pd
from matplotlib.pyplot import imshow
from os import listdir
from os.path import join, isfile

try:
    ZEBRA_DIR = 'small/zebra.train'
    ZEBRA_TRAIN_FILES = sorted([join(ZEBRA_DIR, f) 
                                for f in listdir(ZEBRA_DIR) 
                                if isfile(join(ZEBRA_DIR, f))])   
    HORSE_TRAIN_DIR = 'small/horse.train'
    HORSE_TRAIN_FILES = sorted([join(HORSE_TRAIN_DIR, f) 
                                for f in listdir(HORSE_TRAIN_DIR) 
                                if isfile(join(HORSE_TRAIN_DIR, f))]) 
    ZEBRA_TEST_DIR = 'small/zebra.test'
    ZEBRA_TEST_FILES = [join(ZEBRA_TEST_DIR, f) 
                        for f in listdir(ZEBRA_TEST_DIR) 
                        if isfile(join(ZEBRA_TEST_DIR, f))]      
    HORSE_TEST_DIR = 'small/horse.test'
    HORSE_TEST_FILES = [join(HORSE_TEST_DIR, f) 
                        for f in listdir(HORSE_TEST_DIR) 
                        if isfile(join(HORSE_TEST_DIR, f))]  
except FileNotFoundError:
    print("Make sure to download the image data! See README.md for details.")
    ZEBRA_TRAIN_FILES = []
    HORSE_TRAIN_FILES = []
    ZEBRA_TEST_FILES = []
    HORSE_TEST_FILES = []


def load_image(image_file):   
    """Loads an image as an order-2 torch tensor.
    
    Each element of the tensor corresponds to the intensity of
    an image pixel. Each pixel has an intensity between 0 and 255.
    0 == black and 255 == white.

    The returned tensor has dtype torch.int32.

    """    
    color_img = torchvision.io.read_image(image_file)
    result = torchvision.transforms.Grayscale()(color_img)
    return result.squeeze().long()


def show_image(img):
    imshow(img, cmap='gray')
    

def create_dataframe(extractor, zebra_files, horse_files):
    """
    Creates a pandas dataframe that stores the data extracted using
    the provided feature extraction function.
    
    It will first execute extractor(img) for each image file in the zebra_files
    list (you can load an image from a filename using load_grayscale_image).
    This returns a dictionary that feature names to their values. It will then 
    add the response variable (called 'zebra') to the dictionary and set it to 1.
    
    Then, it will repeat this process for each image in the horse_files
    list, except that when it adds the response variable, it sets it to 0.
    
    The list of dictionaries is then converted into a pandas.Dataframe
    using the pd.DataFrame constructor. The DataFrame is returned.
    
    Important note: the columns of the DataFrame should be sorted
    alphabetically (in order for the unit tests to work).
    
    """    
    instances = []
    for filename in tqdm(zebra_files):
        img = load_image(filename)
        instance = extractor(img)
        instance['zebra'] = 1
        instances.append(instance)
    for filename in tqdm(horse_files):
        img = load_image(filename)
        instance = extractor(img)
        instance['zebra'] = 0
        instances.append(instance)
    return pd.DataFrame(instances)


def extract_image_features(img, features):
    result = dict()
    for feature_name in features:
        feat = features[feature_name]
        result[feature_name] = feat(img)
    return result


def compile_dataset(feats, zebra_files, horse_files, csv_file):
    df = create_dataframe(lambda im: extract_image_features(im, feats),
                          zebra_files,
                          horse_files)
    df.to_csv(csv_file)


def feature_matrix(dataframe):
    """
    Gets the feature matrix (as a torch tensor) from a
    pandas dataframe.
    
    """
    columns = list(dataframe.columns)
    if 'Unnamed: 0' in columns:
        columns.remove('Unnamed: 0')
    columns.remove('zebra')
    return torch.from_numpy(dataframe[columns].values).float()
    

def response_vector(dataframe):
    """
    Gets the response vector (as a torch tensor) from a 
    pandas dataframe.
    
    """
    return torch.from_numpy(dataframe['zebra'].values)   


def load_data(csv_file):
    """Loads the feature matrix and response vector from a CSV file."""
    dataframe = pd.read_csv(csv_file) 
    X = feature_matrix(dataframe)
    y = response_vector(dataframe)
    return X, y