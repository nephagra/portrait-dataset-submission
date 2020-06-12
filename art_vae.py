######################## IMPORTING / GOOGLE DRIVE INTEGRATION ##############################
import torch
import argparse
import copy
import cv2
import json
import numpy as np
import os
import pandas as pd
import pdb
import pickle
import random
import scipy.ndimage
import shutil
import sklearn.preprocessing
import sys
import tensorflow as tf
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image 
from itertools import product
from matplotlib import pyplot as plt
import numpy as np
import pickle
import json
import pandas as pd
import tensorflow as tf
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import argparse
import sys
import pdb
import sklearn
import yaml

from shutil import copy2
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from yaml import load, dump

pdb.set_trace()

#pdb.set_trace()

################################ SETTING UP PROGRAM ARGUMENTS ##############################

IS_RUNNING_NOTEBOOK = False# ** NOTE: PLEASE DO NOT FORGET TO CHANGE THIS FLAG WHEN YOU DEPLOY **
IS_TRAINING_BASE_VAE = False # Which Model to Train / Load

if IS_RUNNING_NOTEBOOK:
  from google.colab import drive
  drive.mount('/content/drive')  
  sys.path.append('/content/drive/Shared drives/GRASP Art Project/Art VAE')

# Have to do this here (although it looks extremely ugly) just because the logger file is only visible in the notebook after the drive is mounted
from logger import Logger

start = time.time() # Starts the timer so we can cleanly exit from the program after the specified timeout (3600s)
device = torch.device("cuda")

if not IS_RUNNING_NOTEBOOK:
  mount_filepath = '/NAS/home/style-disentanglement/GRASP_Art_Project/Art_VAE/'
  parser = argparse.ArgumentParser()
  args = parser.parse_args()

  parser.add_argument("yaml_config", help = "No config file specified. Please specify a yaml file.")
  config_file = file(args.yaml_config, 'r')
  config_args = yaml.load(config_file)

  LATENT_SIZE = config_args['LATENT_SIZE']
  LATENT_ARTIST_SIZE = config_args['LATENT_ARTIST_SIZE']
  LATENT_TIME_PERIOD_SIZE = config_args['LATENT_TIME_PERIOD_SIZE']
  
  TIMEOUT = 3600
else:
  mount_filepath = '/content/drive/Shared drives/GRASP Art Project/'
  LATENT_SIZE = 1024
  LATENT_ARTIST_SIZE = 256
  LATENT_TIME_PERIOD_SIZE = 256
  TIMEOUT = np.inf


MASK_TYPE = "mask"
OUTLIERS_REMOVED = "outliers"


interval = 1 # Interval for which to print the loss
epochs = 500 # Maximum number of epochs to run
epoch = 0 # Start epoch to run from (gets automatically set)


################################ HYPERPARAMETERS ##############################

# Base VAE Hyperparameters
TRIPLET_ALPHA = 1
LATENT_ARTIST_WEIGHT = 1.0
TIME_PERIOD_WEIGHT = (1.0 / 10000.0)
RECONSTRUCTION_WEIGHT = 1
VAE_DIVERGENCE_WEIGHT = (1.0 / 10000.0)
TRIPLET_LOSS_WEIGHT = 1

# Latent Artist Prediction Hyperparameters
ARTIST_PREDICTION_WEIGHT = 1
ARTIST_KLD_WEIGHT = (1.0 / 1000.0)


# Loss Thresholds
MAX_VAE_LOSS_THRESHOLD = 200
MAX_LATENT_ARTIST_LOSS_THRESHOLD = 10

# Random sampling hyperparameters
SAME_ARTIST_PROBABILITY_THRESHOLD = 0.7

interval = 1 # Interval for which to print the loss
epochs = 500 # Maximum number of epochs to run
epoch = 0 # Start epoch to run from (gets automatically set in the checkpointing process)

################################ SETTING UP DIRECTORY STRUCTURE ##############################
model_name_prefix = "full_model"
model_name = f"{model_name_prefix}_artist_{LATENT_ARTIST_SIZE}_timeperiod_{LATENT_TIME_PERIOD_SIZE}_latent_{LATENT_SIZE}" # Name of the model generated from the program arguments

writer = SummaryWriter(f'./tensorboard/{model_name}') # Tensorboard writer
prefix = "padded_" # Prefix for all the files being saved
continue_training = True # Specifies whether you want to continuing training from the previous epoch. Honestly don't know why you'd set it to false but you can if you want.
models_folder = f"{mount_filepath}/saved_models/{model_name}/models"

model_save_root = mount_filepath + '/saved_models/'
results_folder = f"{model_save_root}/{model_name}/images"
plot_folder = f"{model_save_root}/{model_name}/plots"
log_folder = f"{model_save_root}/{model_name}/logs"

load_model_folder = f"{model_save_root}{model_name}/models"

os.makedirs(f"{model_save_root}/{model_name}", exist_ok=True)
os.makedirs(results_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

LATENT_PREFIX = "padded_"

DISCRIMINATOR_PREFIX = "discriminator" # DONT CHANGE THIS
LATENT_VAE_PREFIX = "vae" # DONT CHANGE THIS

LATENT_ARTIST_MODEL_NAME = "discriminator_latent_model"
# If you're training the base VAE, you must still specify a load path for the latent artist network
LATENT_ARTIST_MODEL_LOAD_PATH = f"{model_save_root}{LATENT_ARTIST_MODEL_NAME}/models"

if not IS_TRAINING_BASE_VAE:
  model_name = LATENT_ARTIST_MODEL_NAME
  prefix = LATENT_PREFIX
  load_model_folder = LATENT_ARTIST_MODEL_LOAD_PATH
  


################################ SETTING UP PATHS / DIRECTORY STRUCTURE ##############################

metadata_filepath = mount_filepath + 'FINAL_DATASET_SIZES.csv'
image_zip_folder = mount_filepath + 'Images.zip'
image_folder = mount_filepath + "Images"
triplets_path = mount_filepath + "Triplets/4700_labels_copy.json" 

writer = SummaryWriter(f'./tensorboard/{model_name}') # Tensorboard writer
prefix = "padded_" # Prefix for all the files being savedc
models_folder = f"{mount_filepath}/saved_models/{model_name}/models"

model_save_root = mount_filepath + '/saved_models/'
results_folder = f"{model_save_root}/{model_name}/images"
log_folder = f"{model_save_root}/{model_name}/logs"

os.makedirs(f"{model_save_root}/{model_name}", exist_ok=True)
os.makedirs(results_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)
os.makedirs(f"{model_save_root}/{model_name}/models", exist_ok=True)

################################ ART DATASET CLASS ##############################

class ArtDataset(Dataset):
    def __init__(self, image_folder, triplet_df, metadata_df, largest_height, largest_width, transform=None):
        """
        Args:
            image_folder (string): Directory with all the images.
            metadata_df (Pandas Dataframe): Dataframe containing the metadata
            transform (Torchvision.Transform) Transform to be applied to the image data
        """
        self.transform = transform
        self.metadata = metadata_df.reset_index()
        self.triplets = triplet_df
        self.image_folder = image_folder
        self.largest_height = largest_height
        self.largest_width = largest_width

    def __len__(self):
        return len(self.triplets.index)

    def return_transformed_image(self, image_filename):
        image = Image.open(os.path.join(self.image_folder, image_filename))
        (width, height) = image.size
        tensor_image = self.transform(image)
        pad_layer = nn.ZeroPad2d((0, self.largest_width - width, 0, self.largest_height - height))
        return pad_layer(tensor_image)

    def get_random_artist_image(self, artist):
        artist_indices = self.metadata[self.metadata['cleaned_artist'] == artist].index
        random_artist_idx = random.choice(artist_indices)
        return self.metadata.iloc[random_artist_idx]

    def __getitem__(self, idx):
      row = self.triplets.iloc[idx]
      
      positive_idx = int(row['Positive']) - 1
      negative_idx = (1 - positive_idx)

      triplet_names = ['1', '2', 'anchor']

      # Re-Arranges it so now it's of the form positive, negative, anchor.
      triplet_files = [row[triplet_names[positive_idx]], row[triplet_names[negative_idx]], row[triplet_names[2]]]
      labels = ['positive', 'negative', 'anchor']

      triplet_objects = {}

      for filename, label in zip(triplet_files, labels):
        triplet_object = {}

        triplet_image = {}
        triplet_corresponding = {}
        
        image_metadata = self.metadata.loc[self.metadata['filename'] == filename]
        image_name = image_metadata['filename'].values[0]
        
        triplet_image['artist'] = image_metadata['cleaned_artist'].values[0]
        triplet_image['image'] = self.return_transformed_image(image_name)

        triplet_image['normalized_midpoint'] = image_metadata['normalized_midpoint'].values[0]
        triplet_image['normalized_start'] = image_metadata['normalized_start'].values[0]
        triplet_image['normalized_end'] = image_metadata['normalized_end'].values[0]

        triplet_image['width'] = image_metadata['width'].values[0]
        triplet_image['height'] = image_metadata['height'].values[0]

        random_artist = self.get_random_artist_image(triplet_image['artist'])

        triplet_corresponding['image'] = self.return_transformed_image(random_artist['filename'])
        triplet_corresponding['filename'] = random_artist['filename']

        triplet_corresponding['normalized_midpoint'] = random_artist['normalized_midpoint']
        triplet_corresponding['normalized_start'] = random_artist['normalized_start']
        triplet_corresponding['normalized_end'] = random_artist['normalized_end']

        triplet_corresponding['width'] = random_artist['width']
        triplet_corresponding['height'] = random_artist['height']

        triplet_corresponding['should_mask'] = (random_artist['filename'] == image_metadata['filename'].values[0])

        triplet_object['triplet'] = triplet_image
        triplet_object['corresponding'] = triplet_corresponding

        triplet_objects[label] = triplet_object

      return triplet_objects


class ArtistSampler(Dataset):
    def __init__(self, image_folder, triplet_df, metadata_df, largest_height, largest_width, transform=None):
        """
        Args:
            image_folder (string): Directory with all the images.
            metadata_df (Pandas Dataframe): Dataframe containing the metadata
            transform (Torchvision.Transform) Transform to be applied to the image data
        """
        self.transform = transform
        self.metadata = metadata_df.reset_index()
        self.triplets = triplet_df
        self.image_folder = image_folder
        self.largest_height = largest_height
        self.largest_width = largest_width

    def __len__(self):
        return len(self.triplets.index)

    def return_transformed_image(self, image_filename):
        image = Image.open(os.path.join(self.image_folder, image_filename))
        (width, height) = image.size
        tensor_image = self.transform(image)
        pad_layer = nn.ZeroPad2d((0, self.largest_width - width, 0, self.largest_height - height))
        return pad_layer(tensor_image)

    def get_random_artist_image(self, artist, original_idx):
        artist_indices = (self.metadata[self.metadata['cleaned_artist'] == artist].index).tolist()
        artist_indices.remove(original_idx)
        random_artist_idx = random.choice(artist_indices)
        return self.metadata.iloc[random_artist_idx]
    
    def get_random_diff_artist_image(self, artist):
        artist_indices = self.metadata[self.metadata['cleaned_artist'] != artist].index
        random_artist_idx = random.choice(artist_indices)
        return self.metadata.iloc[random_artist_idx]

    def __getitem__(self, idx):
      row = self.metadata.iloc[idx]
      artist = row['cleaned_artist']

      # < 0.7 is the same artist, >= 0.7 is diff artist
      diff_artist = random.uniform(0, 1)

      if len(self.metadata[self.metadata['cleaned_artist'] == artist].index) == 1:
        diff_artist = 1
      else:
        diff_artist = (diff_artist >= SAME_ARTIST_PROBABILITY_THRESHOLD)
      
      return_image = None
      if diff_artist == 1:
        return_image = self.get_random_diff_artist_image(artist)
      else:
        return_image = self.get_random_artist_image(artist, idx)

      return_object = {}
      return_object["image"] = self.return_transformed_image(row['filename'])
      return_object["other_image"] = self.return_transformed_image(return_image['filename'])
      return_object["is_diff"] = diff_artist

      return return_object

################################ READING IN DATA FROM CSV / PREPROCESSING ##############################

import json
import pandas as pd

metadata_filepath = mount_filepath + 'FINAL_DATASET_SIZES.csv'
image_zip_folder = mount_filepath + 'Images.zip'

image_folder = mount_filepath + "Images"

metadata_df = pd.read_csv(metadata_filepath)
metadata_df = metadata_df.fillna("")
artists_column = metadata_df['cleaned_artist'].str.lower()
known_artists = metadata_df[(artists_column != 'unsure') & (artists_column != 'anonymous')]

anonymous_artists = metadata_df[(artists_column == 'unsure') ^ (artists_column == 'anonymous')]

# To use all the data just set min_thresh = 0 and max_thresh = 1
min_thresh = 0.00
max_thresh = 1.00

# Only keep the artists that are known
metadata_df = known_artists

sorted_widths = sorted(metadata_df['width'].tolist())
sorted_heights = sorted(metadata_df['height'].tolist())

if OUTLIERS_REMOVED == "outliers":
  bottom_range_height = sorted_heights[max(0, int(len(sorted_heights) * min_thresh) - 1)]
  top_range_height = sorted_heights[max(0, int(len(sorted_heights) * max_thresh) - 1)]
  bottom_range_width = sorted_widths[max(0, int(len(sorted_widths) * min_thresh) - 1)]
  top_range_width = sorted_widths[max(0, int(len(sorted_widths) * max_thresh) - 1)]
else:
  bottom_range_width = 100
  top_range_width = 200
  bottom_range_height = 150
  top_range_height = 200

middle_df = metadata_df.loc[(metadata_df['width'] >= bottom_range_width) & (metadata_df['width'] <= top_range_width) & (metadata_df['height'] >= bottom_range_height) & (metadata_df['height'] <= top_range_height)]

string_json = json.load(open(triplets_path, 'r'))
triplets_df = pd.DataFrame(string_json)

non_empty_triplets = triplets_df[triplets_df['Positive'] != '']
valid_filenames = middle_df['filename'].tolist()

encoder = sklearn.preprocessing.OneHotEncoder()
one_hot_encoding = encoder.fit_transform(middle_df['cleaned_artist'].values.reshape(-1,1)).toarray()
num_artists = one_hot_encoding.shape[1]

# No longer need the one-hot encoding since we are no longer performing artist classification
# middle_df['artist_encoding'] = [np.array(a) for a in one_hot_encoding]

min_date = np.min(middle_df['mid-date'].values)
max_date = np.max(middle_df['mid-date'].values)


normalized_middate = (middle_df['mid-date'].values - min_date) / (max_date - min_date)
normalized_start_date = (middle_df['start-date'].replace('', 0).values - min_date) / (max_date - min_date)
normalized_end_date = (middle_df['end-date'].replace('', 0).values - min_date) / (max_date - min_date)

std_devs = np.zeros(len(normalized_start_date))
std_devs[normalized_start_date >= 0.0] = (normalized_middate[normalized_start_date >= 0.0] - normalized_start_date[normalized_start_date >= 0.0]) / 2.0

SAME_DATE_SIGMA = 1.0
std_devs[normalized_start_date < 0.0] = (SAME_DATE_SIGMA / (max_date - min_date))

min_std_dev = np.min(std_devs)
max_std_dev = np.max(std_devs)

middle_df['normalized_midpoint'] = normalized_middate
middle_df['normalized_start'] = normalized_start_date
middle_df['normalized_end'] = normalized_end_date

triplets_within_size = non_empty_triplets[(non_empty_triplets['1'].isin(valid_filenames)) & (non_empty_triplets['2'].isin(valid_filenames)) & (non_empty_triplets['anchor'].isin(valid_filenames))]

print(f"Number of Triplets: {len(triplets_within_size)}")
################################ CREATING TRAIN / TEST SPLIT ##############################
from sklearn.model_selection import train_test_split

data_transform = transforms.Compose([
        transforms.ToTensor()
    ])

split = 0.1

train, test = train_test_split(triplets_within_size, test_size=split)


## This is for training the Base VAE
train_art_dataset = ArtDataset(image_folder, train, middle_df, top_range_height, top_range_width, transform=data_transform)
test_art_dataset = ArtDataset(image_folder, test, middle_df, top_range_height, top_range_width, transform=data_transform)

batch_sz = 50
num_wrkrs = 10

train_dataset_loader = torch.utils.data.DataLoader(train_art_dataset,
                                             batch_size=batch_sz, shuffle=True,
                                             num_workers=num_wrkrs)

test_dataset_loader = torch.utils.data.DataLoader(test_art_dataset,
                                             batch_size=batch_sz, shuffle=True,
                                             num_workers=num_wrkrs)

################################ DEFINING NEURAL NETWORK ARCHITECTURE ##############################

# Discriminator that discriminates between the two latent vectors fed in as input

class VAE_latent_artist_discriminator(nn.Module):
  def __init__(self, encoding_length):
        super(VAE_latent_artist_discriminator, self).__init__()

        self.encoding_length = encoding_length
        self.initialize_latent_artist_discriminator()
        self.sigmoid = nn.Sigmoid()  

  ######################################################################
  ######################################################################
  ####################### Network Initialization #######################
  ######################################################################
  ######################################################################

  def initialize_latent_artist_discriminator(self):
      self.latent_artist_discriminate_1 = nn.Linear(2 * self.encoding_length, self.encoding_length * 4)
      self.latent_artist_bn_1 = nn.BatchNorm1d(self.encoding_length * 4)
      self.latent_artist_discriminate_2 = nn.Linear(self.encoding_length * 4, self.encoding_length * 2)
      self.latent_artist_bn_2 = nn.BatchNorm1d(self.encoding_length * 2)
      self.latent_artist_discriminate_3 = nn.Linear(int(self.encoding_length * 2), int(self.encoding_length))
      self.latent_artist_bn_3 = nn.BatchNorm1d(int(self.encoding_length))
      self.latent_artist_discriminate_4 = nn.Linear(int(self.encoding_length), int(self.encoding_length / 2))
      self.latent_artist_bn_4 = nn.BatchNorm1d(int(self.encoding_length / 2))
      self.latent_artist_discriminate_5 = nn.Linear(int(self.encoding_length / 2), 1)

  ######################################################################
  ######################################################################
  ########################### Forward Pass #############################
  ######################################################################
  ######################################################################

  def latent_artist_discriminator_forward(self, x):
      x = F.relu(self.latent_artist_bn_1(self.latent_artist_discriminate_1(x)))
      x = F.relu(self.latent_artist_bn_2(self.latent_artist_discriminate_2(x)))
      x = F.relu(self.latent_artist_bn_3(self.latent_artist_discriminate_3(x)))
      x = F.relu(self.latent_artist_bn_4(self.latent_artist_discriminate_4(x)))
      x = self.latent_artist_discriminate_5(x)
      return self.sigmoid(x)

  def forward(self, x):
      return self.latent_artist_discriminator_forward(x)



# This is the pretrained latent artist encoding VAE
class VAE_latent_artist_encoding(nn.Module):
  
  def __init__(self, artist_latent_size, num_artists):
        super(VAE_latent_artist_encoding, self).__init__()

        self.artist_latent_size = artist_latent_size
        self.num_artists = num_artists

        self.initialize_latent_artist_encoder()
        self.initialize_latent_artist_reparameterization()


  ######################################################################
  ######################################################################
  ####################### Network Initialization #######################
  ######################################################################
  ######################################################################

  def initialize_latent_artist_encoder(self):
      self.latent_artist_encode_conv1 = nn.Conv2d(3, 8, 3, stride=2, padding = 1)
      self.latent_artist_conv1_bn = nn.BatchNorm2d(8)
      self.latent_artist_encode_conv2 = nn.Conv2d(8, 16, 3, stride=2, padding = 1)
      self.latent_artist_conv2_bn = nn.BatchNorm2d(16)
      self.latent_artist_encode_conv3 = nn.Conv2d(16, 32, 3, stride=2, padding = 1)
      self.latent_artist_conv3_bn = nn.BatchNorm2d(32)
      self.latent_artist_encode_conv4 = nn.Conv2d(32, 64, 3, stride=2, padding = 1)
      self.latent_artist_conv4_bn = nn.BatchNorm2d(64)
      self.latent_artist_encode_conv5 = nn.Conv2d(64, 128, 3, stride=2, padding = 1)
      self.latent_artist_conv5_bn = nn.BatchNorm2d(128)
      self.latent_artist_encode_conv6 = nn.Conv2d(128, 256, 3, stride=2, padding = 1)
      self.latent_artist_conv6_bn = nn.BatchNorm2d(256)

  def initialize_latent_artist_reparameterization(self):
      self.latent_artist_ln_encode_mean = nn.Linear(10752, self.artist_latent_size)
      self.latent_artist_ln_encode_variance = nn.Linear(10752, self.artist_latent_size)

  ######################################################################
  ######################################################################
  ########################### Forward Pass #############################
  ######################################################################
  ######################################################################

  def latent_artist_encode(self, x):
      x = F.relu(self.latent_artist_conv1_bn(self.latent_artist_encode_conv1(x)))
      x = F.relu(self.latent_artist_conv2_bn(self.latent_artist_encode_conv2(x)))
      x = F.relu(self.latent_artist_conv3_bn(self.latent_artist_encode_conv3(x)))
      x = F.relu(self.latent_artist_conv4_bn(self.latent_artist_encode_conv4(x)))
      x = F.relu(self.latent_artist_conv5_bn(self.latent_artist_encode_conv5(x)))
      x = F.relu(self.latent_artist_conv6_bn(self.latent_artist_encode_conv6(x)))

      mean, log_variance = self.latent_artist_ln_encode_mean(x.view(-1,10752)), self.latent_artist_ln_encode_variance(x.view(-1,10752))
      std = torch.exp(0.5*log_variance)
      ns = torch.randn_like(std)
      z = ns * std + mean

      return z, mean, log_variance

  def forward(self, x):
      z, mean, log_variance = self.latent_artist_encode(x)
      return z, mean, log_variance

# This is the base VAE that predicts time period and reconstructs the input image
class VAE_base(nn.Module):

    def __init__(self, latent_size, artist_latent_size, time_period_latent_size, num_artists):
        super(VAE_base, self).__init__()

        self.latent_size = latent_size
        self.artist_latent_size = artist_latent_size
        self.time_period_latent_size = time_period_latent_size
        self.num_artists = num_artists

        self.initialize_base_vae_enocder()
        self.initialize_base_vae_reparameterization()
        self.initialize_base_vae_decoder()
        self.initialize_latent_time_period_decoder()


        self.sigmoid = nn.Sigmoid()


    ######################################################################
    ######################################################################
    ####################### Network Initialization #######################
    ######################################################################
    ######################################################################


    ################ Base VAE Network ####################################
    def initialize_base_vae_enocder(self):
        self.base_encode_conv1 = nn.Conv2d(3, 8, 3, stride=2, padding = 1)
        self.base_conv1_bn = nn.BatchNorm2d(8)
        self.base_encode_conv2 = nn.Conv2d(8, 16, 3, stride=2, padding = 1)
        self.base_conv2_bn = nn.BatchNorm2d(16)
        self.base_encode_conv3 = nn.Conv2d(16, 32, 3, stride=2, padding = 1)
        self.base_conv3_bn = nn.BatchNorm2d(32)
        self.base_encode_conv4 = nn.Conv2d(32, 64, 3, stride=2, padding = 1)
        self.base_conv4_bn = nn.BatchNorm2d(64)
        self.base_encode_conv5 = nn.Conv2d(64, 128, 3, stride=2, padding = 1)
        self.base_conv5_bn = nn.BatchNorm2d(128)
        self.base_encode_conv6 = nn.Conv2d(128, 256, 3, stride=2, padding = 1)
        self.base_conv6_bn = nn.BatchNorm2d(256)

    def initialize_base_vae_reparameterization(self):
        self.base_ln_encode_mean = nn.Linear(10752, self.latent_size)
        self.base_ln_encode_variance = nn.Linear(10752, self.latent_size)
        self.base_ln_decode_variance = nn.Linear(self.latent_size, 4096)

    def initialize_base_vae_decoder(self):
        self.base_decode_trans_conv1 = nn.ConvTranspose2d(4096, 512, (5, 4), stride=2)
        self.base_conv1_trans_bn = nn.BatchNorm2d(512)
        self.base_decode_trans_conv2 = nn.ConvTranspose2d(512, 256, 5, stride=4, padding = 1)
        self.base_conv2_trans_bn = nn.BatchNorm2d(256)
        self.base_decode_trans_conv3 = nn.ConvTranspose2d(256, 128, 3, stride=3, padding = 2)
        self.base_conv3_trans_bn = nn.BatchNorm2d(128)
        self.base_decode_trans_conv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding = (2, 1))
        self.base_conv4_trans_bn = nn.BatchNorm2d(64)
        self.base_decode_trans_conv5 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding = (2, 0))
        self.base_conv5_trans_bn = nn.BatchNorm2d(32)
        self.base_decode_trans_conv6 = nn.ConvTranspose2d(32, 3, 2, stride=2, padding = (3,1), output_padding = (0,1))
        self.sigmoid = nn.Sigmoid()

    ################ Time Period Network ####################################

    def initialize_latent_time_period_decoder(self):
        self.latent_time_decode_1 = nn.Linear(self.time_period_latent_size, int(2 * self.time_period_latent_size))
        self.latent_time_bn_1 = nn.BatchNorm1d(2 * self.time_period_latent_size)
        self.latent_time_decode_2 = nn.Linear(2 * self.time_period_latent_size, self.time_period_latent_size)
        self.latent_time_bn_2 = nn.BatchNorm1d(self.time_period_latent_size)
        self.latent_time_decode_3 = nn.Linear(self.time_period_latent_size, int(self.time_period_latent_size / 2))
        self.latent_time_bn_3 = nn.BatchNorm1d(int(self.time_period_latent_size / 2))
        self.latent_time_decode_4 = nn.Linear(int(self.time_period_latent_size / 2), int(self.time_period_latent_size / 4))
        self.latent_time_bn_4 = nn.BatchNorm1d(int(self.time_period_latent_size / 4))
        self.latent_time_decode_5 = nn.Linear(int(self.time_period_latent_size / 4), 2)

    ######################################################################
    ######################################################################
    ########################### Forward Pass #############################
    ######################################################################
    ######################################################################

    def vae_base_encode(self, x):
        x = F.relu(self.base_conv1_bn(self.base_encode_conv1(x)))
        x = F.relu(self.base_conv2_bn(self.base_encode_conv2(x)))
        x = F.relu(self.base_conv3_bn(self.base_encode_conv3(x)))
        x = F.relu(self.base_conv4_bn(self.base_encode_conv4(x)))
        x = F.relu(self.base_conv5_bn(self.base_encode_conv5(x)))
        x = F.relu(self.base_conv6_bn(self.base_encode_conv6(x)))

        mean, log_variance = self.base_ln_encode_mean(x.view(-1,10752)), self.base_ln_encode_variance(x.view(-1,10752))
        std = torch.exp(0.5*log_variance)
        ns = torch.randn_like(std)
        z = ns * std + mean

        return z, mean, log_variance

    def vae_base_decode(self, variance):
        z = self.base_ln_decode_variance(variance)
        z = F.relu(self.base_conv1_trans_bn(self.base_decode_trans_conv1(z.view(-1,4096,1,1))))
        z = F.relu(self.base_conv2_trans_bn(self.base_decode_trans_conv2(z)))
        z = F.relu(self.base_conv3_trans_bn(self.base_decode_trans_conv3(z)))
        z = F.relu(self.base_conv4_trans_bn(self.base_decode_trans_conv4(z)))
        z = F.relu(self.base_conv5_trans_bn(self.base_decode_trans_conv5(z)))
        z = self.base_decode_trans_conv6(z)

        return self.sigmoid(z)

    def time_period_decode(self, z_timeperiod):
        z = F.relu(self.latent_time_bn_1(self.latent_time_decode_1(z_timeperiod)))
        z = F.relu(self.latent_time_bn_2(self.latent_time_decode_2(z)))
        z = F.relu(self.latent_time_bn_3(self.latent_time_decode_3(z)))
        z = F.relu(self.latent_time_bn_4(self.latent_time_decode_4(z)))
        z = self.latent_time_decode_5(z)

        prediction = self.sigmoid(z)
        return prediction[:,0], prediction[:,1]

    def forward(self, x):
        z, mean, log_variance = self.vae_base_encode(x)
        output_image = self.vae_base_decode(z)

        z_time_period = z[:, self.artist_latent_size : self.artist_latent_size + self.time_period_latent_size]

        time_mean, time_logvariance = self.time_period_decode(z_time_period)

        return output_image, mean, log_variance, z, time_mean, time_logvariance

# Create the two networks
vae = VAE_base(int(LATENT_SIZE), int(LATENT_ARTIST_SIZE), int(LATENT_TIME_PERIOD_SIZE), int(num_artists)).to(device)
artist_vae = VAE_latent_artist_encoding(int(LATENT_ARTIST_SIZE), int(num_artists)).to(device)
discriminator = VAE_latent_artist_discriminator(int(LATENT_ARTIST_SIZE)).to(device)

################################ CHECKPOINT LOADING ##############################
def find_latest_checkpoint_in_dir(load_dir, prefix):
  pretrained_models = os.listdir(load_dir)
  if '.ipynb_checkpoints' in pretrained_models:
    pretrained_models.remove('.ipynb_checkpoints')
  max_model_name = ""
  max_model_no = -1
  for pretrained_model in pretrained_models:
    prefix_split = pretrained_model.split(f"{prefix}")
    if len(prefix_split) != 2:
      continue
    model_no = int(prefix_split[1].split(".pt")[0])
    if model_no > max_model_no:
      max_model_no = model_no
      max_model_name = pretrained_model
  epoch = 0
  if max_model_name == "":
    print("Could not find pretrained model.")
    return None
  else:
    print(f"Loaded pretrained model: {max_model_name}")
    epoch = max_model_no
    return epoch, torch.load(f'{load_dir}/{max_model_name}')

# Checkpoint loading for the latent artist model

if IS_TRAINING_BASE_VAE:
  if(not find_latest_checkpoint_in_dir(load_model_folder, prefix) is None):
    base_epoch, base_state_dict = find_latest_checkpoint_in_dir(load_model_folder, prefix)
    epoch = base_epoch
    vae.load_state_dict(base_state_dict)
  if(not find_latest_checkpoint_in_dir(LATENT_ARTIST_MODEL_LOAD_PATH, f"{LATENT_VAE_PREFIX}_{LATENT_PREFIX}") is None):
    art_epoch, art_state_dict = find_latest_checkpoint_in_dir(LATENT_ARTIST_MODEL_LOAD_PATH, f"{LATENT_VAE_PREFIX}_{LATENT_PREFIX}")
    artist_vae.load_state_dict(art_state_dict)
  if(not find_latest_checkpoint_in_dir(LATENT_ARTIST_MODEL_LOAD_PATH, f"{DISCRIMINATOR_PREFIX}_{LATENT_PREFIX}") is None):
    discrimnator_epoch, discriminator_state_dict = find_latest_checkpoint_in_dir(LATENT_ARTIST_MODEL_LOAD_PATH, f"{DISCRIMINATOR_PREFIX}_{LATENT_PREFIX}")
    discriminator.load_state_dict(discriminator_state_dict)
else:
  if(not find_latest_checkpoint_in_dir(load_model_folder, f"{LATENT_VAE_PREFIX}_{prefix}") is None):
    art_epoch, art_state_dict = find_latest_checkpoint_in_dir(load_model_folder, f"{LATENT_VAE_PREFIX}_{prefix}")
    discriminator_epoch, discriminator_state_dict = find_latest_checkpoint_in_dir(load_model_folder, f"{DISCRIMINATOR_PREFIX}_{prefix}")

    artist_vae.load_state_dict(art_state_dict)
    epoch = art_epoch
    discriminator.load_state_dict(discriminator_state_dict)

# Optimizers for the two networks
base_optimizer = optim.Adam(vae.parameters(), lr=1e-3)
latent_optimizer = optim.Adam(artist_vae.parameters(), lr=1e-3)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3)

################################ HELPER METHODS ##############################

def display_first_n_of_batch(batch, n):
  f, axarr = plt.subplots(1, n)
  f.set_figheight(n)
  f.set_figwidth(2 * n)

  for i in range(n):
      imgorg = np.transpose(batch[i].cpu().detach().numpy(), (1,2,0))
      axarr[i].imshow(imgorg)

  plt.show()

################################ LOSS FUNCTIONS ##############################

def loss_function(recon_x, x, mean, logvariance, data_widths, data_heights):
    recon_x = recon_x.cpu()
    x = x.cpu()
    mean = mean.cpu()
    logvariance = logvariance.cpu()
    data_widths = data_widths.cpu()
    data_heights = data_heights.cpu()

    BCE = F.binary_cross_entropy(recon_x, x, reduction='none').cpu()
    mask_matrix = torch.zeros(recon_x.shape)

    # Iterate through each element in the batch
    for i in range(0, recon_x.shape[0]):
      mask_matrix[i,:,0:data_heights[i],0:data_widths[i]] = torch.ones((3, data_heights[i], data_widths[i]))

    # Normalize KL Divergence loss by batch size
    KLD =  0.5 * (torch.sum(mean.pow(2) + logvariance.exp() - logvariance - 1) / recon_x.shape[0])

    if MASK_TYPE == "mask":
      masked_BCE = BCE * mask_matrix
      return (KLD), (torch.sum(masked_BCE / (recon_x.shape[1] * torch.sum(data_widths * data_heights))))
    else:
      return (KLD), (torch.sum(BCE) / (recon_x.shape[0] * recon_x.shape[1] * recon_x.shape[2] * recon_x.shape[3]))

def time_period_loss(predicted_mean, predicted_sigma, gt_time_midpoint, gt_time_start):
  predicted_mean = predicted_mean.cpu()
  predicted_sigma = predicted_sigma.cpu()
  gt_time_midpoint = gt_time_midpoint.cpu()
  gt_time_start = gt_time_start.cpu()

  gt_time_start[gt_time_start < 0.0] = (gt_time_midpoint[gt_time_start < 0.0] - ( 1 / (max_date - min_date)))

  time_mu = gt_time_midpoint

  sigma = ((gt_time_midpoint - gt_time_start) / 2.0)
  normalized_sigma = (sigma - min_std_dev)/ (max_std_dev - min_std_dev)

  return torch.mean(kl_divergence_two_gaussians_std(predicted_mean, predicted_sigma, time_mu, normalized_sigma))

def discriminator_loss(prediction, gt):
  prediction = prediction.cpu()
  gt = gt.cpu()
  BCE_loss = nn.BCELoss()
  return BCE_loss(prediction, gt)

def discriminator_loss_no_reduction(prediction, gt):
  prediction = prediction.cpu()
  gt = gt.cpu()
  BCE_loss = nn.BCELoss(reduction='none')
  return BCE_loss(prediction, gt)

def discriminator_kl_divergence(mean, logvariance):
  mean = mean.cpu()
  logvariance = logvariance.cpu()

  KLD = 0.5 * torch.sum(mean.pow(2) + logvariance.exp() - logvariance - 1)
  return KLD

def artist_prediction_loss(artist_prediction_vector, gt_one_hot_artist, mean, logvariance):
  artist_prediction_vector = artist_prediction_vector.cpu()
  gt_one_hot_artist = gt_one_hot_artist.cpu()
  mean = mean.cpu()
  logvariance = logvariance.cpu()  

  BCE_loss = nn.BCELoss()
  KLD = 0.5 * (torch.sum(mean.pow(2) + logvariance.exp() - logvariance - 1) / artist_prediction_vector.shape[0])
  return ARTIST_PREDICTION_WEIGHT * BCE_loss(artist_prediction_vector.float(), gt_one_hot_artist.float()) + ARTIST_KLD_WEIGHT * KLD

def kl_divergence_two_gaussians(p_mean, p_logvar, q_mean, q_logvar):
  p_mean = p_mean.cpu()
  q_mean = q_mean.cpu()
  p_var = p_logvar.cpu().exp()
  q_var = q_logvar.cpu().exp()

  p = torch.distributions.normal.Normal(p_mean, torch.sqrt(p_var))
  q = torch.distributions.normal.Normal(q_mean, torch.sqrt(q_var))

  return torch.distributions.kl.kl_divergence(p, q)

def kl_divergence_two_gaussians_std(p_mean, p_std, q_mean, q_std):
  p_mean = p_mean.cpu()
  q_mean = q_mean.cpu()
  p_std = p_std.cpu()
  q_std = q_std.cpu()

  p = torch.distributions.normal.Normal(p_mean, p_std)
  q = torch.distributions.normal.Normal(q_mean, q_std)

  return torch.distributions.kl.kl_divergence(p, q)

def artist_kl_divergence(latent_mean, latent_logvar, pretrained_mean, pretrained_logvar):
  return torch.mean(torch.sum(kl_divergence_two_gaussians(latent_mean, latent_logvar, pretrained_mean, pretrained_logvar), dim = 0))

def triplet_loss(p_mean, p_logvar, n_mean, n_logvar, a_mean, a_logvar):
  positive_divergence = torch.mean(torch.sum(kl_divergence_two_gaussians(a_mean, a_logvar, p_mean, p_logvar), dim=0))
  negative_divergence = torch.clamp(TRIPLET_ALPHA - torch.sum(kl_divergence_two_gaussians(a_mean, a_logvar, n_mean, n_logvar), dim=0), min=0.0)
  return positive_divergence + torch.mean(negative_divergence)


def calculate_artist_discriminator_loss(image_batches, triplet_latent_vectors, corresponding_latent_vectors):
  # Here we go through all (6 choose 2) combinations of images and get the discriminator loss for each

  num_same_artists = 0.0
  num_diff_artists = 12 * triplet_latent_vectors[0].shape[0]

  same_artist_disc_loss = 0.0
  diff_artist_disc_loss = 0.0

  # FIRST CASE (3 Comparisons): The images of the same author (we check if they are also the same image, and if so we mask out that loss)
  for image_batch, triplet_latent, corr_latent in zip(image_batches, triplet_latent_vectors, corresponding_latent_vectors):
    should_mask_vector = image_batch["corresponding"]["should_mask"] == 0
    
    num_same_artists += torch.sum(should_mask_vector)

    latent_concat = torch.cat((triplet_latent, corr_latent), dim = 1)
    prediction = discriminator(latent_concat)
    
    disc_label = torch.ones(prediction.shape)

    same_artist_disc_loss += torch.sum(should_mask_vector * discriminator_loss_no_reduction(prediction.squeeze(), disc_label.float().squeeze()))
  
  # SECOND CASE (6 Comparisons): Compare each image from the triplet to each image in the corresponding that are not the same image
  for triplet_idx, triplet_latent in enumerate(triplet_latent_vectors):
    for corr_idx, corr_latent in enumerate(corresponding_latent_vectors):
      if (corr_idx == triplet_idx):
        continue

    latent_concat = torch.cat((triplet_latent, corr_latent), dim = 1)
    prediction = discriminator(latent_concat)
    
    disc_label = torch.zeros(prediction.shape)
    diff_artist_disc_loss += torch.sum(discriminator_loss_no_reduction(prediction.squeeze(), disc_label.float().squeeze()))
  
  # THIRD CASE (6 Comparisons): Compare each image from the same triplet
  latent_vectors_list = [triplet_latent_vectors, corresponding_latent_vectors]
  for latent_vectors_triplet in latent_vectors_list:
    for trip1_idx, trip1_latent in enumerate(latent_vectors_triplet):
      for trip2_idx, trip2_latent in enumerate(latent_vectors_triplet):
        if(trip1_idx == trip2_idx):
          continue
        latent_concat = torch.cat((trip1_latent, trip2_latent), dim = 1)
        prediction = discriminator(latent_concat)

        disc_label = torch.zeros(prediction.shape)
        diff_artist_disc_loss += torch.sum(discriminator_loss_no_reduction(prediction.squeeze(), disc_label.float().squeeze()))



  total_artist_loss = 0.0
  if torch.sum(num_same_artists) == 0:
    total_artist_loss = diff_artist_disc_loss / num_diff_artists
  else:
    total_artist_loss = (same_artist_disc_loss / num_same_artists) + (diff_artist_disc_loss / num_diff_artists)

  return total_artist_loss

################################ MODEL TESTING ##############################
def calculate_base_vae_test():
  with torch.no_grad():
    test_loss = 0
    for batch_idx, data in enumerate(test_dataset_loader):
        positive_image_batch = data['positive']
        negative_image_batch = data['negative']
        anchor_image_batch = data['anchor']

        image_batches = [positive_image_batch, negative_image_batch, anchor_image_batch]
        latent_vectors = []

        loss = None

        for image_batch in image_batches:
          output_image, mean, log_variance, z_artist, time_mean, time_logvariance = vae(image_batch['image'].to(device))

          latent_vectors.append(z)
          data_widths = image_batch['width']
          data_heights = image_batch['height']

          if loss is None:
            loss = loss_function(recon, data, mean, log_variance, data_widths, data_heights)
          else:
            loss += loss_function(recon, data, mean, log_variance, data_widths, data_heights)

          loss += TIME_PERIOD_WEIGHT * time_period_loss(time_period , image_batch['normalized_midpoint'], image_batch['normalized_start'], image_batch['normalized_end'])
          
          other_artist_latent_vector, _, _, _ = vae(image_batch['same_artist_image'].to(device))

          latent_concat = torch.cat((first_image_latent, other_image_latent), dim = 1)
          prediction = discriminator(latent_concat)

        loss = loss / len(latent_vectors)
        loss += TRIPLET_LOSS_WEIGHT * triplet_loss(latent_vectors[0], latent_vectors[1], latent_vectors[2])
        test_loss += loss.item()
  return test_loss / (len(test_dataset_loader) * batch_sz)

def calculate_loss_artist_test(artist_vae, discriminator):
  test_loss = 0
  for batch_idx, data in enumerate(test_latent_dataset_loader):
    print(f"Evaluating batch {batch_idx} for {model_name}")
    end = time.time()
    time_elapsed = end - start

    first_image = data["image"].to(device)
    other_image = data["other_image"].to(device)
    
    first_image_latent, first_mean, first_logvariance = artist_vae(first_image)
    other_image_latent, other_mean, other_logvariance = artist_vae(other_image)

    kl_first = discriminator_kl_divergence(first_mean, first_logvariance)
    kl_other = discriminator_kl_divergence(other_mean, other_logvariance)

    kl_loss = (kl_first + kl_other) / 2.0

    label = data["is_diff"].to(device)
    latent_concat = torch.cat((first_image_latent, other_image_latent), dim = 1)
    prediction = discriminator(latent_concat)
    
    if torch.isnan(prediction.squeeze()).any() or torch.isnan(label.float()).any():
      print("Prediction contained nan values. Skipping")
      continue
    loss = discriminator_loss(prediction.squeeze(), label.float()) + ARTIST_KLD_WEIGHT * kl_loss

    print(f"Discriminator Loss:{discriminator_loss(prediction.squeeze(), label.float())}")
    print(f"KL Loss: {ARTIST_KLD_WEIGHT * kl_loss}")

    print(loss.item())
    
    test_loss += loss.item()
  return test_loss

################################ MODEL TRAINING ##############################

def train_base_vae(vae, epochs, results_folder, weights_folder, prefix, start_epoch, log_folder):
    loss_array = []
    test_losses = []
    for epoch in range(start_epoch, epochs):
        train_loss = 0
        for batch_idx, data in enumerate(train_dataset_loader):
            print(f"Evaluating batch {batch_idx} for {model_name}")
            end = time.time()
            time_elapsed = end - start

            if not IS_RUNNING_NOTEBOOK:
              if time_elapsed >= TIMEOUT:
                  exit(0)

            base_optimizer.zero_grad()

            positive_image_batch = data['positive']
            negative_image_batch = data['negative']
            anchor_image_batch = data['anchor']

            image_batches = [positive_image_batch, negative_image_batch, anchor_image_batch]
            latent_mean = []
            latent_logvar = []

            triplet_latent_vectors = []
            corresponding_latent_vectors = []

            total_kld_loss = 0
            total_recon_loss = 0
            total_tp_loss = 0
            total_artist_loss = 0

            for image_batch in image_batches:
              triplet_keys = ["triplet", "corresponding"]
              for triplet_key in triplet_keys:
                image_metadata = image_batch[triplet_key]

                recon, mean, log_variance, latent_vector, time_mean, time_logvariance = vae(image_metadata['image'].to(device))

                # Extract out just the portion related to the artist
                z_artist = latent_vector[:, 0:LATENT_ARTIST_SIZE]

                if triplet_key == "triplet":
                  latent_mean.append(mean)
                  latent_logvar.append(log_variance)
                  triplet_latent_vectors.append(z_artist)
                else:
                  corresponding_latent_vectors.append(z_artist)

                data_widths = image_metadata['width']
                data_heights = image_metadata['height']

                kld_loss, recon_loss = loss_function(recon, image_metadata['image'].to(device), mean, log_variance, data_widths, data_heights)
                total_kld_loss += kld_loss
                total_recon_loss += recon_loss

                tp_loss = time_period_loss(time_mean, time_logvariance, image_metadata['normalized_midpoint'].to(device), image_metadata['normalized_start'].to(device))
                total_tp_loss += tp_loss

            
            total_artist_loss = calculate_artist_discriminator_loss(image_batches, triplet_latent_vectors, corresponding_latent_vectors)

            mean_kld_loss = (total_kld_loss / (2 * len(image_batches)))
            print(f"AVG KLD Loss: {mean_kld_loss}")

            mean_recon_loss = (total_recon_loss / (2 * len(image_batches)))
            print(f"AVG Recon Loss: {mean_recon_loss}")

            mean_tp_loss = (total_tp_loss / (2 * len(image_batches)))
            print(f"AVG Time Period Loss: {mean_tp_loss}")

            mean_artist_loss = total_artist_loss
            print(f"AVG Artist Discriminator Loss: {mean_artist_loss}")

            mean_triplet_loss =  triplet_loss(latent_mean[0], latent_logvar[0], latent_mean[1], latent_logvar[1], latent_mean[2], latent_logvar[2])
            print(f"Triplet Loss: {mean_triplet_loss}")

            loss = (VAE_DIVERGENCE_WEIGHT * mean_kld_loss) + (RECONSTRUCTION_WEIGHT * mean_recon_loss) + (TIME_PERIOD_WEIGHT * mean_tp_loss) + (LATENT_ARTIST_WEIGHT * mean_artist_loss) + (TRIPLET_LOSS_WEIGHT * mean_triplet_loss)
            print(f"Overall Loss: {loss}")

            if torch.isnan(loss) or loss.item() > MAX_VAE_LOSS_THRESHOLD:
              pickle_name = f"{log_folder}/epoch{epoch}_batch{batch_idx}.pickle"
              print("Loss outside usual range. Skipping and reporting error")
              continue
            
            loss.backward()
            train_loss += loss.item()
            grid = torchvision.utils.make_grid(recon[0:5])
            writer.add_image(f'Images/{model_name}', grid, epoch * len(train_dataset_loader) + batch_idx)
            writer.add_scalar(f'Loss/Train_{model_name}', loss.item(), epoch * len(train_dataset_loader) + batch_idx)

            base_optimizer.step()
        if epoch % interval == 0:

            ## Generate first 5 images of batch and save them to file
            f, axarr = plt.subplots(1, 5)
            f.set_figheight(5)
            f.set_figwidth(10)

            for i in range(min(5, recon.shape[0])):
                imgorg = np.transpose(recon[i].cpu().detach().numpy(), (1,2,0))
                axarr[i].imshow(imgorg)

            image_save_path = f'{results_folder}/{prefix}{epoch}.png'
            plt.savefig(f"{image_save_path}")
            plt.close()

            ## Calculate the test loss
            test_loss = calculate_base_vae_test()
            writer.add_scalar(f'Loss/Test_{model_name}', test_losses, epoch)

            print('====> Epoch: {} Average loss: {:.4f}'.format(
                  epoch, train_loss / len(train_dataset_loader.dataset)))

            ## Checkpointing
            save_filename = f'{prefix}{epoch}.pt'
            save_path = f'{weights_folder}/{save_filename}'
            torch.save(vae.state_dict(), save_path)
            print(f"Saving weights to: {weights_folder}")
            if epoch % 26 != 0:
                if os.path.exists(f'{weights_folder}/{prefix}{epoch - 1}.pt'):
                    os.remove(f'{weights_folder}/{prefix}{epoch - 1}.pt')
            print(f"Saved images results to: {image_save_path}")

    return artist_vae

if IS_TRAINING_BASE_VAE:
  train_base_vae(vae, epochs, results_folder, models_folder, prefix, epoch, log_folder)
else:
  train_latent_artist_vae(artist_vae, discriminator, epochs, results_folder, models_folder, prefix, epoch, log_folder)