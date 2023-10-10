# model
OB_RADIUS = 2       # observe radius, neighborhood radius
OB_HORIZON = 8      # number of observation frames
PRED_HORIZON = 12   # number of prediction frames
# group name of inclusive agents; leave empty to include all agents
# non-inclusive agents will appear as neighbors only
INCLUSIVE_GROUPS = []
model_hidden_dim = 64
n_clusters=70

smooth_size = 7
random_rotation = False
traj_seg = True

# training
lr = 3e-4 
dist_threshold = 2
batch_size = 128
epoch = 200       # total number of epochs for training
EPOCH_BATCHES = 100 # number of batches per epoch, None for data_length//batch_size
TEST_SINCE = 500    # the epoch after which performing testing during training

# testing
PRED_SAMPLES = 20   # best of N samples

# evaluation
WORLD_SCALE = 1