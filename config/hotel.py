# model
OB_RADIUS = 2       # observe radius, neighborhood radius
OB_HORIZON = 8      # number of observation frames
PRED_HORIZON = 12   # number of prediction frames
# group name of inclusive agents; leave empty to include all agents
# non-inclusive agents will appear as neighbors only
INCLUSIVE_GROUPS = []
model_hidden_dim = 128
n_clusters=90
smooth_size = 3
random_rotation = False
traj_seg = True

# training
lr = 1e-4 
batch_size = 128
dist_threshold = 2
epoch = 100       # total number of epochs for training

# testing
PRED_SAMPLES = 20   # best of N samples
FPC_SEARCH_RANGE = range(40, 50)   # FPC sampling rate

# evaluation
WORLD_SCALE = 1

######
# goal 0.13 0.19
# linear_INT 0.13/0.20
# nonlinear_INT  0.12/0.19

# segment 1
# smooth 1
# rotation 0