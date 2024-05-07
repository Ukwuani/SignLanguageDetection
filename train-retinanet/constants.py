import torch

# Batch size for training
BATCH_SIZE = 4
# Resize image for training and transforms
RESIZE_TO = 640
# Number of parallel workers for data loading.
NUM_WORKERS = 4
# Number of epochs for training
NUM_EPOCHS = 40

# Directory to Paschal Voc Data for Training.
TRAIN_DIR = '../data/voc/train'

# Directory to Paschal Voc Data for Validation.
VALID_DIR = '../data/voc/valid'

# List of classes being worked with.
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

# Size of classes being worked with.
NUM_CLASSES = len(CLASSES)

# Whether to visualize images after creating the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots.
OUT_DIR = '../outputs'

# Device to use GPU (if available) or CPU
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
