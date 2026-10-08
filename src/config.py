# Project hyperparameters

RANDOM_STATE           = 42
IMAGE_SIZE             = 32
BATCH_SIZE             = 64
LEARNING_RATE          = 1e-3
NUM_EPOCHS_CNN         = 15
NUM_EPOCHS_QNN         = 10
LFW_MIN_FACES          = 20

# All ten CIFAR-10 classes act as negatives. The earlier [0, 1, 8, 9] set was
# airplane/automobile/ship/truck only: four rigid man-made categories with no
# animals, no fur or skin texture, and no centred blob structure. A detector
# trained against that set can separate the classes on image statistics alone
# without ever modelling a face. Animal classes (bird, cat, deer, dog, frog,
# horse) put eyes, texture and centred subjects on the negative side, which is
# what makes the task face detection rather than source discrimination.
# See src/baseline_g0.py for the measurement that justifies this.
CIFAR_NEGATIVE_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# quantum circuit configs
N_QUBITS               = 4
N_LAYERS               = 2
KERNEL_SIZE            = 2
STRIDE                 = 2
QUANTUM_OUT_CHANNELS   = 4

LATENCY_REPEATS        = 100
TEST_SPLIT             = 0.2

# Imbalanced ("deployment ratio") evaluation. A sliding-window detector sees
# far more background than faces, so a model tuned on a 1:1 split can look
# excellent and still drown in false positives. IMBALANCE_RATIO background
# patches are drawn per face, from a CIFAR pool held out of both train and
# test, so nothing here has been seen during training.
IMBALANCE_RATIO        = 10
# Safety cap on how many test faces enter the imbalanced set (the background
# pool is RATIO times larger). Batched circuit evaluation makes the quantum
# features cheap enough that the full test split fits comfortably, so this is
# set above the real face count and no subsampling happens in practice.
IMBALANCE_MAX_FACES    = 10_000

# Controlled multi-seed study (study.py). Every model gets the same epoch
# budget and the same rule for picking its epoch: lowest validation loss.
STUDY_SEEDS            = [0, 1, 2, 3, 4]
STUDY_EPOCHS           = 12
VAL_SPLIT              = 0.1   # fraction of the training split held out for epoch selection
