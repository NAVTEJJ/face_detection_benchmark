# Project hyperparameters

RANDOM_STATE           = 42
IMAGE_SIZE             = 32
BATCH_SIZE             = 64
LEARNING_RATE          = 1e-3
NUM_EPOCHS_CNN         = 15
NUM_EPOCHS_QNN         = 10
LFW_MIN_FACES          = 20
CIFAR_NEGATIVE_CLASSES = [0, 1, 8, 9]  # airplane, automobile, ship, truck

# quantum circuit configs
N_QUBITS               = 4
N_LAYERS               = 2
KERNEL_SIZE            = 2
STRIDE                 = 2
QUANTUM_OUT_CHANNELS   = 4

LATENCY_REPEATS        = 100
TEST_SPLIT             = 0.2
