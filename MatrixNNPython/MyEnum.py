from enum import Enum
class GradDescType(Enum):
    BATCH=0
    STOCHASTIC=1
    MINIBATCH=2
class ActivationFunction(Enum):
    SIGMOID=0
    SOFTMAX=1#avoid this when it is not binary classification
    ReLU=2
    TANH=3
class AnnealingAlgorithm(Enum):
    NOT_APPLIED=0
    MOMENTUM=1
    ADAGRAD=2
    ADAM=3
class PoolingType(Enum):
    NONE=0
    MAXPOOLING=1
    AVGPOOLING=2
