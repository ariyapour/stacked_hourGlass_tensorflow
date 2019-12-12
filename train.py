import sys

sys.path.insert(0, "./hourGlass_CNN/")
sys.path.insert(0, "./dataset_generator/")

from hourGlass_CNN.hourGlass import create_hourGlass_module
from dataset_generator.mpii_datagen import MPIIDataGen
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-nc", "--numClasses", type=int, default=16,
	help="number of classes")
ap.add_argument("-ns", "--numStacks", type=int, default=2,
	help="number of hourGlass stacks")
ap.add_argument("-nch", "--numChannels", type=int, default=256,
	help="number of residual bottleNeck channels")
ap.add_argument("-bs", "--batch_size", default=8, type=int,
    help='batch size for training')
ap.add_argument("-mp", "--model_path",
    help='path to store trained model')
ap.add_argument("-ep", "--epochs", default=20, type=int,
    help="number of traning epochs")
ap.add_argument("-rs", "--resume", default=False, type=bool,
    help="resume training or not")
ap.add_argument("-rsm", "--resume_model",
    help="start point to retrain")
ap.add_argument("-rsmj", "--resume_model_json",
    help="model json")
ap.add_argument("-c", "--checkpoints", required=True,
	help="path to output checkpoint directory")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
	help="epoch to restart training at")

args = vars(ap.parse_args())


train = MPIIDataGen()