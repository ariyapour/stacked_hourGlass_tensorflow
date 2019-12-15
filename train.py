import sys
import os
import datetime


from hourGlass_CNN.hourGlass import stack_hourGlass_modules
from data_gen.mpii_datagen import MPIIDataGen
from eval_callback import EvalCallBack
from keras.optimizers import RMSprop
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model, model_from_json
from keras.losses import mean_squared_error

import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-nc", "--numClasses", type=int, default=16,
	help="number of classes")
ap.add_argument("-ns", "--numStacks", type=int, default=2,
	help="number of hourGlass stacks")
ap.add_argument("-nch", "--numChannels", type=int, default=256,
	help="number of residual bottleNeck channels")
ap.add_argument("-bs", "--batchSize", default=8, type=int,
    help='batch size for training')
ap.add_argument("-mp", "--modelPath",
    help='path to store trained model')
ap.add_argument("-ep", "--epochs", default=20, type=int,
    help="number of traning epochs")
ap.add_argument("-rs", "--resume", default=False, type=bool,
    help="resume training or not")
ap.add_argument("-rsm", "--resumeModel",
    help="start point to retrain")
ap.add_argument("-rsmj", "--resumeModelJson",
    help="model json")
ap.add_argument("-c", "--checkPoints", required=True,
	help="path to output checkPoint directory")
ap.add_argument("-s", "--startEpoch", type=int, default=0,
	help="epoch to restart training at")

args = vars(ap.parse_args())

dataset_path = "data/mpii/"
input_shape=[256,256]
output_shape=[64,64]
batch_size = args["batchSize"]
epochs = args["epochs"]

#Start training
if not args["resume"]:
    print("[INFO]Loading training dataset...")
    train = MPIIDataGen(dataset_path + "mpii_annotations.json", dataset_path + "images", inres=(256,256), outres=(64,64), is_train=True)
    train_generator = train.generator(batch_size, args["numStacks"], sigma=1, is_shuffle=True, rot_flag=True, scale_flag=True, flip_flag=True)

    csvlogger = CSVLogger(os.path.join(args["checkPoints"], "csv_train_" + str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))
    checkPoint = EvalCallBack(args["checkPoints"], (256,256), (64,64))

    callBacks = [csvlogger, checkPoint]

    model = stack_hourGlass_modules(args["numClasses"], args["numStacks"], args["numChannels"], (256,256))
    
    #show model summary
    model.summary()
    model.compile(optimizer=RMSprop(lr=5e-4), loss=mean_squared_error, metrics=["accuracy"])
    print("[INFO]Training...")
    model.fit_generator(generator=train_generator, steps_per_epoch=train.get_dataset_size() // batch_size,
                                 epochs=epochs, callbacks=callBacks)
    print("[INFO]Finished training...")
else:
    #load model
    print("[INFO]Loading the model...")
    with open(args["resumeModelJson"]) as f:
        model = model_from_json(f.read())
    model.load_weights(args["resumeModel"])

    model.compile(optimizer=RMSprop(lr=5e-4), loss=mean_squared_error, metrics=["accuracy"])
    print("[INFO]Loading training dataset...")
    train = MPIIDataGen(dataset_path + "mpii_annotations.json", dataset_path + "images", inres=(256,256), outres=(64,64), is_train=True)
    train_generator = train.generator(batch_size, args["numStacks"], sigma=1, is_shuffle=True, rot_flag=True, scale_flag=True, flip_flag=True)

    csvlogger = CSVLogger(os.path.join(args["checkPoints"], "csv_train_" + str(datetime.datetime.now().strftime('%H:%M')) + ".csv"))

    checkpoint = EvalCallBack(args["checkPoints"], (256,256), (64,64))

    callBacks = [csvlogger, checkpoint]
    print("[INFO]Training...")
    model.fit_generator(generator=train_generator, steps_per_epoch=train.get_dataset_size() // batch_size,
                                 initial_epoch=args["startEpoch"], epochs=epochs, callbacks=callBacks)
    print("[INFO]Finished training...")