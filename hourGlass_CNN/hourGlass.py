from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Add
from keras.models import Input
from keras.models import Model

def bottleneck(input, channels, name):
    shortcut = input
    #Make input channels same as output channel if the input has different number of channels, so we can add input with the output of last part of residual block
    if shortcut.shape[-1] != channels:
        shortcut = BatchNormalization()(shortcut)
        shortcut = Conv2D(channels,(1,1), padding="same", activation='relu', name=name + "skip_connection")(shortcut)

    #bottleneck block

    #First conv 1x1
    _next = BatchNormalization()(input)
    _next = Conv2D(channels//2,(1,1), padding="same", activation='relu', name=name + "conv_1x1_1")(_next)

    #Second conv 3x3
    _next = BatchNormalization()(_next)
    _next = Conv2D(channels//2,(3,3), padding="same", activation='relu', name=name + "conv_3x3")(_next)

    #Third conv 1x1
    _next = BatchNormalization()(_next)
    _next = Conv2D(channels,(1,1), padding="same", activation='relu', name=name + "conv_1x1_2")(_next)

    _next = Add(name=name + "residual_output")([shortcut, _next])

    return _next

def create_input_module(input, num_channels):
    #first module
    #first module reduces the input resolution to 1/4
    #conv2D with strite 2, filter size 7x7 and padding of same results 1/2 resolution
    #maxpooling results into another 1/2 of resolution

    next = BatchNormalization()(input)                                                             #Image resolution: 256x256xnum_channels
    next = Conv2D(64,(7,7), padding="same", activation='relu', strides=(2,2), name="input_conv_7x7_stride_2")(next) #Image resolution: 128x128xnum_channels TODO: why not num_channels?

    next = bottleneck(next, num_channels//2,"input_module_bottlenec_1")                          #Image resolution: 128x128xnum_channels TODO: Why num_bottleNeck_channels//2
    next = MaxPool2D(pool_size=(2,2), strides=(2,2))(next)                                       #Image resolution: 64x64xnum_channels

    next = bottleneck(next, num_channels//2, "input_module_bottlenec_2")                         #Image resolution: 64x64xnum_channels TODO: Why num_bottleNeck_channels//2
    next = bottleneck(next, num_channels, "input_module_bottlenec_3")                            #Image resolution: 64x64xnum_channels

    return next

def create_hourGlass_module(input, num_classes, num_bottleNeck_channels, hourGlassId):
    hourGName = "hg" + str(hourGlassId)
    input_shortCut = input

    #First bottleneck Block
    left_b1 = bottleneck(input, num_bottleNeck_channels, hourGName + "__bottleNeck1")                #Image resolution: 64X64xnum_bottleNeck_channels
    b1_shortcut = bottleneck(left_b1, num_bottleNeck_channels, hourGName + "_bottleNeck1_shortcut")  #Image resolution: 64x64xnum_bottleNeck_channels
    next = MaxPool2D(pool_size=(2,2), strides=(2,2))(left_b1)                                       #Image resolution: 32x32xnum_bottleNeck_channels

    #Second bottleneck Block
    left_b2 = bottleneck(next, num_bottleNeck_channels, hourGName + "__bottleNeck2")                #Image resolution: 32x32xnum_bottleNeck_channels
    b2_shortcut = bottleneck(left_b2, num_bottleNeck_channels, hourGName + "_bottleNeck2_shortcut")  #Image resolution: 32x32xnum_bottleNeck_channels
    next = MaxPool2D(pool_size=(2,2), strides=(2,2))(left_b2)                                       #Image resolution: 16x16xnum_bottleNeck_channels

    #Third bottleneck Block
    left_b3 = bottleneck(next, num_bottleNeck_channels, hourGName + "__bottleNeck3")                #Image resolution: 16x16xnum_bottleNeck_channels
    b3_shortcut = bottleneck(left_b3, num_bottleNeck_channels, hourGName + "_bottleNeck3_shortcut")  #Image resolution: 16x16xnum_bottleNeck_channels
    next = MaxPool2D(pool_size=(2,2), strides=(2,2))(left_b3)                                       #Image resolution: 8x8xnum_bottleNeck_channels

    #forth bottleneck Block
    left_b4 = bottleneck(next, num_bottleNeck_channels, hourGName + "_bottleNeck4")                 #Image resolution: 8x8xnum_bottleNeck_channels
    b4_shortcut = bottleneck(left_b4, num_bottleNeck_channels, hourGName + "_bottleNeck4_shortcut")  #Image resolution: 8x8xnum_bottleNeck_channels
    next = MaxPool2D(pool_size=(2,2), strides=(2,2))(left_b4)                                       #Image resolution: 4x4xnum_bottleNeck_channels TODO: Why not adding this max pooling?

    #creat bottom layer (3 consecutive bottlenecks)
    next = bottleneck(next, num_bottleNeck_channels, hourGName + "_bottleNeck_bottom_1")            #Image resolution: 4x4xnum_bottleNeck_channels
    next = bottleneck(next, num_bottleNeck_channels, hourGName + "_bottleNeck_bottom_2")            #Image resolution: 4x4xnum_bottleNeck_channels
    next = bottleneck(next, num_bottleNeck_channels, hourGName + "_bottleNeck_bottom_3")            #Image resolution: 4x4xnum_bottleNeck_channels

    next = UpSampling2D()(next)                                                                     #Image resolution: 8x8xnum_bottleNeck_channels
    next = Add()([next, b4_shortcut])                                                               #Image resolution: 8x8xnum_bottleNeck_channels
    next = bottleneck(next, num_bottleNeck_channels, hourGName + "_bottleNeck_right_4")             #Image resolution: 8x8xnum_bottleNeck_channels
   
    next = UpSampling2D()(next)                                                                     #Image resolution: 16x16xnum_bottleNeck_channels
    next = Add()([next, b3_shortcut])                                                               #Image resolution: 16x16xnum_bottleNeck_channels
    next = bottleneck(next, num_bottleNeck_channels, hourGName + "_bottleNeck_right_3")             #Image resolution: 16x16xnum_bottleNeck_channels

    next = UpSampling2D()(next)                                                                     #Image resolution: 32x32xnum_bottleNeck_channels
    next = Add()([next, b2_shortcut])                                                               #Image resolution: 32x32xnum_bottleNeck_channels
    next = bottleneck(next, num_bottleNeck_channels, hourGName + "_bottleNeck_right_2")             #Image resolution: 32x32xnum_bottleNeck_channels

    next = UpSampling2D()(next)                                                                     #Image resolution: 64x64xnum_bottleNeck_channels
    next = Add()([next, b1_shortcut])                                                               #Image resolution: 64x64xnum_bottleNeck_channels
    next = bottleneck(next, num_bottleNeck_channels, hourGName + "_bottleNeck_right_1")             #Image resolution: 64x64xnum_bottleNeck_channels

    #Intermediate supervision
    next = Conv2D(num_bottleNeck_channels, kernel_size=(1, 1), padding='same', activation='relu',
                    name=str(hourGName) + '_Intermediate_conv_1x1_1')(next)                         #Image resolution: 64x64xnum_bottleNeck_channels
    
    next = BatchNormalization()(next)                                                                 #Image resolution: 64x64xnum_bottleNeck_channels
    heatMap = Conv2D(num_classes, kernel_size=(1, 1), padding='same', activation='linear',
                    name=str(hourGName) + '_Intermediate_heatMap_conv_1x1_1')(next)                 #Image resolution: 64x64xnum_classes

    # next = BatchNormalization(next)                                                               #Image resolution: 64x64xnum_bottleNeck_channels
    next = Conv2D(num_bottleNeck_channels, kernel_size=(1, 1), padding='same', activation='linear',
                    name=str(hourGName) + '_Intermediate_conv_1x1_2')(next)                         #Image resolution: 64x64xnum_bottleNeck_channels

    reMapedHeatMap = Conv2D(num_bottleNeck_channels, kernel_size=(1, 1), padding='same', activation='linear',
                    name=str(hourGName) + '_Intermediate_heatMap_conv_1x1_2')(heatMap)       #Image resolution: 64x64xnum_bottleNeck_channels
    outputHead = Add()([next,input_shortCut, reMapedHeatMap])

    return outputHead, heatMap
    
    
def stack_hourGlass_modules(num_classes, num_stacks, num_channels, input_shape):
    input = Input(shape=(input_shape[0], input_shape[1], 3))

    hourGlass_next = create_input_module(input, num_channels)                                      #Create hourGlass first input module to reduce the resolution to 64x64xnum_channels
    outputs = []

    for i in range(num_stacks):
        hourGlass_next, heatmap_loss = create_hourGlass_module(hourGlass_next, num_classes, num_channels, i)
        outputs.append(heatmap_loss)
    model = Model(inputs=input, outputs=outputs)

    return model