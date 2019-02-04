# -*- coding: utf-8 -*-

# ======================================================================
# List
# SDR Capacity & Comparison (Episode 2), 0:45
# SDR Capacity & Comparison (Episode 2), 3:00
# SDR Capacity & Comparison (Episode 2), 7:36
# SDR Capacity & Comparison (Episode 2), 8:10
# SDR Capacity & Comparison (Episode 2), 10:52
# Scalar Encoding (Episode 5), 1:46
# Scalar Encoding (Episode 5), 8:33
# Scalar Encoding (Episode 5), 9:43
# Datetime Encoding (Episode 6), 3:16
# Datetime Encoding (Episode 6), 3:39
# Datetime Encoding (Episode 6), 4:11
# Spatial Pooling: Input Space & Connections (Episode 7), 4:10
# Spatial Pooling: Input Space & Connections (Episode 7), 8:25
# Spatial Pooling: Input Space & Connections (Episode 7), 9:16
# Spatial Pooling: Input Space & Connections (Episode 7), 10:18
# Spatial Pooling: Input Space & Connections (Episode 7), 17:06
# Spatial Pooling: Learning (Episode 8), 3:53
# Spatial Pooling: Learning (Episode 8), 4:00
# Spatial Pooling: Learning (Episode 8), 4:10
# Spatial Pooling: Learning (Episode 8), 6:50
# Spatial Pooling: Learning (Episode 8), 7:07
# Spatial Pooling: Learning (Episode 8), 8:07
# Spatial Pooling: Learning (Episode 8), 8:14
# Spatial Pooling: Learning (Episode 8), 8:49
# Boosting (Episode 9), 1:00
# Boosting (Episode 9), 2:26
# Boosting (Episode 9), 2:41
# Boosting (Episode 9), 3:54
# Boosting (Episode 9), 7:46
# Topology (Episode 10), 3:13
# Temporal Memory Part 1 (Episode 11), 1:38
# Temporal Memory Part 1 (Episode 11), 1:40
# Temporal Memory Part 1 (Episode 11), 2:58
# Temporal Memory Part 1 (Episode 11), 5:04
# Temporal Memory Part 1 (Episode 11), 6:00
# Temporal Memory Part 1 (Episode 11), 7:00
# Temporal Memory Part 1 (Episode 11), 7:10
# Temporal Memory Part 1 (Episode 11), 8:10
# Temporal Memory Part 1 (Episode 11), 10:56
# Temporal Memory Part 1 (Episode 11), 11:02
# Temporal Memory Part 1 (Episode 11), 16:30
# Temporal Memory Part 2 (Episode 12), 3:06
# Temporal Memory Part 2 (Episode 12), 3:35
# Temporal Memory Part 2 (Episode 12), 3:56
# Temporal Memory Part 2 (Episode 12), 4:08
# Cortical Circuitry (Episode 13), 1:34
# Cortical Circuitry (Episode 13), 1:52

# ======================================================================
from __future__ import division, print_function
import numpy as np
import csv
import subprocess
from datetime import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from nupic.encoders.date import DateEncoder
from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder
from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.algorithms.temporal_memory import TemporalMemory
from nupic.algorithms.sdr_classifier_factory import SDRClassifierFactory
from nupic.algorithms.anomaly import Anomaly
from nupic.algorithms.anomaly_likelihood import AnomalyLikelihood

# ======================================================================
# Not active cell: black
# Active cell: grey
# Predicted cell: yellow
# Winner cell: cyan

colors = [(0,0,0), (0.5,0.5,0.5), (1,1,0), (0,1,1)]

tm_cmap = LinearSegmentedColormap.from_list('tm', colors, N=4)
  
_INPUT_FILE_PATH = "./one_hot_gym_data.csv"

# ======================================================================
# Encoding Data

# @ Datetime Encoding (Episode 6), 3:16
# https://youtu.be/PTYlge2K1G8?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=196
# You can see how date type input data can be encoded by DateEncoder

# (21,1) means 
# bucket's width: 21
# radius: 1 hour
# c timeOfDayEncoder: you create date encoder which will encode timeOfDay input data
timeOfDayEncoder = DateEncoder(timeOfDay=(21,1))
# print("timeOfDayEncoder",timeOfDayEncoder)
# <nupic.encoders.date.DateEncoder object at 0x7f94ee892f10>

# bucket width: 21
# c weekendEncoder: you create date encoder which will encode weekend input data
weekendEncoder = DateEncoder(weekend=21)
# print("weekendEncoder",weekendEncoder)
# <nupic.encoders.date.DateEncoder object at 0x7f94f32a4fd0>

# Used resolution is 0.88
scalarEncoder = RandomDistributedScalarEncoder(resolution=0.88)
# print("scalarEncoder",scalarEncoder)
# RandomDistributedScalarEncoder:
#   minIndex:   500
#   maxIndex:   500
#   w:          21
#   n:          400
#   resolution: 0.88
#   offset:     None
#   numTries:   0
#   name:       [0.88]

# ======================================================================
# @ Datetime Encoding (Episode 6), 4:11 
# https://youtu.be/PTYlge2K1G8?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=251
# This explains how datetime input data can be encoded

# @ Spatial Pooling: Input Space & Connections (Episode 7), 4:10
# https://youtu.be/R5UoFNtv5AU?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=250

# c record: Make up some fake data composed of date type data 
# c record: and power consumption as scalar data
record = ['7/2/10 0:00', '21.2']

# c dateString: convert date string '7/2/10 0:00' into Python date object
dateString = dt.strptime(record[0], "%m/%d/%y %H:%M")

# c consumption: convert "data value" string '21.2' into float
consumption = float(record[1])

# --------------------------------------------------
# To encode input data, 
# you need to provide "numpy arrays" to encoders as placeholders

# @ Datetime Encoding (Episode 6), 3:39
# https://youtu.be/PTYlge2K1G8?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=219
# See "time of day"
# It has 54 cells.
# Even if it looks 2D array, just think of it as 1D flattened array 
# when it's used in real code like following 
# (504 length in following case)
# print("timeOfDayBits",timeOfDayBits.shape)
# timeOfDayBits (504,)

timeOfDayBits = np.zeros(timeOfDayEncoder.getWidth())

weekendBits = np.zeros(weekendEncoder.getWidth())

consumptionBits = np.zeros(scalarEncoder.getWidth())

# c timeOfDayEncoder: get width length of timeOfDayEncoder
# it's like equal to number of cells of "time of day" in video screenshot
# timeOfDayEncoder.getWidth() 504
# weekendEncoder.getWidth() 42
# scalarEncoder.getWidth() 400

# print("timeOfDayBits",timeOfDayBits.shape)
# print("weekendBits",weekendBits.shape)
# print("consumptionBits",consumptionBits.shape)
# timeOfDayBits (504,)
# weekendBits (42,)
# consumptionBits (400,)

# --------------------------------------------------
# @ Datetime Encoding (Episode 6), 3:39
# https://youtu.be/PTYlge2K1G8?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=219
# See "weekend"
# weekendBits placeholder array can be considered as 1D array filled by all 0s
# like all white cells without blue cells
# Values of 1 will be filled into weekendBits placeholder array 
# by encoder after finishing encoding

# In this sentence,
# weekendEncoder.encodeIntoArray(dateString,weekendBits)
# you have input data named dateString
# you have placeholder input space named weekendBits
# By using those 2, you can encode dateString data

# Encoded data is what you can see under "weekend" in video screenshot
# @ Datetime Encoding (Episode 6), 3:39
# https://youtu.be/PTYlge2K1G8?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=219

# c You perform encoding input data by using dateString (input data) and timeOfDayBits (input space array)
timeOfDayEncoder.encodeIntoArray(dateString,timeOfDayBits)

# c You perform encoding input data by using dateString (input data) and weekendBits (input space array)
weekendEncoder.encodeIntoArray(dateString,weekendBits)

# @ Temporal Memory Part 1 (Episode 11), 1:38
# https://youtu.be/UBzemKcUoOk?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=98
# Green grid is input space
# Green circles in input space are on-bits

# c You perform encoding input data by using consumption (input data) and consumptionBits (input space array)
scalarEncoder.encodeIntoArray(consumption,consumptionBits)

# --------------------------------------------------
# @ Datetime Encoding (Episode 6), 3:39
# https://youtu.be/PTYlge2K1G8?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=219
# See "entire encoding"
# Following encoding is "entire encoding" after concatenation

# c Concatenate all encodings into one entire encoding
encoding=np.concatenate(
    (timeOfDayBits,weekendBits,consumptionBits))
# print("encoding",encoding.shape)
# encoding (946,)

# --------------------------------------------------
np.set_printoptions(threshold=np.nan)
# print(encoding.astype("int16"))

np.set_printoptions(threshold=1000)
# [1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
#  0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
#  1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
#  0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0
#  0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

# You can see continuous sections of "on bits" for date encodings. 
# Near bottom, you can see "randomly" distributed on-bits for scalar encoder. 

# ======================================================================
# Another (better?) way to visualize 
# this is to plot bitarray.

plt.figure(figsize=(15,2))
plt.plot(encoding)
# plt.show()

# ======================================================================
# Before moving on, 
# let's write convenience function to encode data for us.

def encode(file_record):
    dateString = dt.strptime(file_record[0], "%m/%d/%y %H:%M")
    consumption = float(file_record[1])

    timeOfDayBits = np.zeros(timeOfDayEncoder.getWidth())
    weekendBits = np.zeros(weekendEncoder.getWidth())
    consumptionBits = np.zeros(scalarEncoder.getWidth())

    timeOfDayEncoder.encodeIntoArray(dateString, timeOfDayBits)
    weekendEncoder.encodeIntoArray(dateString, weekendBits)
    scalarEncoder.encodeIntoArray(consumption, consumptionBits)

    return np.concatenate((timeOfDayBits, weekendBits, consumptionBits))

# ======================================================================
# Spatial Pooling

# c encodingWidth: you get width of entire encoding
encodingWidth=timeOfDayEncoder.getWidth()+\
              weekendEncoder.getWidth()+\
              scalarEncoder.getWidth()
# print("encodingWidth",encodingWidth)
# encodingWidth 946

# @ Temporal Memory Part 1 (Episode 11), 1:40
# https://youtu.be/UBzemKcUoOk?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=100
# 4 vertically located circles consist of one minicolumn

# To clarify concept of column in spatial pooler, 
# try to imagine there is one 3x3x3 cube
# 1x1x3 shape "3 vertically located squares" can be considered as one column, 
# in other words, one minicolumn

# 3x3x3 structure which is composed of minicolumns is cortical column
# See "Cortical Region" in following link
# https://discourse.numenta.org/t/htm-cheat-sheet/828

# @ Temporal Memory Part 1 (Episode 11), 2:58
# https://youtu.be/UBzemKcUoOk?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=178
# Above screenshot shows synapse structure
# (I guess above connections can be considered as connection 
# between "spine" of one neuron and "other spine" of other neuron biologically like following images
# https://www.google.com/search?q=neuron+spine&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiwiYik2KHgAhUEfrwKHWRmAJMQ_AUIDigB&biw=1881&bih=938) 

# Take example of skin which is one of sensory organs for somatic sense.
# There can be many areas on skin which can take stimulus from outside.
# You can consider those many skin areas as many input spaces.
# And each area on skin will have different potential pool,
# for example, I think skin on elbow is weaker (small potential pool?) in its sensory functionality
# like following where various potential pools in input space show up
# @ Spatial Pooling: Input Space & Connections (Episode 7), 8:25
# https://youtu.be/R5UoFNtv5AU?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=505

# @ Spatial Pooling: Input Space & Connections (Episode 7), 17:06
# https://youtu.be/R5UoFNtv5AU?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=1026
# Why each column has different patterns of connection for "one input data"?
# I guess one input data can fire multiple minicolumns
# (one stimulus can fire multiple neurons in biological perspective)
# And also I guess one input data can fire multiple cells in minicolumn

# This is relevant explanation
# @ SDR Capacity & Comparison (Episode 2), 0:45
# https://youtu.be/ZDgCdWTuIzc?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=45
# There are many auditory neurons in temporal lobe
# Even if when you hear some sound, 
# many of them are still inactive, only some neurons become active

# I guess it sounds reasonable and efficient
# if "multiple cells" in minicolumn extract different features and context from one input data

# I guess this is observed as almost same logic in CNN methodology of deep learning
# in where multiple image filters extract multiple spatial features from one image.

# Anyway, by using several criterions 
# like threshold on permanance values in potential pools,
# spatial pooler can extract various features from one input data.
# And I guess those various features from one input data can be represented 
# by various patterns of connection at each column against input data in input space.

# And I guess "one final SDR" in spatial pooler 
# which semantically represents "one input data" is generated through above step.

# Anyway, final relationship seems 
# like "one number of input data" corresponds to "one number of SDR in spatial pooler".

# It's just way to find "one number of SDR in spatial pooler" 
# which best semantically against spatial fatures 
# and robustly against noises can represent "one number of input data" 
# I guess you need multiple patterns of connection 
# between each column and input data

# Explanations which seems related come in
# @ Temporal Memory Part 2 (Episode 12), 3:06
# https://youtu.be/1OhY_u3NjdM?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=186

# Being returned to original topic, 
# according to following part, 
# @ Temporal Memory Part 1 (Episode 11), 1:40
# https://youtu.be/UBzemKcUoOk?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=100
# you'll think of just "one input space" than "multiple input spaces"

# I guess many neurons 
# (I personally consider one minicolumn as one neuron, 
# and I consider each cell in one minicolumn as each spine) 
# are connected to that one input space via nerve system.

# @ Cortical Circuitry (Episode 13), 1:34
# https://youtu.be/mPFx9yuV1Os?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=94
# According to this, neocortex is composed of multiple cortical columns

# @ Cortical Circuitry (Episode 13), 1:52
# https://youtu.be/mPFx9yuV1Os?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=112
# One cortical column is composed of thousands of pyramidal neurons (thousands of minicolumns)

# Anyway, input data first comes into input space 
# like stimulus first comes into body via sensory organ
# and it's passed to spatial pooler.
# And that's "proximal mechanism"
# And spatial pooler processes that raw stimulus 
# to extract features (and better feature when learning and boosting are allowed) of stimulus (input data)
# to normalize it for it to have same sparcity

# And I personally consider "distal mechanism" as connections (which is synapses biologically)
# from spine (one cell in minicolumn) to other spine (other cell in minicolumn)

# c sp: you create spatial pooler
sp=SpatialPooler(
    inputDimensions=(encodingWidth,),
    columnDimensions=(2048,),
    # By using encodingWidth for potentialRadius,
    # let's "every column" of spatial pooler 
    # to see "every cell" of input space
    potentialRadius=encodingWidth,
    # @ Spatial Pooling: Input Space & Connections (Episode 7), 9:16
    # https://youtu.be/R5UoFNtv5AU?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=556
    potentialPct=0.85,
    # @ Spatial Pooling: Learning (Episode 8), 4:10
    # https://youtu.be/rHvjykCIrZM?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=250
    globalInhibition=True,
    localAreaDensity=-1.0,
    # this value (like 40.0) / "total number of columns" = sparsity (40/2048 ~ 2%)
    # @ Spatial Pooling: Learning (Episode 8), 4:00
    # https://youtu.be/rHvjykCIrZM?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=240
    numActiveColumnsPerInhArea=40.0, 
    stimulusThreshold=0,
    # @ Spatial Pooling: Learning (Episode 8), 8:14
    # https://youtu.be/rHvjykCIrZM?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=494
    # How much permanence values decremented when spatial pooler is being diminished?
    synPermInactiveDec=0.005,
    # @ Spatial Pooling: Learning (Episode 8), 8:07
    # https://youtu.be/rHvjykCIrZM?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=487
    # How much permanence values incremented when spatial pooler is being reinforced?
    synPermActiveInc=0.04,
    synPermConnected=0.1,
    # @ Boosting (Episode 9), 2:41
    # https://youtu.be/MSwoNAODrgk?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=161
    # min gree box having 0.00?
    minPctOverlapDutyCycle=0.001,
    # @ Boosting (Episode 9), 2:26
    # https://youtu.be/MSwoNAODrgk?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=146
    # "certain period of time"
    dutyCyclePeriod=100,
    boostStrength=0.0,
    seed=42,
    spVerbosity=0,
    wrapAround=False)

# print("sp",sp)
# <nupic.algorithms.spatial_pooler.SpatialPooler object at 0x7fa7f50cbe50>

# ======================================================================
# Running SP

# @ Spatial Pooling: Learning (Episode 8), 3:53
# https://youtu.be/rHvjykCIrZM?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=233
# In right spatial pooler, you can see green colored square-shape columns
# Those green colored square-shape columns are active column

# How are those active columns determined?
# You can (probably) configure parameter for overlap score
# @ SDR Capacity & Comparison (Episode 2), 8:10
# https://youtu.be/ZDgCdWTuIzc?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=490

# @ SDR Capacity & Comparison (Episode 2), 10:52
# https://youtu.be/ZDgCdWTuIzc?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=652
# When you compare 2 SDRs to see whether they're match or not,
# they don't need to be 100% matched to be matched.
# Match or not match is determined based on configured threshold theta. 

# And with that configured overlap score parameter,
# and if some column in right spatial pooler has more number of connections 
# (which are finally represented by green circles in left input space) 
# which fall into blue cells (which mean on-bits in input space) in left input space
# that column will be categorized into active column

# Gray circles represent connections from that column to input data
# but cell which has gray circle is not on-bit. It's off-bit (0 value)

# And connections from column to input space are determined by threshold and permanence value
# @ Spatial Pooling: Learning (Episode 8), 6:50 
# https://youtu.be/rHvjykCIrZM?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=410 

# How to learn spatial pooler?
# In other words, how to make spatial pooler be more precise when making connections between columns and input space?
# @ Spatial Pooling: Learning (Episode 8), 7:07
# https://youtu.be/rHvjykCIrZM?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=427
# Answer: as permanance values are dynamically up and down, 
# connections can be created and destroyed
# and that adjusting will make better connections from column to input data

# c activeColumns: Create "array" to represent "active columns" in spatial pooler SDR
# c activeColumns: Placeholder for active columns is populated by spatial_pooler.compute()
# c activeColumns: "Active columns" (like 2048) must have same dimensions as "Spatial Pooler" (like 2048)
activeColumns = np.zeros(2048)
# print("activeColumns",activeColumns)
# [ 0.  0.  0. ...,  0.  0.  0.]

activeColumnIndices_before_sp = np.nonzero(activeColumns)[0]
# print("activeColumnIndices_before_sp",activeColumnIndices_before_sp)
# []

# @ Spatial Pooling: Learning (Episode 8), 8:49
# https://youtu.be/rHvjykCIrZM?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=529
# I guess you can turn on-and-off learning capability of spatial pooler
# by using True or False in sp.compute(encoding, True, activeColumns)

# If you say True, you dynamically increment or decrement permanance values 
# to adjust connections and to finally find best connection pattern 
# between input data in input space and each column in spatial pooler

# @ Topology (Episode 10), 3:13
# https://youtu.be/HTW2Q_UrkAw?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=193
# You first prepare placeholder array for activeColumns
# activeColumns placeholder array will be populated by spatial pooler

# c Execute "Spatial Pooling algorithm" with "input data" called encoding 
# and placeholder for active column called activeColumnIndices
sp.compute(encoding, True, activeColumns)

# c activeColumnIndices: non zero indices from active columns which are populated by spatial pooler
activeColumnIndices=np.nonzero(activeColumns)[0]
# print("activeColumnIndices",activeColumnIndices)
# [  15  127  212  213  318  438  444  468  531  591  606  630  646  717  720
#   756  851  861  870  883  909 1014 1112 1181 1190 1219 1541 1568 1611 1696
#  1776 1784 1842 1846 1850 1885 1919 1943 1962 2030]

# ======================================================================
def showSDR(file_record):
    """
    Act
      * Visualize "input data" and "SDR of spatial pooler"
    """
    # print("sp",sp)
    # sp <nupic.algorithms.spatial_pooler.SpatialPooler object at 0x7fc6d17d3ed0>

    # c encoding: encode file_record input data
    encoding=encode(file_record)
    # print("encoding",encoding)
    # [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  1.  1.
    #   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
    #   1.  1.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.
    #   0.  0.  0.  1.  0.  0.  0.  0.  1.  0.  1.  1.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.
    #   0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  1.  1.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  1.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   1.  0.  1.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.
    #   0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]

    # c col_dim_sp: column dimension of SP
    col_dim_sp=sp.getColumnDimensions()
    # print("col_dim_sp",col_dim_sp)
    # [2048]

    # c activeCols: placeholder for active column which will be populated by SP
    activeCols = np.zeros(col_dim_sp)

    # Use SP
    sp.compute(encoding, False, activeCols)

    nEN = int(np.math.ceil(encodingWidth**0.5))
    nSP = int(np.math.ceil(sp.getColumnDimensions()[0]**0.5))
    fig, ax = plt.subplots(1,2,figsize=(15,8))
    imgEN = np.ones(nEN**2)*-1
    imgEN[:len(encoding)] = encoding
    imgSP = np.ones(nSP**2)*-1
    imgSP[:len(activeCols)] = activeCols
    ax[0].imshow(imgEN.reshape(nEN,nEN))
    ax[1].imshow(imgSP.reshape(nSP,nSP))

    for a in ax:
        a.tick_params(
            axis='both',
            which='both',
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False)

    ax[0].set_title('Encoder output', fontsize=20)
    ax[1].set_title('SDR', fontsize=20)
    ax[0].set_ylabel("{0}  --  {1}".format(*file_record), fontsize=20)
    plt.show()

# showSDR(['7/2/10 2:00',4.7])
showSDR(['7/2/10 2:00',5.5])

# ======================================================================

# c tm: creating TM
tm=TemporalMemory(
    # Must be same dimensions as SP
    columnDimensions=(2048,), 
    # reduced from default
    cellsPerColumn=16, 
    # increased from default
    activationThreshold=16, 
    # make synapse connect when it's created
    initialPermanence=0.55, 
    connectedPermanence=0.5, 
    # increased from default
    minThreshold=12, 
    maxNewSynapseCount=20,
    permanenceIncrement=0.1, 
    # 1/5th of default to slow down "forgetting"
    permanenceDecrement=0.02, 
    predictedSegmentDecrement=0.0, 
    seed=42, 
    # reduced from default
    maxSegmentsPerCell=128, 
    # reduced from default
    maxSynapsesPerSegment=32)

# Choose value based on Numenta's recommendation
# @ Spatial Pooling: Learning (Episode 8), 4:00
# https://youtu.be/rHvjykCIrZM?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=240
num_a_col_per_inh_a=sp.getNumActiveColumnsPerInhArea()
col_dim=sp.getColumnDimensions()[0]
perman_inc=tm.permanenceIncrement
# print("num_a_col_per_inh_a",num_a_col_per_inh_a)
# print("col_dim",col_dim
# print("perman_inc",perman_inc)
# sp.getNumActiveColumnsPerInhArea() 40
# sp.getColumnDimensions() [2048]
# tm.permanenceIncrement 0.1

tm.predictedSegmentDecrement=num_a_col_per_inh_a/(col_dim*perman_inc)

# ======================================================================
# Running TM

# Case where values (10.2, 21.4, 30.7) are semantically close
# records = [('7/2/10 0:00', '10.2'),
#            ('7/2/10 1:00', '21.4'),
#            ('7/2/10 2:00', '30.7')] * 2

# Case where values (5.2, 20.7, 86.4) are semantically far apart
records = [('7/2/10 0:00', '5.2'),
           ('7/4/10 18:00', '20.7'),
           ('7/2/10 12:00', '86.4')] * 2

nPlots = len(records)
fig, ax = plt.subplots(nPlots,1,figsize=(15, nPlots*2.5))

for i,record in enumerate(records):
    # print("i",i)
    # i 0

    # print("record",record)
    # ('7/2/10 0:00', '5.2')
    
    # c encoded_r: encode record by encoder
    encoded_r=encode(record)
    
    # print("encoded_r",encoded_r)
    # [1. 1. 1. 1. 1. 1.

    # print("encoded_r",encoded_r.shape)
    # encoded_r (946,)

    # print("activeColumns",activeColumns.shape)
    # activeColumns (2048,)

    # Use spatial pooler with encoded_r and placeholder of activeColumns
    sp.compute(encoded_r, False, activeColumns)

    # c activeColumnIndices: active columns from spatial pooler
    activeColumnIndices = np.nonzero(activeColumns)

    activeColumnIndices=activeColumnIndices[0]

    # --------------------------------------------------
    # Temporal Memory Part 1 (Episode 11), 6:00
    # Temporal memory learning sees all active columns (yellow ones) at first stage of learing
    # Then, bursting is performed 
    # when there is no predictive cells in active columns
    # In other words, bursting happens when temporal learning algorithm sees something for the first time
    # c Use temporal memory algorithm
    tm.compute(activeColumnIndices,learn=True)

    # c get all cells from one column
    cells_per_col=tm.getCellsPerColumn()
    
    # c col_dim: get number of columns
    col_dim=tm.getColumnDimensions()
    
    # print("cells_per_col",cells_per_col)
    # cells_per_col 16

    # print("col_dim",col_dim)
    # col_dim (2048,)

    # c imgTM: placeholder for image obtained from TM, 3x3x3 shape-like
    imgTM = np.zeros(cells_per_col*col_dim[0])

    # @ Temporal Memory Part 1 (Episode 11), 7:00
    # https://youtu.be/UBzemKcUoOk?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=420
    # When there is no bursting,
    # active cells are orangish yellow ones

    # @ Temporal Memory Part 2 (Episode 12), 3:56
    # https://youtu.be/1OhY_u3NjdM?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=236
    # About burst

    # c Get "active cells" from TM and assign 1 into them
    imgTM[tm.getActiveCells()] = 1
    
    # @ Temporal Memory Part 1 (Episode 11), 7:10
    # https://youtu.be/UBzemKcUoOk?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=430
    # c Get "predictive cells" from TM and assign 2 into them
    imgTM[tm.getPredictiveCells()] = 2
    
    # @ Temporal Memory Part 1 (Episode 11), 8:10
    # https://youtu.be/UBzemKcUoOk?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=490
    # "How would these cells become predictive state" is related 
    # to mechanism for choosing winner cell 
    # which is 2nd phase of temporal memory algorithm

    # @ Temporal Memory Part 1 (Episode 11), 16:30
    # https://youtu.be/UBzemKcUoOk?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=990
    # Another meaning of winner cell is for selecting specific cell based on context
    # The word eat' which is followed by boys in the sequential context is represented 
    # by winner 3 glittering cells as screenshot
    # So, note that raw representation on word eat (having all cells at 1st, 5th, (last-1)th)
    # (see "eat" under "Active cells (before learning)")
    # is different with representation (having 3 glittering cells in each column at 1st, 5th, (last-1)th) 
    # on eat' which is shown in the context of boys and many'
    # In this way, computer can distinguish those same word but showing in different context
    # and can learn it

    # @ Temporal Memory Part 2 (Episode 12), 3:35
    # https://youtu.be/1OhY_u3NjdM?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=215
    # Another meaning of winner cell is for selecting specific cell based on context

    # @ Temporal Memory Part 2 (Episode 12), 4:08
    # https://youtu.be/1OhY_u3NjdM?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=248
    # Winner cell idea

    # c Get "winner cells" from TM and assign 3 into them
    imgTM[tm.getWinnerCells()] = 3

    # c re_imgTM: reshape imgTM into 2D image shape
    re_imgTM=imgTM.reshape((cells_per_col,col_dim[0]), order='F')

    ax[i].imshow(re_imgTM,cmap=tm_cmap)
    ax[i].set_aspect(20)
    ax[i].set_ylabel('{} ({}/{})'.format({0:'A',1:'B',2:'C'}[i % 3],i,len(records)-1),
                     fontsize=20,rotation=0,labelpad=40)
    ax[i].tick_params(axis='both',which='both',bottom=False,labelbottom=False,
                      left=False, labelleft=False)

    if i % 3 == 2:
        tm.reset()
# plt.show()

# ======================================================================
# Try varying number of steps and alpha
num_of_steps = 1
alpha = 0.01

# c classifier: SDRClassifier affected by num_of_steps, alpha
classifier = SDRClassifierFactory.create(steps=[num_of_steps], alpha=alpha)

# Make up some things to classify
# c vals: made up values
vals = [4.5, 23.4, 56.7]

# c bucketIdx: made up bucket indices
bucketIdx = [1, 5, 9]

# c cellIdx: made up "active cells indices" from TM
cellIdx = []
cellIdx.append([1449, 2009, 602, 1810])
cellIdx.append([901, 570, 377, 2005])
cellIdx.append([87, 516, 1270, 232])

# ======================================================================
# Let classifier learn them (above data) over N cycles

# if you've done any work with gradient descent, 
# you'll know that to get good predictions, 
# you will need "larger N" for "smaller alpha"

N = 1000
for i in range(3*N):
    # c dict_f_clsc: dict for classification
    dict_f_clsc={"bucketIdx": bucketIdx[i%3], "actValue":vals[i%3]}

    classifier.compute(
        recordNum = i, 
        patternNZ = cellIdx[i%3],
        classification=dict_f_clsc, 
        learn=True, 
        infer=False)

print("Sequence trained on: {{{0}, {1}, {2}}}".format(*vals))
print("Predicting what comes {} steps later\n".format(num_of_steps))
# Sequence trained on: {4.5, 23.4, 56.7}
# Predicting what comes 1 steps later

# --------------------------------------------------
for j in range(3):
    dict_for_cls={"bucketIdx": bucketIdx[j], "actValue":vals[j]}

    result = classifier.compute(
        recordNum = None, 
        patternNZ = cellIdx[j],
        classification=dict_for_cls,
        learn=False,
        infer=True)

    # c res_num_s: result of num_of_steps
    res_num_s=result[num_of_steps]

    # c res_actualValues: result of actualValues
    res_actualValues=result["actualValues"]

    a = sorted(zip(res_num_s, res_actualValues), reverse=True)[:3]
    
    print("When values are",vals) 
    print("If you see {}, {} steps later, you'll see:".format(vals[j], num_of_steps))

    for x in a:
        print("  {0:.2f}% chance of seeing {1}".format(x[0]*100, x[1]))

# When values are [4.5, 23.4, 56.7]
# If you see 4.5, 1 steps later, you'll see:
#   87.06% chance of seeing 23.4
#   3.23% chance of seeing 56.7
#   3.23% chance of seeing 4.5
# When values are [4.5, 23.4, 56.7]
# If you see 23.4, 1 steps later, you'll see:
#   97.49% chance of seeing 56.7
#   0.28% chance of seeing 23.4
#   0.28% chance of seeing 23.4
# When values are [4.5, 23.4, 56.7]
# If you see 56.7, 1 steps later, you'll see:
#   69.01% chance of seeing 4.5
#   3.87% chance of seeing 56.7
#   3.87% chance of seeing 56.7

# ======================================================================
# Detecting Anomalies

# Make "sine wave" with "two anomalies"
fig, ax = plt.subplots(2,1,figsize=(15,4))
t = np.arange(0,100*2*np.pi,2*np.pi/20)
swave = np.sin(t)

# spatial anomaly
swave[1110] = 1

# temporal anomaly
swave[1280:1700] = np.sin(t[1280:1700]*3)

ax[0].plot(swave)

# show anomalies
ax[1].plot(swave[1000:1750],'-')
plt.show()

# --------------------------------------------------
np.sort(np.unique(np.abs(np.diff(swave[:]))))[0]
# 0.04894348370482271

# --------------------------------------------------
# @ SDR Capacity & Comparison (Episode 2), 3:00
# https://youtu.be/ZDgCdWTuIzc?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=180
# You can see grid.
# White cell means 0 (off-bit), blue cell means 1 (on-bit)

# In code, it seems like you use 1D array rather than 2D array
# I guess, for simplicity, lecturer Matt uses 2D array in video lecture
# Try to find "c d1" and "c d2" in this code file by pressing CTRL+F
# You will see encoded array from scalar values via encoders
# It'll show 1D array which is 400 length

# What you're doing here is to create encoder
# And this encoder you're creating can be used to encode scalar value like 1, 1.4, ...

# And by using RandomDistributedScalarEncoder, 
# you'll get "encoded input data" whose form is randomly distributed form

# Opposed form of RandomDistributedScalarEncoder is dense form encoding 
# where 1 values show up in sequence like 
# [0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 ]
# showing in @ Scalar Encoding (Episode 5), 1:46
# https://youtu.be/V3Yqtpytif0?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=106

# RandomDistributedScalarEncoder creates array where "on-bits" cells are scattered randomly in array
# RandomDistributedScalarEncoder shows up in 
# @ Scalar Encoding (Episode 5), 8:33
# https://youtu.be/V3Yqtpytif0?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=514

# resolution shows in @ Scalar Encoding (Episode 5), 9:43
# https://youtu.be/V3Yqtpytif0?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=583
# c encoder: RandomDistributedScalarEncoder instance
encoder = RandomDistributedScalarEncoder(resolution=0.1)
# print("encoder",encoder)
# RandomDistributedScalarEncoder:
#   minIndex:   500
#   maxIndex:   500
#   w:          21
#   n:          400
#   resolution: 0.1
#   offset:     None
#   numTries:   0
#   name:       [0.1]

# c mysp: SpatialPooler
mysp = SpatialPooler(
    inputDimensions=[400,],
    columnDimensions=(1024,),
    globalInhibition=True,
    # @ Spatial Pooling: Learning (Episode 8), 4:00
    # https://youtu.be/rHvjykCIrZM?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=240
    numActiveColumnsPerInhArea=20,
    wrapAround=False)
# print("mysp",mysp)
# <nupic.algorithms.spatial_pooler.SpatialPooler object at 0x7f3dc8f23150>

# c dummy: placeholder for active column
dummy = np.zeros(mysp.getColumnDimensions()[0])
# print("dummy",dummy.shape)
# dummy (1024,)

for x in np.sin(np.linspace(0,100*np.pi,1000)):
    # c encoded_x: encode input data x
    encoded_x=encoder.encode(x)

    # Use spatial pooler with encoded_x, active column
    mysp.compute(encoded_x, True, dummy)

# c x: scalar input data
x = 0.7

# c d1: encoded x
d1 = encoder.encode(x)
# print("d1",d1.shape)
# d1 (400,)
# print("d1",d1)
# d1 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0
#  0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

# c d2: encoded x+0.1
d2 = encoder.encode(x+0.1)
# print("d2",d2.shape)
# d2 (400,)
# print("d2",d2)
# d2 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0
#  0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

# c encCols1: nonzero indices
encCols1 = np.nonzero(d1)[0]
# c encCols2: nonzero indices
encCols2 = np.nonzero(d2)[0]

num_col_1=mysp.getColumnDimensions()[0]
# print("num_col_1",num_col_1)
# num_col_1 1024
sdr1 = np.zeros(num_col_1)

num_col_2=mysp.getColumnDimensions()[0]
# print("num_col_2",num_col_2)
# num_col_2 1024
sdr2 = np.zeros(num_col_2)

# Use spatial pooler with encoded data and active column placeholder
mysp.compute(d1, False, sdr1)
mysp.compute(d2, False, sdr2)

sdrCols1 = np.nonzero(sdr1)[0]
sdrCols2 = np.nonzero(sdr2)[0]

fig, ax = plt.subplots(2, 1, figsize=(15,2))
ax[0].plot(range(400),d1,range(400),d2)
ax[1].plot(range(1024),sdr1,range(1024),sdr2)

# @ SDR Capacity & Comparison (Episode 2), 7:30
# https://youtu.be/ZDgCdWTuIzc?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=450
# About the overlap between 2 SDRs
for a in ax:
    a.set_xticks([])
print("Encoding Overlap = {}/{}".format(
    len(np.intersect1d(encCols1, encCols2)),
    len(encCols1)))
print("     SDR Overlap = {}/{}".format(
    len(np.intersect1d(sdrCols1, sdrCols2)), 
    len(sdrCols1)))
# plt.show()
# Encoding Overlap = 20/21
#      SDR Overlap = 20/20

def run(
    returnResults=False, 
    resolution=0.05, 
    historicWindowSize=2500, 
    slidingWindowSize=0):

    aScore = Anomaly(slidingWindowSize=slidingWindowSize)
    aLikely = AnomalyLikelihood(historicWindowSize=historicWindowSize)

    # Create our encoder, spatial pooler, and temporal memory
    encoder = RandomDistributedScalarEncoder(resolution=resolution)
    
    sp = SpatialPooler(
        inputDimensions = (encoder.getWidth(),),
        columnDimensions = (2048,),
        potentialRadius = encoder.getWidth(),
        potentialPct = 0.85,
        globalInhibition = True,
        localAreaDensity = -1.0,
        numActiveColumnsPerInhArea = 40.0,
        stimulusThreshold = 0,
        synPermInactiveDec = 0.005,
        synPermActiveInc = 0.04,
        synPermConnected = 0.1,
        minPctOverlapDutyCycle = 0.001,
        dutyCyclePeriod = 100,
        boostStrength = 3.0,
        seed = 42,
        spVerbosity = 0,
        wrapAround = False)
    
    tm = TemporalMemory(
        columnDimensions = sp.getColumnDimensions(),
        cellsPerColumn = 16,
        activationThreshold = 16,
        initialPermanence = 0.55,
        connectedPermanence = 0.5,
        minThreshold = 12,
        maxNewSynapseCount = 20,
        permanenceIncrement = 0.1,
        permanenceDecrement = 0.1,
        predictedSegmentDecrement = 0.0,
        seed = 42,
        maxSegmentsPerCell = 128,
        maxSynapsesPerSegment = 40)
    
    tm.predictedSegmentDecrement = sp.getNumActiveColumnsPerInhArea() / sp.getColumnDimensions()[0] * tm.permanenceIncrement

    # Let SP learn data it has to represent
    activeCols = np.zeros(sp.getColumnDimensions()[0])
    
    for s in swave[:1000]:
        sp.compute(encoder.encode(s), True, activeCols)
    
    scores = np.zeros(len(swave))
    hoods = np.zeros(len(swave))
    loghoods = np.zeros(len(swave))
    
    for i,v in enumerate(swave):
        if i % 20 == 0: # reset when sine wave starts again
            tm.reset()
        sp.compute(encoder.encode(v), False, activeCols)
        predColIndices = [tm.columnForCell(cell) for cell in tm.getPredictiveCells()]
        activeColIndices = np.nonzero(activeCols)[0]
        tm.compute(activeColIndices, learn=True)
        scores[i] = aScore.compute(activeColIndices, predColIndices)
        hoods[i] = aLikely.anomalyProbability(v, scores[i])
        loghoods[i] = aLikely.computeLogLikelihood(hoods[i])

    fig, ax = plt.subplots(3,1,figsize=(15,2*3))
    ax[0].plot(hoods,label="normal")
    ax[0].plot(loghoods, label="log")
    ax[1].plot(scores)
    ax[2].plot(swave)
    ax[0].set_ylabel('Anomaly\nLikelihoods')
    ax[1].set_ylabel('Anomaly\nScore')
    ax[2].set_ylabel('Sine wave')
    ax[0].set_title("res={}, histWS={}, anomSW={}".format(resolution, historicWindowSize, slidingWindowSize), fontsize=16)
    ax[0].grid()
    ax[0].legend()
    ax[1].legend()
    for ax in ax[:2]:
        ax.set_ylim((0,1.05))
    plt.show()
    if returnResults:
        return scores, hoods, loghoods

# --------------------------------------------------
# Now that we have our function ready to go, we can run bunch of different scenarios and see how anomaly detection changes. I chose to vary encoding resolution, historicWindowSize of AnomalyLikelihood class, and slidingWindowSize of Anomaly class.

# Results
# resolution: Looking at anomaly score as indication of how quickly TM learned sequence, it appears 0.01 caused TM to learn sequence fastest; with others, it took longer. The anomaly detection was thwarted by resolution of 1, as expected. For all other resolutions, anomalies were always detected (likelihood >= 0.5). The log likelihood was most pronounced for resolution of 0.1.

# historicWindowSize: This affected anomaly likelihoods as expected. Larger windows prolonged memory of anomaly (high likelihoods) while smaller windows forgot about anomalies faster.

# slidingWindowSize: This smoothed likelihoods nicely when set to 20 (which makes some sense since sequence is 20 points long).

# for r in (0.0001, 0.001, 0.01, 0.1, 1):
#     run(resolution=r, historicWindowSize=513, slidingWindowSize=20)

# for h in (1000, 800, 513, 500, 250, 113, 100):
#     run(resolution=0.01, historicWindowSize=h, slidingWindowSize=20)

# for w in (5, 10, 15, 20, 25, 30):
#     run(resolution=0.01, historicWindowSize=513, slidingWindowSize=w)

# ======================================================================
# Putting It All Together
# Now we'll combine all parts into one example. 
# We'll use favorite One Hot Gym data for NuPIC beginners.

# ======================================================================
_NUM_RECORDS = 4390

# ======================================================================
print("Creating encoders and encodings...")

# c timeOfDayEncoder: encoder for timeOfDay type input data
timeOfDayEncoder = DateEncoder(timeOfDay=(21,1))

# c weekendEncoder: encoder for weekend type input data
weekendEncoder = DateEncoder(weekend=21)

# c scalarEncoder: encoder for scalar type input data
# c scalarEncoder: but you encode scalar type input data 
# c scalarEncoder: as random distributed form
scalarEncoder = RandomDistributedScalarEncoder(resolution=0.5)

# c encodingWidth: width of entirely encoded data
encodingWidth = timeOfDayEncoder.getWidth() + \
                weekendEncoder.getWidth() + \
                scalarEncoder.getWidth()
# print("encodingWidth",encodingWidth)
# encodingWidth 946

# c timeOfDayBits: array for timeOfDayEncoder
timeOfDayBits = np.zeros(timeOfDayEncoder.getWidth())
# c weekendBits: array for weekendEncoder
weekendBits = np.zeros(weekendEncoder.getWidth())
# c consumptionBits: array for scalarEncoder
consumptionBits = np.zeros(scalarEncoder.getWidth())

# c activeColumns: placeholder for active columns
activeColumns = np.zeros(2048)

# ======================================================================
print("Initializing Spatial Pooler...")
sp = SpatialPooler(
    inputDimensions = (encodingWidth,),
    columnDimensions= (2048,),
    # global coverage
    potentialRadius = encodingWidth, 
    potentialPct = 0.85,
    # @ Spatial Pooling: Learning (Episode 8), 4:10
    # https://youtu.be/rHvjykCIrZM?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=250
    globalInhibition = True,
    localAreaDensity = -1.0,
    # 2% sparsity
    numActiveColumnsPerInhArea = 40.0, 
    stimulusThreshold = 0,
    synPermInactiveDec = 0.005,
    synPermActiveInc = 0.04,
    synPermConnected = 0.1,
    minPctOverlapDutyCycle = 0.001,
    dutyCyclePeriod = 100,
    # @ Boosting (Episode 9), 1:00
    # https://youtu.be/MSwoNAODrgk?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=60 
    # In order for column in spatial pooler
    # to express itself, that column must be selected as winning column
    # Top most columns with the highest overlaps with input space
    # being selected as winning columns
    # ...
    # @ Boosting (Episode 9), 3:54
    # https://youtu.be/MSwoNAODrgk?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=234
    # How aggressive boosting will you use for homeostasis?
    # @ Boosting (Episode 9), 7:46
    # https://youtu.be/MSwoNAODrgk?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=466
    # Advantage of using boosting
    boostStrength = 3.0, 
    seed = 42,
    spVerbosity = 0,
    wrapAround = False)

# ======================================================================
print("Letting spatial pooler learn dataspace...")

# c n_train_for_SP: number of train for SP
n_train_for_SP = 3000
with open(_INPUT_FILE_PATH, 'r') as fin:
    reader = csv.reader(fin)
    reader.next() # ignore headers
    reader.next()
    reader.next()

    # print("reader",reader)
    # reader <_csv.reader object at 0x7fa7d77497c0>
    for count, record in enumerate(reader):
        # print("count",count)
        # count 0

        # print("record",record)
        # record ['7/2/10 0:00', '21.2']
        
        if count >= n_train_for_SP:
            break

        # c dateString: date input data
        dateString = dt.strptime(record[0], "%m/%d/%y %H:%M")

        # c consumption: consumption level scalar input data
        consumption = float(record[1])

        # Encode dateString data into array 
        # by using timeOfDayBits placeholder
        timeOfDayEncoder.encodeIntoArray(dateString, timeOfDayBits)

        # Encode dateString data into array 
        # by using weekendBits placeholder
        weekendEncoder.encodeIntoArray(dateString, weekendBits)

        # Encode consumption data into array 
        # by using consumptionBits placeholder
        scalarEncoder.encodeIntoArray(consumption, consumptionBits)

        # c encoding: entirely encoded "final input data"
        encoding = np.concatenate(
            [timeOfDayBits, weekendBits, consumptionBits])

        # Use spatial pooler
        # by using encoding and activeColumns
        sp.compute(encoding, True, activeColumns)

# ======================================================================
# @ Boosting (Episode 9), 1:00
# https://youtu.be/MSwoNAODrgk?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=60
# You turn off boosting
print("...all done. Turning off boosting")
sp.setBoostStrength(0.0)

# ======================================================================
# print('sp.getColumnDimensions()',sp.getColumnDimensions())
# sp.getColumnDimensions() [2048]

print("Initializing temporal memory learning algorithm...")
tm = TemporalMemory(
    # @ Spatial Pooling: Input Space & Connections (Episode 7), 10:18
    # https://youtu.be/R5UoFNtv5AU?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=618
    # sp.getColumnDimensions() probably means length of 1D array of spatial pooler
    # See "Spatial Pooler Columns"
    # And number of cells in it is ColumnDimensions of spatial pooler
    columnDimensions = sp.getColumnDimensions(),
    # @ Temporal Memory Part 1 (Episode 11), 5:04
    # https://youtu.be/UBzemKcUoOk?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=304
    # In video screenshot, there are 4 cells in per column
    # I guess this option with 16 will create 16 cells per column
    cellsPerColumn = 16,
    activationThreshold = 13,
    initialPermanence = 0.55,
    connectedPermanence = 0.5,
    minThreshold = 10,
    maxNewSynapseCount = 20,
    permanenceIncrement = 0.1,
    permanenceDecrement = 0.1,
    predictedSegmentDecrement = 0.0,
    seed = 42,
    # @ Temporal Memory Part 1 (Episode 11), 11:02
    # https://youtu.be/UBzemKcUoOk?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=662
    # Number of max "kind" of connections (like magenta colored lines) per cell (blue one)
    # In this case, there is one kind of connection (magenta color) to that blue cell
    maxSegmentsPerCell = 128,
    # @ Temporal Memory Part 1 (Episode 11), 10:56
    # https://youtu.be/UBzemKcUoOk?list=PL3yXMgtrZmDqhsFQzwUC9V8MeeVOQ7eZ9&t=656
    # Number of max connections (which each kind of synapse can have) per each kind of segment
    maxSynapsesPerSegment = 40)

# ======================================================================
print("Initializing classification and anomaly calculators")
# c classifier: SDRClassifier
classifier = SDRClassifierFactory.create(steps=[1], alpha=0.01)

# c predictions: placeholder for prediction
predictions = np.zeros(_NUM_RECORDS+2)

# c aScore: abnomaly score
aScore = Anomaly(slidingWindowSize=25)

# c aScore: abnomaly score in log likelyhood
aLikely = AnomalyLikelihood(learningPeriod=600, historicWindowSize=313)

ascores = np.zeros(_NUM_RECORDS+1)
alhoods = np.zeros(_NUM_RECORDS+1)
alloghoods = np.zeros(_NUM_RECORDS+1)

with open(_INPUT_FILE_PATH, "r") as fin:
    reader = csv.reader(fin)
    reader.next()
    reader.next()
    reader.next()

    print("Beginning record processing...")
    for count, record in enumerate(reader):
        # print("count",count)
        # count 0

        # print("record",record)
        # record ['7/2/10 0:00', '21.2']

        if count > _NUM_RECORDS: 
            break

        if count % 500 == 0:
            print("Processed {0:4d}/{1} records".format(
                count, _NUM_RECORDS))

        dateString = dt.strptime(record[0], "%m/%d/%y %H:%M")

        consumption = float(record[1])

        timeOfDayEncoder.encodeIntoArray(dateString, timeOfDayBits)
        
        weekendEncoder.encodeIntoArray(dateString, weekendBits)
        
        scalarEncoder.encodeIntoArray(consumption, consumptionBits)
        
        encoding = np.concatenate(
            [timeOfDayBits, weekendBits, consumptionBits])

        sp.compute(encoding, False, activeColumns)

        # c activeColumnIndices: active column indices
        activeColumnIndices = np.nonzero(activeColumns)[0]

        predictedColumnIndices=[]
        for cell in tm.getPredictiveCells():
            col_for_cell=tm.columnForCell(cell)
            predictedColumnIndices.append(col_for_cell)
        
        # Perform temporal learning algorithm
        tm.compute(activeColumnIndices, learn=True)

        ascores[count] = aScore.compute(
            activeColumnIndices, predictedColumnIndices)

        alhoods[count] = aLikely.anomalyProbability(
            consumption, ascores[count])

        alloghoods[count] = aLikely.computeLogLikelihood(alhoods[count])

        bucketIdx = scalarEncoder.getBucketIndices(consumption)[0]

        classifierResult = classifier.compute(
            recordNum=count, 
            patternNZ=tm.getActiveCells(),
            classification={
                "bucketIdx": bucketIdx,
                "actValue": consumption},
            # Let classifier learn once TM has learned little
            learn=count > 500, 
            infer=True)

        zipped=zip(classifierResult[1], 
                   classifierResult["actualValues"])
        
        predConf,predictions[count+1]=sorted(zipped,reverse=True)[0]

consumption = np.loadtxt(
    _INPUT_FILE_PATH,
    delimiter=',',
    skiprows=3,
    usecols=1)
# print("consumption",consumption)
# [ 21.2  16.4   4.7 ...,   5.3   5.1   5. ]

# ======================================================================
# Alright, let's see what we got. 

# But before we do, 
# let's look at "data" 
# and see if we can pick out any anomalies.

plt.figure(figsize=(15,3))
plt.plot(consumption)
plt.xlim(0, 4400)
plt.xticks(range(0,4401,250))
plt.show()

# So, if I had to guess, 
# I'd say 3 things pop out as potential anomalies. 

# First, consumption doesn't return to its normal baseline 
# somewhere between 1800 and 2550. 

# Second, there seems to be uncharacteristic increase in consumption around 3250. 

# Finally, there's strange dip near 4250. 

# Let's see what our code found.

# ======================================================================
possible_anomaly_indices = np.where(alloghoods >= 0.5)[0]
y = np.array([100,80,100]*len(possible_anomaly_indices))
possible_anomaly_indices = np.sort(np.concatenate((possible_anomaly_indices, possible_anomaly_indices, possible_anomaly_indices)))
fig, ax = plt.subplots(3,1,figsize=(15,10))
ax[0].plot(alhoods, label='normal')
ax[0].plot(alloghoods, label='log')
ax[1].plot(ascores)
ax[2].plot(consumption, label='actual')
ax[2].plot(predictions, label='predicted')
ax[2].plot(possible_anomaly_indices, y, label='anomalies?')
ax[0].grid()
for a in ax:
    a.set_xlim((0,4400))
    a.set_xticks(range(0,4401,250))
ax[0].set_ylim((0,1))
ax[1].set_ylim((0,1))
ax[2].set_ylim((0,100))
ax[0].legend()
ax[2].legend()
ax[0].set_ylabel("Anomaly\nLikelihoods")
ax[1].set_ylabel("Anomaly\nScores")
ax[2].set_ylabel("Energy\nConsumption")
plt.show()

# So our code found 7 anomalies, 
# but only 3 of them correspond to our predictions. 

# Let's zoom in on some of areas 
# and see what actual vs predicted consumptions look like.

# ======================================================================
def zoom(N, w=150):
    plt.figure(figsize=(15,3))
    plt.plot(consumption[N-w:N+w], label="actual")
    plt.plot(predictions[N-w:N+w], label="predicted")
    plt.plot([w, w], [0, 100], label="anomaly")
    plt.xlim((0,w))
    plt.xticks(range(0,301,10))
    plt.title("Centered at anomaly index {}".format(N))
    plt.show()

for i in np.unique(possible_anomaly_indices):
    zoom(i)

# ======================================================================
# Create aforementioned curves
hour = dict.fromkeys(range(24))
counts = np.zeros(24,dtype=int)

for h in range(24):
    hour[h] = np.zeros(183)

with open(_INPUT_FILE_PATH,'r') as fin:
    reader = csv.reader(fin)
    reader.next()
    reader.next()
    reader.next()
    for rec in reader:
        h = int(rec[0].split()[1].split(':')[0])
        hour[h][counts[h]] = float(rec[1])
        counts[h] += 1

ind = np.round(np.unique(possible_anomaly_indices)/24)
y = [0,100,0]*len(ind)
ind = np.sort(np.concatenate((ind, ind, ind)))
fig, ax = plt.subplots(24,1,figsize=(15,20))
for i in range(24):
    ax[i].plot(hour[i])
    ax[i].plot(ind, y)
    ax[i].set_xlim((0,184))
    ax[i].set_ylim((0,100))
    ax[i].set_ylabel("{}:00".format(i), rotation=0, labelpad=20)
    ax[i].yaxis.set_label_position("right")
    ax[i].tick_params(axis='x',which='both',bottom=False,labelbottom=False)
plt.show()
