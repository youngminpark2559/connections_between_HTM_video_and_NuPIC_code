# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

# ======================================================================
import csv
import json
from datetime import datetime

from pkg_resources import resource_filename

from nupic.engine import Network
from nupic.encoders import DateEncoder

# ======================================================================
def createNetwork():
  # c network: create Network instance
  network = Network()

  # --------------------------------------------------
  # Add sensors to network

  # c param_f_consumptionSensor: parameter for consumptionSensor
  param_f_consumptionSensor={
    'n': 120,
    'w': 21,
    'minValue': 0.0,
    'maxValue': 100.0,
    'clipInput': True}

  # c jparam_f_cs: json param for consumptionSensor
  jparam_f_cs=json.dumps(param_f_consumptionSensor)

  # C++
  # c consumptionSensor: add consumptionSensor region into network
  consumptionSensor = network.addRegion(
    'consumptionSensor', 'ScalarSensor', jparam_f_cs)

  # --------------------------------------------------
  # Python
  # c timestampSensor: add timestampSensor region into network
  timestampSensor = network.addRegion(
    "timestampSensor",'py.PluggableEncoderSensor', "")

  # c date_encoder: create date encoder
  date_encoder=DateEncoder(timeOfDay=(21, 9.5), name="timestamp_timeOfDay")

  # c date_encoder: assing date encoder into timestampSensor
  timestampSensor.getSelf().encoder = date_encoder

  # --------------------------------------------------
  # c consumptionEncoderN: get number of bits "n" from consumptionSensor
  consumptionEncoderN = consumptionSensor.getParameter('n')
  # print("consumptionEncoderN",consumptionEncoderN)
  # ('consumptionEncoderN', 120)

  # print("timestampSensor.getSelf()",timestampSensor.getSelf())
  # <nupic.regions.pluggable_encoder_sensor.PluggableEncoderSensor object at 0x7fa428bf31d0>

  # c encoder_of_tss: encoder of timestampSensor
  encoder_of_tss=timestampSensor.getSelf().encoder

  # c timestampEncoderN: width of encoder of timestampSensor
  timestampEncoderN = encoder_of_tss.getWidth()
  # print("timestampEncoderN",timestampEncoderN)
  # ('timestampEncoderN', 54)

  # c inputWidth: width of input
  inputWidth = consumptionEncoderN + timestampEncoderN
  # print("inputWidth",inputWidth)
  # ('inputWidth', 174)

  # --------------------------------------------------
  # c param_f_SP: parameter for spatial pooler
  param_f_SP={
    # c spatialImp: spatial pooler implementation in C++
    "spatialImp": "cpp",
    # c globalInhibition: 1 -> on
    "globalInhibition": 1,
    "columnCount": 2048,
    "inputWidth": inputWidth,
    # c numActiveColumnsPerInhArea: number of active columns per inhibition area
    "numActiveColumnsPerInhArea": 40,
    "seed": 1956,
    # c potentialPct: potential pool percent
    "potentialPct": 0.8,
    # c "synPermConnected: synaptic permanence connected
    "synPermConnected": 0.1,
    # c synPermActiveInc: synaptic permanence active increment
    "synPermActiveInc": 0.0001,
    # c synPermInactiveDec: synaptic permanence inactive decrement
    "synPermInactiveDec": 0.0005,
    "boostStrength": 0.0,}

  # c param_f_SP_j: parameter for spatial pooler in JSON
  param_f_SP_j=json.dumps(param_f_SP)

  # c Add "SPRegion" into network
  # SPRegion can contain spatial pooler
  network.addRegion("sp", "py.SPRegion", param_f_SP_j)

  # --------------------------------------------------
  # Link each configured one in network
  network.link("consumptionSensor", "sp", "UniformLink", "")
  network.link("timestampSensor", "sp", "UniformLink", "")

  # --------------------------------------------------
  # c param_f_TM: parameter for temporal memory learning algorithm
  param_f_TM={
    "columnCount": 2048,
    "cellsPerColumn": 32,
    "inputWidth": 2048,
    "seed": 1960,
    "temporalImp": "cpp",
    "newSynapseCount": 20,
    # c maxSynapsesPerSegment: maximum number of synapses per segment
    "maxSynapsesPerSegment": 32,
    # c maxSegmentsPerCell: maximum number of segments per cell
    "maxSegmentsPerCell": 128,
    # c initialPerm: initial permanence value for newly created synapses
    "initialPerm": 0.21,
    # c permanenceInc: active synapses get their permanence counts incremented by this value
    "permanenceInc": 0.1,
    # c permanenceDec: all other synapses get their permanence counts decremented by this value
    "permanenceDec": 0.1,
    "globalDecay": 0.0,
    "maxAge": 0,
    "minThreshold": 9,
    # c activationThreshold: if "number of active connected synapses" on segment is 
    # c activationThreshold: at least this threshold, segment is said to be active
    "activationThreshold": 12,
    "outputType": "normal",
    "pamLength": 3,}
  
  # c param_f_TM_j: parameter for temporal memory learning algorithm in JSON
  param_f_TM_j=json.dumps(param_f_TM)

  # Add TMRegion into network 
  # TMRegion is region containing "Temporal Memory Learning algorithm"
  network.addRegion("tm", "py.TMRegion", param_f_TM_j)

  # --------------------------------------------------
  network.link("sp", "tm", "UniformLink", "")
  network.link("tm", "sp", "UniformLink", "", srcOutput="topDownOut", destInput="topDownIn")

  # --------------------------------------------------
  # Enable anomalyMode so TM calculates anomaly scores
  network.regions['tm'].setParameter("anomalyMode", True)
  
  # Enable inference mode to be able to get predictions
  network.regions['tm'].setParameter("inferenceMode", True)

  return network

def runNetwork(network):
  # c consumptionSensor: get consumptionSensor from regions of network
  consumptionSensor = network.regions['consumptionSensor']
  
  # c timestampSensor: get timestampSensor from regions of network
  timestampSensor = network.regions['timestampSensor']
  
  # c tmRegion: get TM-region from regions of network
  tmRegion = network.regions['tm']

  # --------------------------------------------------
  # c filename: configure file name you're going to use
  filename = resource_filename("nupic.datafiles", "extra/hotgym/rec-center-hourly.csv")

  # c csvReader: get data from CSV file
  csvReader = csv.reader(open(filename, 'r'))
  csvReader.next()
  csvReader.next()
  csvReader.next()

  for row in csvReader:
    # c timestampStr: timestamp in string
    # c consumptionStr: consumption value in string
    timestampStr, consumptionStr = row
    # print("timestampStr",timestampStr)
    # '7/2/10 0:00'
    # print("consumptionStr",consumptionStr)
    # '21.2'

    # --------------------------------------------------
    # c give consumptionStr to consumptionSensor
    consumptionSensor.setParameter('sensedValue', float(consumptionStr))

    # c t: convert timestampStr into datetime object
    t = datetime.strptime(timestampStr, "%m/%d/%y %H:%M")
    # c give t input data to timestampSensor
    timestampSensor.getSelf().setSensedValue(t)

    # --------------------------------------------------
    # c run network
    network.run(1)

    # --------------------------------------------------
    # c anomalyScore: get anomaly score from tmRegion
    anomalyScore = tmRegion.getOutputData('anomalyScore')[0]
    print "Consumption: %s, Anomaly score: %f" % (consumptionStr, anomalyScore)

if __name__ == "__main__":
  network = createNetwork()
  runNetwork(network)
