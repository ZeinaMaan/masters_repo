# Dependent Libraries

import Lib

# Importing Utility Libraries

import numpy as np
import tensorflow as tf
import cv2
import math
import time
import random
import msvcrt
import argparse
import os
from matplotlib import pyplot as plt
import shutil

# Importing XBee Libraries

from digi.xbee.devices import XBeeDevice
from digi.xbee.devices import RemoteXBeeDevice
from digi.xbee.devices import XBee64BitAddress

# Importing Object Detection Libraries

from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util

# Defining test configuration

test = False

# Defining risk density option and parameters

option = 3
alpha = 1
beta = 0.0001

shutil.rmtree('data/option{}/frames'.format(option))
shutil.rmtree('data/option{}/plots'.format(option))

os.mkdir('data/option{}/frames'.format(option))
os.mkdir('data/option{}/plots'.format(option))

# Defining Important Paths

PATH_TO_FROZEN_GRAPH = 'frozen_inference_graph.pb'
PATH_TO_LABELS = 'label_map.pbtxt'

# Pre-initialization variable assignment

iterator = 0

if not test:

    # Define Controller Device

    device = XBeeDevice("COM4", 9600)

    # Open Controller Device

    device.open()

    # Define Remote Devices

    remote_devices = []

    remote_devices.append(RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string("0013A20040D835D1")))
    remote_devices.append(RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string("0013A200418FE7A3")))
    remote_devices.append(RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string("0013A200415E8441")))
    remote_devices.append(RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string("0013A2004190B158")))
    remote_devices.append(RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string("0013A200417CA46A")))

else:
    remote_devices = [[0], [1]]

# Initialization variable assignment

positionInit = [] # Initialization position variable
changePosition = []
distanceInit = [] # Initialization distance variable
idOrder = [] # For matching XBee id to object detection id
orientationInit = [] # Initialization orientation variable

# Run loop variable assignment

pass_var = False
agent = 0
iterations = 0
complete = 0
frame_count = 0

edgeTimer = []
timer = []

isSuccess = [] # Indicates if desired position has been achieved
isEdge = [] # Indicates if agent is near an edge
reset = []
distanceDes = [] # Real time distance between agent and desired position

currentPosition = [] # Current position of agent
prevPosition = [] # Previous position of agent
timestepDistance = [] # Distance between current position and previous position of agent

orientation = [] # Current orientation of each agent
orientationDes = [] # Current orientation of each agent
angle = [] # Angle from current orientation, to optimal orientation
messageArray = [] # The message to be sent to each agent

for i in range(len(remote_devices)):

    # Initialization variable assignment

    positionInit.append(None) # Initialization position variable
    changePosition.append(None)
    distanceInit.append(0.0) # Initialization distance variable
    idOrder.append(None) # For matching XBee id to object detection id
    orientationInit.append(None) # Initialization orientation variable

    # Run loop variable assignment

    edgeTimer.append(time.time())
    timer.append(time.time())

    isSuccess.append(False) # Indicates if desired position has been achieved
    isEdge.append(False) # Indicates if agent is near an edge
    reset.append(True)
    distanceDes.append(None) # Real time distance between agent and desired position

    currentPosition.append(None) # Current position of agent
    prevPosition.append(None) # Previous position of agent
    timestepDistance.append(None) # Distance between current position and previous position of agent

    orientation.append(None) # Current orientation of each agent
    orientationDes.append(None) # Current orientation of each agent
    angle.append(None) # Angle from current orientation, to optimal orientation
    messageArray.append(None) # The message to be sent to each agent

# Define Centroid Tracker

tracker = Lib.Tracker()

# Define Voronoi Library

voronoi = Lib.Voronoi()

# Define Frozen Graph From Path

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Define Label Map From Path

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Start Video Capture

if test:
    vidcap = cv2.VideoCapture("TestTrim.mp4")
else:
    vidcap = cv2.VideoCapture(1)

success, frame = vidcap.read()

width_sections = 64
height_sections = 48

xdim = np.linspace(0, frame.shape[1], width_sections)
ydim = np.linspace(0, frame.shape[0], height_sections)
X, Y = np.meshgrid(xdim,ydim)

# Define map of risk density based on selected option

if option == 1:
    risk = np.ones((height_sections, width_sections))

elif option == 2:
    target = [frame.shape[1]/2, frame.shape[0]/2]
    risk = voronoi.risk_density(xdim, ydim, target, alpha=alpha, beta=beta)
    print(target)

elif option == 3:
    #target = [random.random()*frame.shape[1], random.random()*frame.shape[0]]
    target = [485, 320]
    risk = voronoi.risk_density(xdim, ydim, target, alpha=alpha, beta=beta)
    print(target)

if not test:

    # Start Pre-Initialization loop

    while True:
        success, frame = vidcap.read()

        if (success):
            cv2.imshow("Frame", frame)
            cv2.waitKey(1)

        if msvcrt.kbhit():

            key = msvcrt.getch()
            print(key)

            if key == b'w':
                message = "P,F>"

            if key == b'a':
                message = "P,90>"

            if key == b's':
                message = "P,B>"

            if key == b'd':
                message = "P,270>"

            if key == b'z':
                message = "P,S>"

            if key == b'q':
                iterator = iterator - 1
                message = "P,S>"

                if (iterator < 0):
                    iterator = 0

            if key == b'e':
                iterator = iterator + 1
                message = "P,S>"

            if key == b'x':
                break

            if (iterator > (len(remote_devices) - 1)):
                break

            device.send_data_async(remote_devices[iterator], message)
            print(message)

# Start Object Detection

with detection_graph.as_default() as graph:
    with tf.compat.v1.Session() as sess:

            if not test:

    # Start Initialization loop

                while True:

                    for i in range(0, len(remote_devices)):

                          success, frame = vidcap.read()

                          frame_expanded = np.expand_dims(frame, axis=0)

                          output_dict = Lib.run_inference_for_single_image(frame_expanded, graph, sess)

                          rects = []

                          for j in range(0, len(remote_devices)):
                            if output_dict['detection_scores'][j] > 0.85:

                                  (startY, startX, endY, endX) = output_dict['detection_boxes'][j]

                                  rects.append([int(startX*frame.shape[1]), int(startY*frame.shape[0]), int(endX*frame.shape[1]), int(endY*frame.shape[0])])

                                  cv2.rectangle(frame, (int(startX*frame.shape[1]), int(startY*frame.shape[0])), (int(endX*frame.shape[1]), int(endY*frame.shape[0])),
                                     (0, 255, 0), 1)

                          objects = tracker.update(rects)

                          for (objectID, centroid) in objects.items():
                              text = "ID {}".format(objectID)
                              cv2.putText(frame, text, (centroid[0] - 15, centroid[1] - 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                              cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                              positionInit[objectID] = centroid

                          cv2.imshow("Frame", frame)
                          cv2.waitKey(1)

                          device.send_data_async(remote_devices[i], "P,F>")

                          for j in range(0, 35):
                              success, frame = vidcap.read()

                              frame_expanded = np.expand_dims(frame, axis=0)

                              output_dict = Lib.run_inference_for_single_image(frame_expanded, graph, sess)

                              rects = []

                              for k in range(0, len(remote_devices)):
                                if output_dict['detection_scores'][k] > 0.85:

                                      (startY, startX, endY, endX) = output_dict['detection_boxes'][k]

                                      rects.append([int(startX*frame.shape[1]), int(startY*frame.shape[0]), int(endX*frame.shape[1]), int(endY*frame.shape[0])])

                                      cv2.rectangle(frame, (int(startX*frame.shape[1]), int(startY*frame.shape[0])), (int(endX*frame.shape[1]), int(endY*frame.shape[0])),
                                         (0, 255, 0), 1)

                              objects = tracker.update(rects)

                              for (objectID, centroid) in objects.items():
                                  text = "ID {}".format(objectID)
                                  cv2.putText(frame, text, (centroid[0] - 15, centroid[1] - 25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                  cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                                  changePosition[objectID] = centroid
                                  distanceInit[objectID] = Lib.get_distance(point1=positionInit[objectID], point2=changePosition[objectID])
                                  #print(distanceInit[objectID])

                              cv2.imshow("Frame", frame)
                              cv2.waitKey(1)

                          idOrder[i] = np.argmax(distanceInit)

                          device.send_data_async(remote_devices[i], "P,B>")

                          for j in range(0, 35):
                              success, frame = vidcap.read()

                              frame_expanded = np.expand_dims(frame, axis=0)

                              output_dict = Lib.run_inference_for_single_image(frame_expanded, graph, sess)

                              rects = []

                              for k in range(0, len(remote_devices)):
                                if output_dict['detection_scores'][k] > 0.85:

                                      (startY, startX, endY, endX) = output_dict['detection_boxes'][k]

                                      rects.append([int(startX*frame.shape[1]), int(startY*frame.shape[0]), int(endX*frame.shape[1]), int(endY*frame.shape[0])])

                                      cv2.rectangle(frame, (int(startX*frame.shape[1]), int(startY*frame.shape[0])), (int(endX*frame.shape[1]), int(endY*frame.shape[0])),
                                         (0, 255, 0), 1)

                              objects = tracker.update(rects)

                              for (objectID, centroid) in objects.items():
                                  text = "ID {}".format(objectID)
                                  cv2.putText(frame, text, (centroid[0] - 15, centroid[1] - 25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                  cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                                  positionInit[objectID] = centroid

                              cv2.imshow("Frame", frame)
                              cv2.waitKey(1)

                          orientationInit[idOrder[i]] = Lib.get_orientation(point1=positionInit[idOrder[i]], point2=changePosition[idOrder[i]])

                          device.send_data_async(remote_devices[i], "I," + str(orientationInit[idOrder[i]]) + ">")


    # Check User Input

                    print(idOrder)

                    test1 = Lib.test_1(idArray=idOrder, numRobots=len(remote_devices))

                    test2 = Lib.test_2(idArray=idOrder, numRobots=len(remote_devices))

                    if test1:
                        print("Test 1 passed.")

                    if not test1:
                        print("Test 1 failed.")

                    if test2:
                        print("Test 2 passed.")

                    if  not test2:
                        print("Test 2 failed.")

                    success, frame = vidcap.read()

                    frame_expanded = np.expand_dims(frame, axis=0)

                    output_dict = Lib.run_inference_for_single_image(frame_expanded, graph, sess)

                    rects = []

                    for i in range(0, len(remote_devices)):
                        if output_dict['detection_scores'][i] > 0.85:

                            (startY, startX, endY, endX) = output_dict['detection_boxes'][i]

                            rects.append([int(startX*frame.shape[1]), int(startY*frame.shape[0]), int(endX*frame.shape[1]), int(endY*frame.shape[0])])

                            cv2.rectangle(frame, (int(startX*frame.shape[1]), int(startY*frame.shape[0])), (int(endX*frame.shape[1]), int(endY*frame.shape[0])),
                                         (0, 255, 0), 1)

                    objects = tracker.update(rects)

                    for (objectID, centroid) in objects.items():
                            text = "ID {}".format(objectID)
                            cv2.putText(frame, text, (centroid[0] - 15, centroid[1] - 25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(frame, str(orientationInit[objectID]), (centroid[0] - 15, centroid[1] - 50),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                            cv2.arrowedLine(frame, (centroid[0], centroid[1]), (changePosition[objectID][0], changePosition[objectID][1]), (255, 0, 0), 1)
                            cv2.line(frame, (centroid[0], centroid[1]), (centroid[0] + 30, centroid[1]), (255, 0, 0), 1)

                    cv2.imshow("Frame", frame)
                    cv2.waitKey(1)

                    inputVar = input("Are you satisfied with the initialization? (y/n)")

                    if (inputVar == 'y'):
                        break

                    if cv2.waitKey(1) & 0xFF == ord("x"):
                        break

# Start Run loop

            idOrder_array = np.array(idOrder)

            data_txt = open("data/option{}/data.txt".format(option), "w")

            while True:

              success, frame = vidcap.read()

              frame_expanded = np.expand_dims(frame, axis=0)

              output_dict = Lib.run_inference_for_single_image(frame_expanded, graph, sess)

              rects = []

              for i in range(0, len(remote_devices)):
                if output_dict['detection_scores'][i] > 0.85:

                      (startY, startX, endY, endX) = output_dict['detection_boxes'][i]

                      rects.append([int(startX*frame.shape[1]), int(startY*frame.shape[0]), int(endX*frame.shape[1]), int(endY*frame.shape[0])])

                      cv2.rectangle(frame, (int(startX*frame.shape[1]), int(startY*frame.shape[0])), (int(endX*frame.shape[1]), int(endY*frame.shape[0])),
                         (0, 255, 0), 2)

              objects = tracker.update(rects)

              agentPosition = []

              for (objectID, centroid) in objects.items():
                  text = "ID {}".format(objectID)
                  cv2.putText(frame, text, (centroid[0] - 15, centroid[1] - 25),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                  cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                  agentPosition.append([centroid[0], centroid[1]])

# Get Desired Location of Each Robot

              distance_update, voronoi_update, index_update = voronoi.voronoi(agentPosition=agentPosition, numAgents=len(remote_devices), xdim=xdim, ydim=ydim)

              if np.sum(isSuccess) == len(remote_devices) or np.sum(reset) == len(remote_devices):

                  iterations = iterations + 1

                  desPosition = []

                  for n in range(len(remote_devices)):
                      desPosition.append(voronoi.centroid(voronoi_update[n], index_update[n], risk))
                      isSuccess[n] = False

                  agent = 0

              coverage = 0

              for i in range(len(remote_devices)):
                  cv2.circle(frame, (int(desPosition[i][0]), int(desPosition[i][1])), 4, (255, 0, 0), -1)
                  cv2.circle(frame, (int(desPosition[i][0]), int(desPosition[i][1])), 30, (255, 0, 0), 1)
                  coverage += voronoi.coverage_metric(agentPosition[i], voronoi_update[i], index_update[i], risk)

              if np.sum(reset) == len(remote_devices):
                  start_time = time.time()
                  data_txt.write("Time: {}, Coverage Metric: {}, Iteration: {}, Agent Positions: {}, Desired Positions: {}".format(0, coverage, iterations, agentPosition, desPosition))
              else:
                  data_txt.write("Time: {}, Coverage Metric: {}, Iteration: {}, Agent Positions: {}, Desired Positions: {}".format(time.time() - start_time, coverage, iterations, agentPosition, desPosition))
              data_txt.write("\n")

# Check Distance From Desired Location

              distanceDes[agent] = Lib.get_distance(point1=desPosition[agent], point2=agentPosition[agent])

              if (distanceDes[agent] > 30.0):
                        isSuccess[agent] = False

            # if (distanceDes[objectID] <= 30.0 and isSuccess[objectID] == True):
            #           complete = complete + 1

              if (distanceDes[agent] <= 30.0 and isSuccess[agent] == False):
                        if (reset[agent] == True) and not test:
                            index = np.where(idOrder_array==agent)
                            device.send_data_async(remote_devices[index[0][0]], "A," + str(orientationInit[agent]) + ">")
                        if (reset[agent] == False) and not test:
                            index = np.where(idOrder_array==agent)
                            device.send_data_async(remote_devices[index[0][0]], "A," + str(orientation[agent]) + ">")

                        isSuccess[agent] = True
                        agent = agent + 1
                        if agent > (len(remote_devices) - 1):
                            agent = len(remote_devices) - 1

                        pass_var = True


# Check Edge Conditions

              # if (((time.time() - edgeTimer[agent]) >= 5) and ((agentPosition[agent][0] <= frame.shape[1]*0.01) or (agentPosition[agent][0] >= frame.shape[1]*0.99) or (agentPosition[agent][1] <= frame.shape[0]*0.01) or (agentPosition[agent][1] >= frame.shape[0]*0.99))):
              #     isEdge[agent] = True
              #     timer[agent] = time.time() + 5
              #     edgeTimer[agent] = time.time()
              #
              #     currentPosition[agent] = agentPosition[agent]
              #     orientationDes[agent] = Lib.get_orientation(currentPosition[agent], desPosition[agent])
              #
              #     if (agentPosition[agent][0] <= frame.shape[1]*0.01) and not test:
              #         device.send_data_async(remote_devices[idOrder_array[agent]], "E,L," + str(orientationDes[agent]) + ">")
              #
              #     if (agentPosition[agent][0] >= frame.shape[1]*0.99) and not test:
              #         device.send_data_async(remote_devices[idOrder_array[agent]], "E,R," + str(orientationDes[agent]) + ">")
              #
              #     if (agentPosition[agent][1] <= frame.shape[0]*0.01) and not test and not (agentPosition[agent][0] <= frame.shape[1]*0.01 or agentPosition[agent][0] >= frame.shape[1]*0.99):
              #         device.send_data_async(remote_devices[idOrder_array[agent]], "E,U," + str(orientationDes[agent]) + ">")
              #
              #     if (agentPosition[agent][1] >= frame.shape[0]*0.99) and not test and not (agentPosition[agent][0] <= frame.shape[1]*0.01 or agentPosition[agent][0] >= frame.shape[1]*0.99):
              #         device.send_data_async(remote_devices[idOrder_array[agent]], "E,D," + str(orientationDes[agent]) + ">")

# Check Time Conditions

              if ((((time.time() - timer[agent]) >= 5) or (reset[agent])) and not isSuccess[agent] and not isEdge[agent] and not pass_var):

# Determine Orientation and Position of Each Robot

                    timer[agent] = time.time()

                    if (reset[agent] == False):

                            messageArray[agent] = "F"

                            timestepDistance[agent] = Lib.get_distance(currentPosition[agent], agentPosition[agent])
                            #print(timestepDistance)
                            if (timestepDistance[agent] > 5):
                                prevPosition[agent] = currentPosition[agent]
                                currentPosition[agent] = agentPosition[agent]
                                orientation[agent] = Lib.get_orientation(prevPosition[agent], currentPosition[agent])
                                orientationDes[agent] = Lib.get_orientation(currentPosition[agent], desPosition[agent])
                                angle[agent] = Lib.get_angle(orientation[agent], orientationDes[agent])
                                messageArray[agent] = Lib.angle_to_message(angle[agent])

                            if not test:
                                index = np.where(idOrder_array==agent)
                                device.send_data_async(remote_devices[index[0][0]], "R," + messageArray[agent] + ">")
                            print("Object ID: " + str(agent) + " Orientation: " + str(orientation[agent])[:4] + " Desired Orientation: " + str(orientationDes[agent])[:4] + " Message Sent: "+ messageArray[agent][:4])

                    if (reset[agent] == True):
                        reset[agent] = False
                        currentPosition[agent] = agentPosition[agent]
                        if not test:
                            orientation[agent] = orientationInit[agent]
                        else:
                            orientation[agent] = 0
                        orientationDes[agent] = Lib.get_orientation(currentPosition[agent], desPosition[agent])
                        angle[agent] = Lib.get_angle(orientation[agent], orientationDes[agent])
                        messageArray[agent] = Lib.angle_to_message(angle[agent])
                        if not test:
                            index = np.where(idOrder_array==agent)
                            device.send_data_async(remote_devices[index[0][0]], "R," + messageArray[agent] + ">")
                        print("Object ID: " + str(agent) + " Orientation: " + str(orientation[agent])[:4] + " Desired Orientation: " + str(orientationDes[agent])[:4] + " Message Sent: "+ messageArray[agent][:4])

              isEdge[agent] = False
              pass_var = False

# Annotate Image

              # cv2.putText(frame, str(Lib.get_orientation(currentPosition[objectID], agentPosition[agent])),
              #         (agentPosition[agent][0] - 15, agentPosition[agent][1] - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
              #
              # cv2.putText(frame, str(Lib.get_orientation(agentPosition[agent], desPosition[objectID])),
              #         (agentPosition[agent][0] - 15, agentPosition[agent][1] - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
              #
              # cv2.putText(frame, Lib.angle_to_message(Lib.get_angle(Lib.get_orientation(currentPosition[objectID], agentPosition[agent]), Lib.get_orientation(agentPosition[agent], desPosition[objectID]))),
              #         (agentPosition[agent][0] - 15, agentPosition[agent][1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

              #cv2.arrowedLine(frame, (currentPosition[objectID][0], currentPosition[objectID][1]), (agentPosition[agent][0], agentPosition[agent][1]), (255, 0, 0), 1)
              cv2.line(frame, (agentPosition[agent][0], agentPosition[agent][1]), (int(desPosition[agent][0]), int(desPosition[agent][1])), (255, 0, 0), 1)


# Display Image

              if complete == len(remote_devices):
                  break
              complete = 0

              fig, axs = plt.subplots(1, 1, figsize=(15, 10))

              for i in range(len(remote_devices)):
                  x, y = zip(*voronoi_update[i])
                  plt.scatter(x, y, s=80)

              x, y = zip(*desPosition)
              plt.scatter(x, y, color='b', s = 150)

              x, y = zip(*agentPosition)
              plt.scatter(x, y, color='k', s = 150)

              if option == 2 or option == 3:
                  plt.scatter(target[0], target[1], color='r')

              plt.axis('off')
              plt.gca().invert_yaxis()
              plt.savefig("Data/option{}/plots/plot{}".format(option, frame_count))
              plt.close('all')

              if option == 2 or option == 3:
                  cv2.circle(frame, (int(target[0]), int(target[1])), 4, (0, 0, 255), -1)
              cv2.imwrite("Data/option{}/frames/frame{}.jpg".format(option, frame_count), frame)
              cv2.imshow("Frame", frame)
              cv2.waitKey(1)
              frame_count = frame_count + 1

# End Loop

              if cv2.waitKey(1) & 0xFF == ord("x"):
                  for i in range(0, len(remote_devices)):
                      if not test:
                          device.send_data_async(remote_devices[i], "P,S>")
                  break

# Close Video Capture and Windows

cv2.destroyAllWindows()
vidcap.release()
data_txt.close()
if not test:
    device.close()
