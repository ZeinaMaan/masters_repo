from scipy.spatial import distance as dist
from collections import OrderedDict

# Importing utility libraries
import numpy as np
import tensorflow as tf
import cv2
import math
import time
import random

# Importing Object Detection libraries
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util

class Voronoi:
    def get_distance(self, point1, point2):
         distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
         return distance

    def voronoi(self, agentPosition, numAgents, xdim, ydim):
        distanceArray = []
        distanceCollection  = []
        indexCollection = []
        voronoiCollection = []
        counter = 0

        for n in range(numAgents):
            distanceArray.insert(n,[])
            indexCollection.insert(n,[])
            voronoiCollection.insert(n,[])
            for i in range(len(ydim)):
                for j in range(len(xdim)):
                    dist = self.get_distance(agentPosition[n], [xdim[j], ydim[i]])
                    distanceArray[n].append(dist)
            distanceCollection.insert(n, np.array(distanceArray[n]).reshape( np.size(ydim),np.size(xdim) ))

        for n in range(numAgents):
            for ii in range(len(ydim)):
                for jj in range(len(xdim)):

                    for m in range(numAgents):
                        if distanceCollection[n][ii,jj] <= distanceCollection[m][ii,jj]:
                            counter = counter + 1

                        if counter == numAgents:
                            voronoiCollection[n].append([xdim[jj],ydim[ii]])
                            indexCollection[n].append([jj,ii])

                    counter = 0

        return distanceCollection, voronoiCollection, indexCollection

    def risk_density(self, x_dim, y_dim, target, alpha, beta):
      phi = np.zeros((len(y_dim), len(x_dim)))

      for i in range(len(y_dim)):
          for j in range(len(x_dim)):
              dis = self.get_distance([x_dim[j], y_dim[i]], target)
              dis = dis**2
              phi[i][j] = alpha*(np.exp(-beta*dis))

      return phi

    def centroid(self, vor, index, risk_den):
        risk = 0
        risk_x = 0
        risk_y = 0

        for i in range(len(vor)):
            risk += risk_den[index[i][1]][index[i][0]]
            risk_x += vor[i][0]*risk_den[index[i][1]][index[i][0]]
            risk_y += vor[i][1]*risk_den[index[i][1]][index[i][0]]

        C_x = (risk_x/risk)
        C_y = (risk_y/risk)

        return [C_x, C_y]

    def sensing(self, dis):
        sense = np.exp(-(dis**2)/(30**2))
        return sense

    def coverage_metric(self, agent_pos, vor, index, risk_den):
        coverage = 0

        for i in range(len(vor)):
            dis = self.get_distance(agent_pos, vor[i])
            coverage += risk_den[index[i][1]][index[i][0]]*self.sensing(dis)

        return coverage

class Tracker():
	def __init__(self, maxDisappeared=50):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		# check to see if the list of input bounding box rectangles
		# is empty
		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# return early as there are no centroids or tracking info
			# to update
			return self.objects

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				# val
				if row in usedRows or col in usedCols:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		# return the set of trackable objects
		return self.objects

def angle_to_message(angle):
    message = str(angle)
    return message

def get_angle(angle1, angle2):
    angle = angle2 - angle1
    if angle > 360:
        angle = angle - 360
    if angle < 0:
        angle = angle + 360
    return angle

def get_distance(point1, point2):
     distance = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
     return distance

def get_orientation(point1, point2):
     orientation = math.degrees(math.atan2(point1[1] - point2[1], point2[0] - point1[0]))
     if orientation < 0:
         orientation = 360 + orientation
     return orientation

def test_1(idArray, numRobots):
     test1 = False
     testSum = ((numRobots - 1)*(numRobots))/2
     sum = 0
     for i in range(0, numRobots):
         sum += idArray[i]

     if (sum == testSum):
         test1 = True

     return test1

def test_2(idArray, numRobots):
     test2 = False
     testArray = []
     for i in range(0, numRobots):
         testArray.append(idArray[i])

     if (len(testArray) == len(set(testArray))):
         test2 = True

     return test2

def run_inference_for_single_image(image, graph, sess):
      # Get handles to input and output tensors
      ops = tf.compat.v1.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
      return output_dict
