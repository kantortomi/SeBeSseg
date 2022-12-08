import torch
import numpy as np
import cv2
from time import time
from motpy import Detection, MultiObjectTracker
from shapely.geometry import Point, Polygon
import yaml
import pandas as pd
import os
from tqdm import tqdm


# a class named detectionarea which gets 4 points, defines the area
# the points are tuples
class DetectionArea:
    def __init__(self, coors, detector_count, config, lanes):
        # the coordinates of the area
        self.p1 = coors[0]
        self.p2 = coors[2]
        self.p3 = coors[1]
        self.p4 = coors[3]

        self.config = config

        self.detector_count = detector_count

        self.coordinates1 = self.get_points_on_line(
            self.p1, self.p2, detector_count)
        self.coordinates2 = self.get_points_on_line(
            self.p3, self.p4, detector_count)
        self.detectors = []
        self.create_detectors()

        self.lanes = self.create_lanes(lanes)

    # a function that receives two points that define a line, and gives back n points that are equal distance from each other
    def get_points_on_line(self, p1, p2, n):
        x1, y1 = p1
        x2, y2 = p2
        x = np.linspace(x1, x2, n)
        y = np.linspace(y1, y2, n)
        return list(zip(x, y))

    # a function that creates detectors from coordinates1 and coordinates2
    def create_detectors(self):
        for i in range(len(self.coordinates1)):
            self.detectors.append(Detector(
                i, self.coordinates1[i][0], self.coordinates1[i][1], self.coordinates2[i][0], self.coordinates2[i][1]))

    # a function that draws the area on the image
    def draw(self, frame):
        if self.config['drawDetectors']:
            cv2.line(frame, self.p1, self.p2, (0, 255, 0), 2)
            cv2.line(frame, self.p3, self.p4, (0, 255, 0), 2)
            for i in range(len(self.detectors)):
                self.detectors[i].draw(frame)
        if self.config['drawLanes']:
            self.draw_lanes(frame, self.lanes)

    # checking if the object is in the area of the detection
    def is_in_area(self, point, frame):
        x, y = point
        p1 = Point(x, y)
        coords = [(self.p1[0]-25, self.p1[1]-50), (self.p2[0]-50, self.p2[1]+150),
                  (self.p4[0]+75, self.p4[1]+200), (self.p3[0]+300, self.p3[1]-75)]
        if self.config['drawArea']:
            cv2.polylines(frame, np.asarray([coords]), True, (0, 255, 0), 2)
        if self.config['drawMid']:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), 2)

        poly = Polygon(coords)
        return poly.contains(p1)

    # a function that creates lane objects from the coordinates
    def create_lanes(self, lanes):
        lane_objects = []
        for i in range(len(lanes)-1):
            lane_objects.append(
                Lane(lanes[i][0], lanes[i][1], lanes[i+1][0], lanes[i+1][1], i))
        return lane_objects

    # a function that draws the lanes on the image
    def draw_lanes(self, frame, lanes):
        for lane in lanes:
            lane.draw(frame)


# a class named lane that gets 2 lines, and the area between them is a lane
class Lane:
    def __init__(self, p1, p2, p3, p4, id):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.id = int(id)

    # a function that defines an area between two lines
    def get_area(self):
        return [(self.p1[0], self.p1[1]), (self.p2[0], self.p2[1]), (self.p4[0], self.p4[1]), (self.p3[0], self.p3[1])]

    # check if an object is in the lane
    def is_in_lane(self, point):
        x, y = point
        p1 = Point(x, y)
        poly = Polygon(self.get_area())
        return poly.contains(p1)

    # a function that draws the lane on the image, and the id of the lane
    def draw(self, frame):
        cv2.line(frame, self.p1, self.p2, (0, 255, 0), 1)
        cv2.line(frame, self.p3, self.p4, (0, 255, 0), 1)
        cv2.putText(frame, str(self.id), (int(
            self.p1[0])+25, int(self.p1[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)


# a class named detector which defines a detector line from the given points
class Detector:
    def __init__(self, id, x1, y1, x2, y2):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)

        self.detector_id = id
        self.object_memory = []

    # check if an objects midpoint passed through the detectors line
    def check_object(self, object, timeframe):
        x1, y1 = self.x1, self.y1
        x2, y2 = self.x2, self.y2
        x3, y3 = object.mid_point[0], object.mid_point[1]

        # calculate the distance between the midpoint of the object and the line
        distance = abs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 *
                       y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

        # if the distance is less than 3, the object passed through the detector
        if distance < 2 and object.id not in [x[0].id for x in self.object_memory]:
            self.add_object(object, timeframe)
            return [True, self.detector_id]
        else:
            return [False]

    # if the objects id is not in the memory, add it
    def add_object(self, object, timeframe):
        if object.id not in [x[0].id for x in self.object_memory]:
            object.time_to_live -= 1
            self.object_memory.append([object, timeframe])

    # remove a object from the memory if by the time it crosses the detector the time to live is 0
    def remove_object(self, object):
        if object in self.object_memory:
            self.object_memory.remove(object)

    def __str__(self):
        return f'Detector {self.detector_id} ({self.x1},{self.y1}),({self.x2},{self.y2})'

    # drawing the detector on screen
    def draw(self, frame):
        cv2.line(frame, (self.x1, self.y1), (self.x2, self.y2), (0, 0, 255), 2)
        cv2.putText(frame, str(self.detector_id), (self.x1, self.y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


class TrackedObject:
    def __init__(self, track, detectionarea, framepersecond, _distances, currentfps):
        self._track = track
        self.speedestimate = []
        self._speed = 0

        self.id = track.id
        self.bounding_box = track.box[:]

        self.framepersecond = framepersecond
        self._distances = _distances
        self.currentfps = currentfps

        self.mid_point = self.midpoint()
        self.last_detector = None
        self.time_to_live = detectionarea.detector_count

        self.previousdetector = None
        self.passeddetectors = 0
        self.lane = 0

    # a function that check which lane is the object in
    def check_lanes(self, lanes):
        for lane in lanes:
            if lane.is_in_lane(self.mid_point):
                return lane.id

    def __repr__(self):
        return f'{self.id} {self._speed} {self.last_detector}'

    # updating the position of the object
    def updatepos(self, box, fps):
        self.bounding_box = box
        self.mid_point = self.midpoint()
        self.currentfps = fps

    # claculating the current speed of an object
    def calculatespeed(self, framenumber, detectionarea):
        timestamp = framenumber
        self.mid_point = self.midpoint()
        for detector in detectionarea.detectors:
            if self.time_to_live > 0:
                has_passed_detector = detector.check_object(self, timestamp)
                if len(has_passed_detector) != 1 and has_passed_detector[1] == detector.detector_id:
                    self.current_detector = detector
                    self.passeddetectors += 1
                    section_speed = self.getsectionspeed(
                        detectionarea, timestamp)
                    self.lane = self.check_lanes(detectionarea.lanes)

                    if section_speed != 0:
                        self.speedestimate.append(section_speed)

                    self._speed = 0 if not len(self.speedestimate) else sum(
                        self.speedestimate) / len(self.speedestimate)

                    self.last_detector = detector

            else:
                detector.remove_object(self)

    # claculating the speed of a section that the object has passed through
    def getsectionspeed(self, detectionarea, timestamp):
        if self.last_detector is not None and self.current_detector is not None and self.last_detector.detector_id != self.current_detector.detector_id:
            for object in self.last_detector.object_memory:
                if object[0].id == self.id:
                    # get the time the object passed through the detector
                    previoustimestamp = object[1]
                    if timestamp != 0 and self.last_detector != None and self.lane != None:
                        t = timestamp - previoustimestamp
                        timeInSeconds = (t)
                        distanceInMeters = self._distances[self.lane] / \
                            (detectionarea.detector_count-1)

                        section_speed = 3.6 * \
                            (distanceInMeters / abs(timeInSeconds)) * \
                            self.framepersecond
                        return section_speed

        return 0

    # a function that calculates the objects midpoint
    def midpoint(self):
        return [int((self.bounding_box[0] + self.bounding_box[2])/2), int((self.bounding_box[1] + self.bounding_box[3])/2)]


class ObjectDetection:
    def __init__(self, file, coors, distances, detectionarea, config, out_file="Labeled_Video.mp4"):
        self._file = file
        self._area = coors
        self._distances = distances
        self.model = self.load_model()
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tracker = MultiObjectTracker(dt=0.1)
        self.tracks = []
        self.trackedobjects = []
        self.framepersecond = 0
        self.currentfps = 0
        self.detectionarea = detectionarea
        self.config = config

    def get_video_from_file(self):
        return cv2.VideoCapture(self._file)

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5',
                               'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame2 = frame
        frame = [frame]
        if not None:
            results = self.model(frame)

        results = [obj for obj in results.pandas().xyxy[0].to_numpy() if self.check_if_in_area(
            self.midpoint(obj), self.detectionarea, frame2) and (obj[6] == 'car' or obj[6] == 'truck')]
        detections = []
        for object in results:
            detections.append(
                Detection(box=np.array([int(object[0]), int(object[1]), int(object[2]), int(object[3])])))

        _ = self.tracker.step(detections=detections)
        self.tracks = self.tracker.active_tracks()

        tracklist = []
        objectlist = []
        midpointlist = []

        for trackedobject in self.trackedobjects:
            if trackedobject.lane == 0:
                pass
            objectlist.append(trackedobject.id)
            midpointlist.append(trackedobject.mid_point)

        for track in self.tracks:
            tracklist.append(track.id)

        for track in self.tracks:
            if track.id not in objectlist:
                self.trackedobjects.append(TrackedObject(
                    track, self.detectionarea, self.framepersecond, self._distances, self.currentfps))
            for trackedobject in self.trackedobjects:
                if trackedobject.id == track.id:
                    trackedobject.updatepos(track.box[:], self.currentfps)
                if trackedobject.id not in tracklist:
                    if trackedobject._speed != 0 and trackedobject._speed < 75:
                        output_speed_data[trackedobject.lane].append(
                            trackedobject._speed)
                        if trackedobject.lane == 4:
                            output_speed_data_4lanes[3].append(
                                trackedobject._speed)
                        else:
                            output_speed_data_4lanes[trackedobject.lane].append(
                                trackedobject._speed)
                    self.trackedobjects.remove(trackedobject)
        return results

    def printspeed(self, frame, framenumber):
        for track in self.trackedobjects:
            label = f'{track.id[:5]}'
            track.calculatespeed(framenumber, self.detectionarea)
            if track.last_detector is not None:
                speed = "{:.2f}".format(track._speed)
                # {track.current_detector.detector_id}
                if self.config['drawText']:
                    cv2.putText(frame, f"{speed} km/h {track.lane} ", (int(track.bounding_box[0]), int(
                        track.bounding_box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    def plot_boxes(self, results, frame):
        for i in range(len(results)):
            row = results[i]
            if row[4] >= 0.4:
                x1, y1, x2, y2 = int(row[0]), int(
                    row[1]), int(row[2]), int(row[3])
                if self.config['drawRect']:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    # function that calculates midpoint of a bounding box
    def midpoint(self, box):
        return [int((box[0] + box[2])/2), int((box[1] + box[3])/2)]

    def __call__(self):
        player = self.get_video_from_file()
        assert player.isOpened()
        self.framepersecond = player.get(cv2.CAP_PROP_FPS)
        pbar = tqdm(total=int(player.get(cv2.CAP_PROP_FRAME_COUNT)))
        while player.isOpened():
            start_time = time()
            ret, frame = player.read()
            if frame is not None:
                results = self.score_frame(frame)
                frame = self.plot_boxes(results, frame)
                self.detectionarea.draw(frame)
                #frame = self.plot_midpoint(results, frame)
                end_time = time()

                self.currentfps = 1/np.round(end_time - start_time, 3)
                framenumber = player.get(cv2.CAP_PROP_POS_FRAMES)

                self.printspeed(frame, framenumber)

                cv2.putText(frame, "FPS : " + str(int(self.currentfps)), (100, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

                cv2.imshow("Autopalya", frame)
                pbar.update(1)
                if cv2.waitKey(5) == 27:
                    break
            else:
                break

    def check_if_in_area(self, point, area, frame):
        # check if a point is in the area defined by the detectionarea
        return area.is_in_area(point, frame)


def main():
    global output_speed_data
    global output_speed_data_4lanes
    output_speed_data = {0: [], 1: [], 2: [], 3: [], 4: []}
    output_speed_data_4lanes = {0: [], 1: [], 2: [], 3: []}

    config = yaml.safe_load(open("conf.yml"))
    coors = np.array([[28, 148], [492, 105],   # x1, y1
                     [156, 288], [866, 189]])   # x2, y2

    lanes = np.array([
        [[49, 95], [257, 542]],   # lane 0
        [[128, 92], [614, 443]],  # lane 1
        [[177, 86], [799, 393]],  # lane 2
        [[236, 92], [936, 343]],  # lane 3
        [[289, 83], [955, 255]],  # lane 4
        [[380, 79], [954, 140]],  # lane 5
    ])
    distances = [19.46, 20.76, 22.06, 23.36, 24.66]

    area = DetectionArea(coors, 5, config, lanes)

    a = ObjectDetection("./pecs.avi", coors, distances, area, config)
    a()
    # create dataframe from output_speed_data and output_speed_data_4lanes
    df = pd.DataFrame(dict([(k, pd.Series(v))
                      for k, v in output_speed_data.items()]))
    df_lanes = pd.DataFrame(dict([(k, pd.Series(v))
                                  for k, v in output_speed_data_4lanes.items()]))
    five_lanes = 0
    four_lanes = 0

    while os.path.exists("output_data/output_speed_data%s.csv" % five_lanes):
        five_lanes += 1
    df.to_csv('output_data/output_speed_data%s.csv' % 
            five_lanes, index=False)

    while os.path.exists("output_data/output_speed_data_4lanes%s.csv" % four_lanes):
        four_lanes += 1
    df_lanes.to_csv('output_data/output_speed_data_4lanes%s.csv' %
              four_lanes, index=False)


if __name__ == "__main__":
    main()
