import math
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import style
plt.rcParams.update({'font.size': 10})

limit = 40  # km/hr
distance = 16 # Field Of View Keseluruhan dalam sebuah footage

file = open("SpeedRecord.txt", "w")
file.write("ID \t SPEED\n------\t-------\n")
file.close()


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}

        self.id_count = 0
        # self.start = 0
        # self.stop = 0
        self.et = 0
        self.s1 = np.zeros((1, 100000))
        self.s2 = np.zeros((1, 100000))
        self.s = np.zeros((1, 100000))
        self.f = np.zeros(100000)
        self.capf = np.zeros(100000)
        self.count = 0
        self.exceeded = 0
        self.ids_DATA = []
        self.spd_DATA = []

    def update(self, objects_rect):
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h, index = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # CHECK IF OBJECT IS DETECTED ALREADY
            same_object_detected = False

            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1]) # mengembalikan norma Euclidean

                if dist < 40:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id, index])
                    same_object_detected = True

                    # START TIMER
                    if 235 <= y <= 255:
                        self.s1[0, id] = time.time()

                    # STOP TIMER and FIND DIFFERENCE
                    if 135 <= y <= 155:
                        self.s2[0, id] = time.time()
                        self.s[0, id] = self.s2[0, id] - self.s1[0, id]

                    # CAPTURE FLAG

                    if y <= 135:
                        self.f[id] = 1

                    break

            # NEW OBJECT DETECTION
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count, index])
                self.id_count += 1
                self.s[0, self.id_count] = 0
                self.s1[0, self.id_count] = 0
                self.s2[0, self.id_count] = 0

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id, index = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids

    # SPEEED FUNCTION
    def getsp(self, id):
        if self.s[0, id] != 0:
            s = distance / self.s[0, id] * 3.6
        else:
            s = 0

        return int(s)

    # SAVE VEHICLE DATA
    def capture(self, img, x, y, h, w, sp, id):
        if (self.capf[id] == 0):
            self.capf[id] = 1
            self.f[id] = 0

            crop_img = img[y - 2:y + h + 2, x - 2:x + w + 2]
            n = str(id) + "_speed_" + str(sp)
            file = 'output/detect/detect' + n + '.jpg'
            cv2.imwrite(file, crop_img)
            self.count += 1

            filet = open("SpeedRecord.txt", "a")
            if sp > limit:
                file2 = 'output/overspeed/OverSpeed' + n + '.jpg'
                cv2.imwrite(file2, crop_img)
                filet.write(str(id) + " \t " + str(sp) + " KM/H" + " <---overspeed\n")
                self.exceeded += 1
            if sp < limit:
                filet.write(str(id) + " \t " + str(sp) + "\n")
            filet.close()
            self.ids_DATA.append((id))
            self.spd_DATA.append((sp))

    # STORE DATA
    def dataset(self):
        return self.ids_DATA, self.spd_DATA

    # DATA VISUALIZATION
    def datavis(self, id_lst, spd_lst):
        x = id_lst
        y = spd_lst
        valx = []

        for i in x:
            valx.append(str(i))

        plt.figure(figsize=(20, 5))
        style.use('dark_background')
        plt.axhline(y=limit, color='r', linestyle='-', linewidth='5')
        plt.bar(x, y, width=0.5, linewidth='3', edgecolor='yellow', color='blue', align='center')
        plt.xlabel('ID')
        plt.ylabel('SPEED')
        plt.xticks(x, valx)
        plt.legend(["speed limit"])
        plt.title('SPEED OF VEHICLES CROSSING ROAD\n')
        plt.savefig("output//datavis.png", bbox_inches='tight', pad_inches=1, edgecolor='w', orientation='landscape')

    # SPEED_LIMIT
    def limit(self):
        return limit

    # TEXT FILE SUMMARY
    def end(self):
        file = open("SpeedRecord.txt", "a")
        file.write("\n-------------\n")
        file.write("-------------\n")
        file.write("SUMMARY\n")
        file.write("-------------\n")
        file.write("Total Vehicles :\t" + str(self.count) + "\n")
        file.write("Vehicle Overspeed :\t" + str(self.exceeded))
        file.close()
