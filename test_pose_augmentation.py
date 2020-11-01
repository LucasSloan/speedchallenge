import numpy as np
import cv2
from extract import create_records, write_records
import glob
from multiprocessing import Pool
import camera
import orientation
import tensorflow as tf
import random

# takes in 2 ecef quaternions and returns the euler angle between them
def relative_orientation(ecef_quat_1, ecef_quat_2):
    ecef_mat_1 = orientation.quat2rot(ecef_quat_1)
    ecef_mat_2 = orientation.quat2rot(ecef_quat_2)

    relative_mat = np.matmul(ecef_mat_2, np.transpose(ecef_mat_1))

    return orientation.rot2euler(relative_mat)

def extract_segment(path):
    frame_velocities =  np.linalg.norm(np.load(path + '/global_pose/frame_velocities'),axis=1)
    frame_velocities = list(map(lambda x: x.item(), frame_velocities))


    cam = cv2.VideoCapture(path + "/video.hevc")
    frames = []

    for i in range(10):
        ret, frame = cam.read()
        frames.append(frame)

    frame_positions = np.load(path + '/global_pose/frame_positions')
    frame_orientations = np.load(path + '/global_pose/frame_orientations')

    for i in range(len(frame_positions[:-3])):
        plus_one_position = camera.device_from_ecef(frame_positions[i], frame_orientations[i], frame_positions[i+1])
        plus_one_orientation = relative_orientation(frame_orientations[i], frame_orientations[i+1])

        plus_two_position = camera.device_from_ecef(frame_positions[i], frame_orientations[i], frame_positions[i+2])
        plus_two_orientation = relative_orientation(frame_orientations[i], frame_orientations[i+2])

        plus_three_position = camera.device_from_ecef(frame_positions[i], frame_orientations[i], frame_positions[i+3])
        plus_three_orientation = relative_orientation(frame_orientations[i], frame_orientations[i+3])

        base_frame = frames[i]
        cv2.imwrite("test1.jpg", base_frame)

        plus_one_real = frames[i+1]
        # plus_one_fake = camera.transform_img(np.array(base_frame), augment_trans=np.array(plus_one_position), augment_eulers=np.array(plus_one_orientation))
        plus_one_fake = camera.transform_img(np.array(base_frame), augment_trans=np.array([0, 0, -1]))
        cv2.imwrite("plus_one_real.jpg", plus_one_real)
        cv2.imwrite("plus_one_fake.jpg", plus_one_fake)

        print(plus_one_position)
        print(plus_one_orientation)

        # plus_two_real = frames[i+2]
        # plus_two_fake = camera.transform_img(np.array(base_frame), augment_trans=np.array(plus_two_position), augment_eulers=np.array(plus_two_orientation))
        # cv2.imwrite("plus_two_real.jpg", plus_two_real)
        # cv2.imwrite("plus_two_fake.jpg", plus_two_fake)

        # plus_three_real = frames[i+3]
        # plus_three_fake = camera.transform_img(np.array(base_frame), augment_trans=np.array(plus_three_position), augment_eulers=np.array(plus_three_orientation))
        # cv2.imwrite("plus_three_real.jpg", plus_three_real)
        # cv2.imwrite("plus_three_fake.jpg", plus_three_fake)
        break

extract_segment("D:/commaai/comma2k19/Chunk_1/b0c9d2329ad1606b_2018-07-27--06-03-57/3")