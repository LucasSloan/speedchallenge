import numpy as np
import cv2
from extract import create_records, write_records
import glob
from multiprocessing import Pool
import camera
import orientation
import tensorflow as tf
import random

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_list_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

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

    while(True):
        ret, frame = cam.read()

        if ret:
            frame = frame[219:-218]
            frame = cv2.resize(frame, (416, 128), interpolation=cv2.INTER_AREA)
            ret, jpg = cv2.imencode(".jpg", frame)
            if not ret:
                raise Exception("couldn't encode the image")

            frame_bytes = jpg.tostring()
            frames.append(frame_bytes)

        else:
            break


    frame_positions = np.load(path + '/global_pose/frame_positions')
    frame_orientations = np.load(path + '/global_pose/frame_orientations')

    assert len(frames) == len(frame_positions) == len(frame_orientations)

    examples = []
    for i in range(len(frame_positions[:-3])):
        plus_one_position = camera.device_from_ecef(frame_positions[i], frame_orientations[i], frame_positions[i+1])
        plus_one_orientation = relative_orientation(frame_orientations[i], frame_orientations[i+1])

        plus_two_position = camera.device_from_ecef(frame_positions[i], frame_orientations[i], frame_positions[i+2])
        plus_two_orientation = relative_orientation(frame_orientations[i], frame_orientations[i+2])

        plus_three_position = camera.device_from_ecef(frame_positions[i], frame_orientations[i], frame_positions[i+3])
        plus_three_orientation = relative_orientation(frame_orientations[i], frame_orientations[i+3])
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'frame_one': _bytes_feature(frames[i]),
            'frame_two': _bytes_feature(frames[i+1]),
            'frame_three': _bytes_feature(frames[i+2]),
            'frame_four': _bytes_feature(frames[i+3]),
            'plus_one_position': _float_list_feature(plus_one_position),
            'plus_one_orientation': _float_list_feature(plus_one_orientation),
            'plus_two_position': _float_list_feature(plus_two_position),
            'plus_two_orientation': _float_list_feature(plus_two_orientation),
            'plus_three_position': _float_list_feature(plus_three_position),
            'plus_three_orientation': _float_list_feature(plus_three_orientation),
            'speed': _float_feature(frame_velocities[i]),
        }))
        examples.append(example.SerializeToString())

    return examples

def write_tfrecord(dirs, out):
    with tf.python_io.TFRecordWriter(out) as writer:
        example_buffer = []
        for i, dir in enumerate(dirs):
            example_buffer.extend(extract_segment(dir))
            print("extracted {} segments".format(i))

            if len(example_buffer) > 1000000:
                random.shuffle(example_buffer)
                for e in example_buffer:
                    writer.write(e)
                example_buffer = []

                print("\n\nflushed the buffer\n\n")

        random.shuffle(example_buffer)
        for e in example_buffer:
            writer.write(e)


segments = glob.glob("/mnt/Bulk/commaai/comma2k19/*/*/*")
random.shuffle(segments)

train = segments[:1933]
validation = segments[1933:]

write_tfrecord(train, "/mnt/Bulk/commaai/monolithic_train.tfrecord")
write_tfrecord(validation, "/mnt/Bulk/commaai/monolithic_validation.tfrecord")

# extract_segment("/mnt/d/commaai/comma2k19/Chunk_1/b0c9d2329ad1606b_2018-07-27--06-03-57/3")