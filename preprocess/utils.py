import numpy as np
import os
from typing import Tuple
from scipy.spatial.transform import Rotation as R


##############################################################################################
def read_raw_gt_files(verbose=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    gt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ground_truth_files'))
    if verbose:
        print("============= Reading ground truth files =============")
    
    # KITTI
    kitti_gt_file = gt_dir + '/kitti.txt'
    kitti_gt_traj = np.loadtxt(kitti_gt_file, delimiter=' ')
    kitti_times_file = gt_dir + '/kitti_timestamps.txt'
    kitti_times = np.loadtxt(kitti_times_file, delimiter=' ').reshape(-1, 1)
    kitti_gt = np.hstack((kitti_times, kitti_gt_traj))
    if verbose:
        print("[KITTI] Ground truth has %d frames, %d columns" % (kitti_gt.shape[0], kitti_gt.shape[1]))
    
    # MALAGA
    malaga_gt_file = gt_dir + "/malaga.txt"
    malaga_gt = np.genfromtxt(malaga_gt_file, delimiter=None, skip_header=1)
    if verbose:
        print("[MALAGA] Ground truth has %d frames, %d columns" % (malaga_gt.shape[0], malaga_gt.shape[1]))
    
    # Parking
    parking_gt_file = gt_dir + "/parking.txt"
    parking_gt = np.genfromtxt(parking_gt_file, delimiter=' ')
    if verbose:
        print("[Parking] Ground truth has %d frames, %d columns" % (parking_gt.shape[0], parking_gt.shape[1]))
    
    if verbose:
        print("======================================================")
    
    print("============= Finished reading ground truth files =============")
        
    return kitti_gt, malaga_gt, parking_gt


##############################################################################################
def generate_gt_trajs(verbose=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    kitti_gt, malaga_gt, parking_gt = read_raw_gt_files(verbose=verbose)
    
    # KITTI
    kitti_gt_traj = np.zeros((kitti_gt.shape[0], 8))
    kitti_timestamps = kitti_gt[:, 0]    # timestamps
    kitti_gt_traj[:, 0] = kitti_timestamps
    # convert each pose
    for i in range(kitti_gt.shape[0]):
        transform_mat = kitti_gt[i, 1:].reshape(3, 4, order='C')
        rot_mat, trans_vec = transform_mat[:, :3], transform_mat[:, 3]
        quaternion = R.from_matrix(rot_mat).as_quat()
        pose = np.concatenate([trans_vec, quaternion])
        kitti_gt_traj[i, 1:] = pose
    if verbose:
        print("[KITTI] Ground truth trajectory has %d frames, %d columns" % (kitti_gt_traj.shape[0], kitti_gt_traj.shape[1]))
 
    # MALAGA
    malaga_gt_traj = np.zeros((malaga_gt.shape[0], 8))
    start_time = malaga_gt[0, 0]
    malaga_timestamps = malaga_gt[:, 0] - start_time    # timestamps (GPS frequency is 1 Hz)
    malaga_gt_traj[:, 0] = malaga_timestamps
    # convert each pose
    for i in range(malaga_gt.shape[0]):
        trans_vec = malaga_gt[i, 8:11]
        quaternion = np.array([0.0, 0.0, 0.0, 1.0])         # no orientation information
        pose = np.concatenate([trans_vec, quaternion])
        malaga_gt_traj[i, 1:] = pose
    if verbose:
        print("[MALAGA] Ground truth trajectory has %d frames, %d columns" % (malaga_gt_traj.shape[0], malaga_gt_traj.shape[1]))
    
    # Parking
    parking_gt_traj = np.zeros((parking_gt.shape[0], 8))
    parking_timetamps = np.linspace(0.0, parking_gt.shape[0]-1, parking_gt.shape[0]) / 10.0    # timestamps (every 0.1 seconds)
    parking_gt_traj[:, 0] = parking_timetamps
    # convert each pose
    for i in range(parking_gt.shape[0]):
        transform_mat = parking_gt[i, :].reshape(3, 4, order='C')
        rot_mat, trans_vec = transform_mat[:, :3], transform_mat[:, 3]
        quaternion = R.from_matrix(rot_mat).as_quat()
        pose = np.concatenate([trans_vec, quaternion])
        parking_gt_traj[i, 1:] = pose
    if verbose:
        print("[Parking] Ground truth trajectory has %d frames, %d columns" % (parking_gt_traj.shape[0], parking_gt_traj.shape[1]))
        
    print("============= Finished generating ground truth trajectories =============")
        
    return kitti_gt_traj, malaga_gt_traj, parking_gt_traj, kitti_timestamps, malaga_timestamps, parking_timetamps
           

##############################################################################################
def preprocess_raw_trajs(kitti_times, malaga_times, parking_times, verbose=False):
    
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'raw_result_files'))
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'processed_result_files'))
    # detectors = ['sift', 'harris', 'shi_tomasi', 'fast']
    detectors = ['sift', 'harris', 'shi_tomasi']

    # KITTI
    raw_kitti_dir = raw_dir + '/kitti/'
    processed_kitti_dir = processed_dir + '/kitti/'
    for detector in detectors:
        # read raw trajectory
        kitti_raw_traj = np.loadtxt(raw_kitti_dir + detector + '.txt', delimiter=' ')
        if verbose:
            print("[KITTI] Recorded trajectory [%s] has %d frames, %d columns" % (detector.upper(), kitti_raw_traj.shape[0], kitti_raw_traj.shape[1]))
        # preprocess the trajectory
        kitti_processed_traj = np.zeros((kitti_raw_traj.shape[0], 8))
        kitti_processed_traj[:, 0] = kitti_times[-kitti_processed_traj.shape[0]:].reshape(-1)
        # convert each pose
        for i in range(kitti_raw_traj.shape[0]):
            transform_mat = kitti_raw_traj[i, :].reshape(3, 4, order='C')
            rot_mat, trans_vec = transform_mat[:, :3], transform_mat[:, 3]
            quaternion = R.from_matrix(rot_mat).as_quat()
            pose = np.concatenate([trans_vec, quaternion])
            kitti_processed_traj[i, 1:] = pose
        # save the processed trajectory
        np.savetxt(processed_kitti_dir + detector + '.txt', kitti_processed_traj, delimiter=' ')
        
    # MALAGA
    raw_malaga_dir = raw_dir + '/malaga/'
    processed_malaga_dir = processed_dir + '/malaga/'
    for detector in detectors:
        # read raw trajectory
        malaga_raw_traj = np.loadtxt(raw_malaga_dir + detector + '.txt', delimiter=' ')
        if verbose:
            print("[MALAGA] Recorded trajectory [%s] has %d frames, %d columns" % (detector.upper(), malaga_raw_traj.shape[0], malaga_raw_traj.shape[1]))
        # preprocess the trajectory
        malaga_processed_traj = np.zeros((malaga_raw_traj.shape[0], 8))
        malaga_processed_traj[:, 0] = np.linspace(malaga_times[0], malaga_times[-1], malaga_raw_traj.shape[0])    # timestamps
        # convert each pose
        for i in range(malaga_raw_traj.shape[0]):
            transform_mat = malaga_raw_traj[i, :].reshape(3, 4, order='C')
            rot_mat, trans_vec = transform_mat[:, :3], transform_mat[:, 3]
            quaternion = R.from_matrix(rot_mat).as_quat()
            pose = np.concatenate([trans_vec, quaternion])
            malaga_processed_traj[i, 1:] = pose
        # save the processed trajectory
        np.savetxt(processed_malaga_dir + detector + '.txt', malaga_processed_traj, delimiter=' ')
        
    # Parking
    raw_parking_dir = raw_dir + '/parking/'
    processed_parking_dir = processed_dir + '/parking/'
    for detector in detectors:
        # read raw trajectory
        parking_raw_traj = np.loadtxt(raw_parking_dir + detector + '.txt', delimiter=' ')
        if verbose:
            print("[Parking] Recorded trajectory [%s] has %d frames, %d columns" % (detector.upper(), parking_raw_traj.shape[0], parking_raw_traj.shape[1]))
        # preprocess the trajectory
        parking_processed_traj = np.zeros((parking_raw_traj.shape[0], 8))
        parking_processed_traj[:, 0] = parking_times[-parking_processed_traj.shape[0]:].reshape(-1)    # timestamps
        # convert each pose
        for i in range(parking_raw_traj.shape[0]):
            transform_mat = parking_raw_traj[i, :].reshape(3, 4, order='C')
            rot_mat, trans_vec = transform_mat[:, :3], transform_mat[:, 3]
            quaternion = R.from_matrix(rot_mat).as_quat()
            pose = np.concatenate([trans_vec, quaternion])
            parking_processed_traj[i, 1:] = pose
        # save the processed trajectory
        np.savetxt(processed_parking_dir + detector + '.txt', parking_processed_traj, delimiter=' ')
        
    print("============= Finished processing raw estimated trajectories =============")
