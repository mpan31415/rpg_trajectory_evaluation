import numpy as np
import os
from typing import Tuple
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
from pandas import read_pickle


##############################################################################################
def extract_coordinates_from_kml(kml_file):
        
    tree = ET.parse(kml_file)
    root = tree.getroot()
    
    # Find coordinates tag (note KML namespace)
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}
    coords_elem = root.find('.//kml:coordinates', ns)
    
    if coords_elem is not None:
        # Parse coordinate string
        coords_str = coords_elem.text.strip()
        coords_list = []
        
        for line in coords_str.split():
            if line.strip():
                # Split coordinate values, default height to 0 if not provided
                parts = line.split(',')
                lon = float(parts[0])
                lat = float(parts[1])
                coords_list.append([lon, lat])
        
        return np.array(coords_list)
    return None


##############################################################################################
def read_raw_gt_files(verbose=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    gt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ground_truth_files'))
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
    malaga_gt_file = gt_dir + "/malaga_kml.kml"
    gps_coords = extract_coordinates_from_kml(malaga_gt_file)
    if gps_coords is not None:
        # Convert to relative coordinates (first point as origin)
        malaga_gt = np.zeros((len(gps_coords), 2))
        
        # Convert latitude/longitude to meters
        lat_scale = 111320  # 1 degree latitude ≈ 111.32km
        lon_scale = 111320 * np.cos(np.radians(gps_coords[0, 1]))  # Longitude scale factor varies with latitude
        
        malaga_gt[:, 0] = (gps_coords[:, 0] - gps_coords[0, 0]) * lon_scale
        malaga_gt[:, 1] = (gps_coords[:, 1] - gps_coords[0, 1]) * lat_scale
        
        if verbose:
            print("Successfully loaded GPS ground truth from KML file")
            print(f"Number of GPS points: {len(malaga_gt)}")
    else:
        malaga_gt = None
        print("No coordinates found in KML file")
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
    malaga_timestamps = np.arange(malaga_gt.shape[0])    # timestamps (GPS is 1 Hz)
    malaga_gt_traj[:, 0] = malaga_timestamps
    # convert each pose
    for i in range(malaga_gt.shape[0]):
        # trans_vec = np.array([malaga_gt[i, 0], malaga_gt[i, 1], 0.0])    # no height information [TODO: check order of axes]
        trans_vec = np.array([malaga_gt[i, 0], 0.0, malaga_gt[i, 1]])    # no height information [TODO: check order of axes]
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
    
    def convert_poses(raw_traj_lst: list) -> np.ndarray:
        num_frames = len(raw_traj_lst)
        processed_traj = np.zeros((num_frames, 8))
        # convert each pose
        for i in range(num_frames):
            transform_mat = raw_traj_lst[i]
            rot_mat, trans_vec = transform_mat[:, :3], transform_mat[:, 3]
            quaternion = R.from_matrix(rot_mat).as_quat()
            pose = np.concatenate([trans_vec, quaternion])
            processed_traj[i, 1:] = pose      # leave first column for timestamps
        return processed_traj
            
    # KITTI
    kitti_raw_traj = read_pickle(raw_dir + '\\KITTI.pkl')
    if verbose:
        print("[KITTI] Recorded trajectory has %d frames" % (len(kitti_raw_traj)))
    # preprocess the trajectory and save it
    kitti_processed_traj = convert_poses(kitti_raw_traj)
    kitti_processed_traj[:, 0] = kitti_times[-kitti_processed_traj.shape[0]:].reshape(-1)     # timestamps 
    np.savetxt(processed_dir + '\\kitti.txt', kitti_processed_traj, delimiter=' ')
    
    # KITTI (Kalman Filter)
    kitti_kf_raw_traj = read_pickle(raw_dir + '\\KITTI_KF.pkl')
    if verbose:
        print("[KITTI_KF] Recorded trajectory has %d frames" % (len(kitti_kf_raw_traj)))
    # preprocess the trajectory and save it
    kitti_kf_processed_traj = convert_poses(kitti_kf_raw_traj)
    kitti_kf_processed_traj[:, 0] = kitti_times[-kitti_kf_processed_traj.shape[0]:].reshape(-1)     # timestamps
    np.savetxt(processed_dir + '\\kitti_kf.txt', kitti_kf_processed_traj, delimiter=' ')
        
    # MALAGA
    malaga_raw_traj = read_pickle(raw_dir + '\\Malaga.pkl')
    if verbose:
        print("[MALAGA] Recorded trajectory has %d frames" % (len(malaga_raw_traj)))
    # preprocess the trajectory and save it
    malaga_processed_traj = convert_poses(malaga_raw_traj)
    malaga_processed_traj[:, 0] = np.linspace(malaga_times[0], malaga_times[-1], len(malaga_raw_traj))    # timestamps
    np.savetxt(processed_dir + '\\malaga.txt', malaga_processed_traj, delimiter=' ')
    
    # MALAGA (Kalman Filter)
    malaga_kf_raw_traj = read_pickle(raw_dir + '\\Malaga_KF.pkl')
    if verbose:
        print("[MALAGA_KF] Recorded trajectory has %d frames" % (len(malaga_kf_raw_traj)))
    # preprocess the trajectory and save it
    malaga_kf_processed_traj = convert_poses(malaga_kf_raw_traj)
    malaga_kf_processed_traj[:, 0] = np.linspace(malaga_times[0], malaga_times[-1], len(malaga_kf_raw_traj))    # timestamps
    np.savetxt(processed_dir + '\\malaga_kf.txt', malaga_kf_processed_traj, delimiter=' ')
        
    # Parking
    parking_raw_traj = read_pickle(raw_dir + '\\parking.pkl')
    if verbose:
        print("[PARKING] Recorded trajectory has %d frames" % (len(parking_raw_traj)))
    # preprocess the trajectory and save it
    parking_processed_traj = convert_poses(parking_raw_traj)
    parking_processed_traj[:, 0] = parking_times[-parking_processed_traj.shape[0]:].reshape(-1)    # timestamps
    np.savetxt(processed_dir + '\\parking.txt', parking_processed_traj, delimiter=' ')
    
    # Parking (Kalman Filter)
    parking_kf_raw_traj = read_pickle(raw_dir + '\\parking_KF.pkl')
    if verbose:
        print("[PARKING_KF] Recorded trajectory has %d frames" % (len(parking_kf_raw_traj)))
    # preprocess the trajectory and save it
    parking_kf_processed_traj = convert_poses(parking_kf_raw_traj)
    parking_kf_processed_traj[:, 0] = parking_times[-parking_kf_processed_traj.shape[0]:].reshape(-1)    # timestamps
    np.savetxt(processed_dir + '\\parking_kf.txt', parking_kf_processed_traj, delimiter=' ') 
    
        
    print("============= Finished processing raw estimated trajectories =============")




# kitti_gt_traj, malaga_gt_traj, parking_gt_traj, kitti_timestamps, malaga_timestamps, parking_timetamps = generate_gt_trajs(verbose=True)

# preprocess_raw_trajs(kitti_timestamps, malaga_timestamps, parking_timetamps, verbose=True)
