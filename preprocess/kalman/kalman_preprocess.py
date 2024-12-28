from kalman_utils import *

##############################################################

datasets = ['kitti', 'malaga', 'parking', 'kitti_kf', 'malaga_kf', 'parking_kf']

est_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'processed_result_files'))
analysis_results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', "results", "kalman_vio_mini_project"))

##############################################################

# generate ground truth trajectories
kitti_gt_traj, malaga_gt_traj, parking_gt_traj, kitti_times, malaga_times, parking_times = generate_gt_trajs()

# preprocess raw estimated trajectories
preprocess_raw_trajs(kitti_times, malaga_times, parking_times)

# loop through 6 datasets (3 with Kalman, 3 without)
for dataset in datasets:
    
    if dataset in ["kitti", "kitti_kf"]:
        gt_traj = kitti_gt_traj
    elif dataset in ["malaga", "malaga_kf"]:
        gt_traj = malaga_gt_traj
    elif dataset in ["parking", "parking_kf"]:
        gt_traj = parking_gt_traj
        
    # load the estimated trajectory
    est_traj = np.loadtxt(est_dir + '/' + dataset + '.txt', delimiter=' ')
    num_recorded_frames = est_traj.shape[0]
        
    # write them into the analysis results folder
    np.savetxt(analysis_results_dir + '/' + dataset + '/' + 'stamped_groundtruth.txt', gt_traj, delimiter=' ')
    np.savetxt(analysis_results_dir + '/' + dataset + '/' + 'stamped_traj_estimate.txt', est_traj, delimiter=' ')

    print("============= Successfully written files for [%s] =============" % (dataset))