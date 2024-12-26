from utils import *

##############################################################

# datasets = ['kitti', 'malaga', 'parking']
datasets = ['kitti']
# detectors = ['sift', 'harris', 'shi_tomasi', 'fast']
detectors = ['sift', 'harris', 'shi_tomasi']
# detectors = ['shi_tomasi']

est_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'processed_result_files'))
analysis_results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "results", "vio_mini_project"))

##############################################################

# generate ground truth trajectories
kitti_gt_traj, malaga_gt_traj, parking_gt_traj, kitti_times, malaga_times, parking_times = generate_gt_trajs()

# preprocess raw estimated trajectories
preprocess_raw_trajs(kitti_times, malaga_times, parking_times)

# write them to the analysis results folder
for dataset in datasets:
    for detector in detectors:
        
        # load the estimated trajectory
        est_traj = np.loadtxt(est_dir + '/' + dataset + '/' + detector + '.txt', delimiter=' ')
        num_recorded_frames = est_traj.shape[0]
        
        # slice the ground truth trajectory (only the last num_recorded_frames frames)
        if dataset == 'kitti':
            gt_traj = kitti_gt_traj[-num_recorded_frames:, :]
        # elif dataset == 'malaga':
        #     gt_traj = malaga_gt_traj[-num_recorded_frames:, :]
        elif dataset == 'parking':
            gt_traj = parking_gt_traj[-num_recorded_frames:, :]
            
        # write them into the analysis results folder
        np.savetxt(analysis_results_dir + '/' + dataset + '/' + detector + '/' + 'stamped_groundtruth.txt', gt_traj, delimiter=' ')
        np.savetxt(analysis_results_dir + '/' + dataset + '/' + detector + '/' + 'stamped_traj_estimate.txt', est_traj, delimiter=' ')

        print("============= Successfully written files for [%s] [%s] =============" % (dataset.upper(), detector.upper()))