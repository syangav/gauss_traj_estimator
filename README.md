# gauss_traj_estimator

____

This package contains a ROS implementation (C++) of a Gaussian process regression predicting the future trajectory of an unknown target in intricate dense 3D-environments based on sampling-rejection principle for exploration.

This package utilizes the C++ libraries roscpp, Eigen, octomap as well es dynamicEDT-3D:

- eigen3-library (C++)
- ros-melodic-octomap
- ros-melodic-octomap-msgs
- dynamicEDT3D

sudo make install for dynamicEDT3D (non-ROS package)

ros-melodic-octomap-mapping ros-melodic-octomap-ros ros-melodic-octomap-server

Currently the received message /target_pose is not used at all. The bool lost_track is always true in line 475 gauss_traj_estimator.cpp file. Working on to change it, maybe adding the latest mssg to training data would work.

/sampled_pred_paths -> random purple lines, lots of them
/valid_sampled_pred_paths -> blue lines, many of them, should be a subset of purple lines, sometimes invalid paths may occur, should be fixed
/valid_pred_path_mean -> one thick green line
/target_pred_path_mean -> one purple line, pass through the pre-known waypoints
/evaltd_training_points -> set of training points in purple
/valid_pred_path_cov_pos -> one light green line
/valid_pred_path_cov_neg -> one light green line
