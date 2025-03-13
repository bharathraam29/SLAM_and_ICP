# LiDAR SLAM and ICP implementation
This repository implements LiDAR SLAM, and utilizes Factor graph optimization to introduces loop closure constraints based on proximity at fixed intervals. 
Occupancy grid is also constructedd using logodds update, and a razterized version of the occupancy map is also constructed using the RBGD images. 

***Check LIDAR SLAM Report.pdf to know more about the implementation and problem formulation*
**
The steps to use the repo is as below.

Final runner.ipynb : Is the notebook to run the Lidar SLAM. set the "dataset_idx" in the top of the notebook and you can run all the cells. The trajectories, occupancy and textured grid map will be displayed as inline outputs

ICP_stuff.py : has all the implementations involving ICP.

icp_playground_v2.ipynb : Run the first 2 cells setting the dataset you wish to see the ICP warmup out for. Change the number_of_yaw_splits to higher value to get more yaw angle discretization.  
