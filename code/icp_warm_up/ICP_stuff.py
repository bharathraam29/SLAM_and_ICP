from sklearn.neighbors import NearestNeighbors
import transforms3d as t3d
from tqdm import tqdm
import numpy as np

def get_pose(init_R, init_P):
    init_T= np.zeros([4,4])
    init_T[:3,:3]= init_R
    init_T[:, -1]= np.hstack((init_P, np.array([1])))
    return init_T

def get_R_and_P(T):
    return T[:3,:3], T[:-1,-1]

def get_pc_centroid(pc):
    return pc.mean(axis=0)
    
def get_corres_points(target_pc_moved, source_pc):
    kn_obj=NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(target_pc_moved)
    dist, idx= kn_obj.kneighbors(source_pc)
    return np.sum(dist.flatten()), idx.flatten()

def rotate_pc(R, pc):
    return (R@ pc.T).T

def ICP_warm_up(source_pc, target_pc,number_of_yaw_splits=4, viz= False):
    yaw_split= (2*np.pi)/number_of_yaw_splits
    max_iters= 100
    filter_dist_threshold= 0.5

    sampled_target_pc= target_pc[::5]
    sampled_source_pc= source_pc[::5]
    euclidean_dist= []
    best_pose=[]

    for ii in range(number_of_yaw_splits):
        yaw_val= yaw_split*ii
        print('\n\n###########################\n\n')
        print("YAW_VALUE :: ", yaw_val)
        init_R= t3d.euler.euler2mat(0,0, yaw_val)
        init_p=  get_pc_centroid(source_pc)- get_pc_centroid(rotate_pc(init_R,target_pc) )
        # print("Initial R: \n",init_R)
        # print("Initial P: ",init_p)
        init_T= get_pose(init_R, init_p)
        print("Initial Pose:\n ",init_T)
        

        moved_target_pc= rotate_pc(init_R, sampled_target_pc)+init_p
        prev_euclidean_dist= 0
        accumulated_pose = init_T
        dist, corres_idxs = get_corres_points(moved_target_pc,sampled_source_pc)
        # print(dist)
        dist_between_corresponding_pc_pts= np.linalg.norm(sampled_source_pc-moved_target_pc[corres_idxs],axis=1)
        
        filtered_idx = np.where(dist_between_corresponding_pc_pts < filter_dist_threshold)
        
        sampled_source_pc=sampled_source_pc[filtered_idx]
        moved_target_pc=moved_target_pc[corres_idxs][filtered_idx]
        for itr in tqdm(range(max_iters)):
            dist, corres_idxs = get_corres_points(moved_target_pc,sampled_source_pc)

            #Kabasch stuff
            src_centroid = get_pc_centroid(sampled_source_pc)
            tar_centroid = get_pc_centroid(moved_target_pc)
        
            centered_sampled_source_pc= sampled_source_pc - src_centroid
            centered_moved_target_pc= moved_target_pc- tar_centroid
        
            Q= np.matmul(centered_sampled_source_pc.transpose(), centered_moved_target_pc[corres_idxs])
            U, S, Vt = np.linalg.svd(Q, full_matrices=True)
            R= U@Vt
            p= src_centroid - rotate_pc(R, tar_centroid)

            pred_pose= get_pose(R,p)
            accumulated_pose= pred_pose@accumulated_pose

            moved_target_pc = rotate_pc( R, moved_target_pc) + p
            if np.abs(dist-prev_euclidean_dist) < 1e-10:
                best_dist= dist
                break
            prev_euclidean_dist = dist
        print(accumulated_pose, best_dist)
        euclidean_dist.append(best_dist)
        best_pose.append(accumulated_pose)

    best_pose_pred= best_pose[np.argmin(euclidean_dist)]
    if viz:
        visualize_icp_result(target_pc, source_pc, best_pose_pred)
    return best_pose_pred



def ICP(source_pc, target_pc,init_T):

    max_iters= 100
    filter_dist_threshold= 0.5

    sampled_target_pc= target_pc[::]
    sampled_source_pc= source_pc[::]
    euclidean_dist= []
    best_pose=[]


    # print("Initial Pose:\n ",init_T)
    init_R, init_p= get_R_and_P(init_T)
    

    moved_target_pc= rotate_pc(init_R, sampled_target_pc)+init_p
    prev_euclidean_dist= 0
    accumulated_pose = init_T
    dist, corres_idxs = get_corres_points(moved_target_pc,sampled_source_pc)
    # print(dist)
    dist_between_corresponding_pc_pts= np.linalg.norm(sampled_source_pc-moved_target_pc[corres_idxs],axis=1)
    
    filtered_idx = np.where(dist_between_corresponding_pc_pts < filter_dist_threshold)
    
    sampled_source_pc=sampled_source_pc[filtered_idx]
    # moved_target_pc=moved_target_pc[corres_idxs][filtered_idx]
    moved_target_pc = moved_target_pc[corres_idxs][filtered_idx]
    for itr in (range(max_iters)):
        dist, corres_idxs = get_corres_points(moved_target_pc,sampled_source_pc)

        #Kabasch stuff
        src_centroid = get_pc_centroid(sampled_source_pc)
        tar_centroid = get_pc_centroid(moved_target_pc)
    
        centered_sampled_source_pc= sampled_source_pc - src_centroid
        centered_moved_target_pc= moved_target_pc- tar_centroid
    
        Q= np.matmul(centered_sampled_source_pc.transpose(), centered_moved_target_pc[corres_idxs])
        U, S, Vt = np.linalg.svd(Q, full_matrices=True)
        R= U@np.diag([1,1,np.linalg.det(U@Vt)])@Vt
        
        p= src_centroid - rotate_pc(R, tar_centroid)

        pred_pose= get_pose(R,p)
        accumulated_pose= pred_pose@accumulated_pose

        moved_target_pc = rotate_pc( R, moved_target_pc) + p
        if np.abs(dist-prev_euclidean_dist) < 1e-10:
            best_dist= dist
            best_pose = accumulated_pose
            return best_pose
            break
        prev_euclidean_dist = dist
        # print(dist)
    # print(accumulated_pose, best_dist)
    euclidean_dist.append(best_dist)
    best_pose.append(accumulated_pose)
    return accumulated_pose
