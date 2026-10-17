# Message timestamp and coordinate frame for this candidate.
std_msgs/Header header

# Candidate index in the current response.
uint32 candidate_id

# Matched peduncle instance id from InstanceMask.instance_id.
uint32 peduncle_id

# Matched tomato cluster instance id from InstanceMask.instance_id.
uint32 cluster_id

# Matched single-fruit instance ids from InstanceMask.instance_id.
uint32[] fruit_ids

# Pick point image coordinate u in pixels.
float32 image_u

# Pick point image coordinate v in pixels.
float32 image_v

# Pick point depth in meters.
float32 depth

# 6D pose in the camera frame: [x, y, z, roll, pitch, yaw].
# pose_xyzrpy_camera[0]: x position in meters.
# pose_xyzrpy_camera[1]: y position in meters.
# pose_xyzrpy_camera[2]: z position in meters.
# pose_xyzrpy_camera[3]: roll angle in radians.
# pose_xyzrpy_camera[4]: pitch angle in radians.
# pose_xyzrpy_camera[5]: yaw angle in radians.
# Position is in meters. Roll, pitch, and yaw are in radians.
float32[6] pose_xyzrpy_camera

# 6D pose in the robot base frame: [x, y, z, roll, pitch, yaw].
# pose_xyzrpy_base[0]: x position in meters.
# pose_xyzrpy_base[1]: y position in meters.
# pose_xyzrpy_base[2]: z position in meters.
# pose_xyzrpy_base[3]: roll angle in radians.
# pose_xyzrpy_base[4]: pitch angle in radians.
# pose_xyzrpy_base[5]: yaw angle in radians.
# Position is in meters. Roll, pitch, and yaw are in radians.
# This field is valid only when has_pose_base is true.
float32[6] pose_xyzrpy_base

# Standard ROS pose in the camera frame, kept for RViz, TF, and Marker visualization.
geometry_msgs/PoseStamped pose_camera

# Standard ROS pose in the robot base frame.
# This field is valid only when has_pose_base is true.
geometry_msgs/PoseStamped pose_base

# True when pose_xyzrpy_base and pose_base are valid.
bool has_pose_base

# Candidate quality score. Higher is better.
float32 score

# Cluster maturity label, such as unknown, green, turning, or ripe.
string maturity

# Estimated ripe ratio of the associated cluster, range 0.0 to 1.0.
float32 ripe_ratio

# True when the peduncle or cluster touches the crop/image border.
bool truncated_by_crop

# Human-readable debug reason, including pick point, depth, and orientation sources.
string reason
