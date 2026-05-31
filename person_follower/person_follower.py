# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import rclpy
import cv2
import numpy as np
import struct
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data

class Cluster:
    def __init__(self, indices, centroid):
        self.indices = indices
        self.centroid = centroid
        self.grayscale_value = self._compute_grayscale()

    def _compute_grayscale(self):
        gray = int(math.floor(self.centroid * 24.962 - 2.042))
        return max(0, min(255, gray))

class PersonFollower(Node):
    def __init__(self):
        super().__init__('person_follower')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_lidar,
            qos_profile_sensor_data
        )

        self.linear_k = 0.55
        self.angular_k = 2.0
        self.distance = 1.5
        self.stop = 1.0
        self.max_vx = 0.15
        self.max_wz = 0.8
        
        # LiDAR values
        self.max_dist = 2.5                     # maximum detection distance
        self.start = -75                        # start vision angle
        self.end = 75                           # end vision angle
        self.min_cluster_size = 3               # min number of points in a cluster
        self.cluster_distance_threshold = 0.3   # max distance between cluster points    

        self.depth_image = []
        self.lidar_clusters = []

        self.timer = self.create_timer(1.0/10.0, self.process_loop)

        self.bridge = CvBridge()
        video_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.im_sub = self.create_subscription(
            CompressedImage, 
            '/astra/depth/image_raw/compressedDepth', 
            self.listener_astra,
            video_qos
        )

    def listener_astra(self, msg):
        depth_fmt, compr_type = msg.format.split(';')
        depth_fmt = depth_fmt.strip()
        compr_type = compr_type.strip()

        raw_data = msg.data[12:]
        depth_img_raw = cv2.imdecode(np.frombuffer(raw_data, np.uint8), cv2.IMREAD_UNCHANGED)

        if depth_img_raw is None:
            return

        if depth_fmt == '16UC1':
            depth_vis = cv2.normalize(depth_img_raw, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)

        elif depth_fmt == '32FC1':
            raw_header = msg.data[:12]
            _, depthQuantA, depthQuantB = struct.unpack('iff', raw_header)
            depth_img = depthQuantA / (depth_img_raw.astype(np.float32) - depthQuantB)
            depth_img[depth_img_raw == 0] = 0
            depth_vis = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)

        else:
            return

        lower_row = 50
        upper_row = 50
        h, w = depth_vis.shape

        depth_vis_cropped = depth_vis[upper_row:h-lower_row, 0:w]

        self.depth_image = depth_vis_cropped.copy()

    def listener_lidar(self, input_msg):
        angle_min = input_msg.angle_min
        angle_increment = input_msg.angle_increment
        ranges = list(input_msg.ranges)

        valid_indices = []
        valid_ranges = []

        start_angle = math.radians(-75)
        end_angle = math.radians(75)        

        for i, r in enumerate(ranges):
            angle = angle_min + i * angle_increment

            if angle < start_angle or angle > end_angle:
                continue

            if math.isfinite(r) and r <= self.max_dist:
                valid_indices.append(i)
                valid_ranges.append(r)

        clusters = []
        centroids = []

        if len(valid_ranges) < self.min_cluster_size:
            return

        current_cluster = [valid_indices[0]]
        prev_theta = angle_min + valid_indices[0] * angle_increment
        prev_x = valid_ranges[0] * math.cos(prev_theta)
        prev_y = valid_ranges[0] * math.sin(prev_theta)

        for i in range(1, len(valid_ranges)):
            theta = angle_min + valid_indices[i] * angle_increment
            x = valid_ranges[i] * math.cos(theta)
            y = valid_ranges[i] * math.sin(theta)

            euclidean_dist = math.sqrt((x - prev_x)**2 + (y - prev_y)**2)

            if euclidean_dist > self.cluster_distance_threshold:
                if len(current_cluster) >= self.min_cluster_size:
                    cluster_ranges = [valid_ranges[j] for j in range(len(valid_ranges)) if valid_indices[j] in current_cluster]
                    centroid = np.mean(cluster_ranges)

                    clusters.append(current_cluster)
                    centroids.append(centroid)

                current_cluster = [valid_indices[i]]
                prev_x, prev_y = x, y
                prev_theta = theta
            else:
                current_cluster.append(valid_indices[i])
                prev_x, prev_y = x, y
                prev_theta = theta

        if len(current_cluster) >= self.min_cluster_size:
            cluster_ranges = [valid_ranges[j] for j in range(len(valid_ranges)) if valid_indices[j] in current_cluster]
            centroid = np.mean(cluster_ranges)

            clusters.append(current_cluster)
            centroids.append(centroid)

        #print(f"Found {len(clusters)} clusters with centroids: {centroids}")

        self.lidar_clusters = []
        for cluster_indices in clusters:
            cluster_ranges = [valid_ranges[j] for j in range(len(valid_ranges)) if valid_indices[j] in cluster_indices]
            centroid = np.mean(cluster_ranges)
            cluster_obj = Cluster(cluster_indices, centroid)
            center_index = cluster_indices[len(cluster_indices) // 2]
            cluster_obj.avg_angle = center_index * angle_increment
            self.lidar_clusters.append(cluster_obj)
        
        vx = 0.0
        wz = 0.0
        """
        if best_range is not None:
            error_distance = best_range - self.distance
            error_angle = (best_i * angle_increment)

            vx = self.linear_k * error_distance
            wz = self.angular_k * error_angle

            vx = max(min(vx, self.max_vx), -self.max_vx)
            wz = max(min(wz, self.max_wz), -self.max_wz)

            if best_range < self.stop:
                vx = 0.0
                wz = 0.0

        """
            
        #self.move_robot(vx, wz)
    
    def process_loop(self):
        if len(self.depth_image) != 0 and len(self.lidar_clusters) != 0:
            best_cluster = None
            best_score = -1  

            for cluster in self.lidar_clusters:
                score, metrics = self.score_person_from_segmentation(cluster)

                print(f"Cluster at {cluster.centroid:.2f}m: score={score:.1f}, area={metrics['area']}, aspect_ratio={metrics['aspect_ratio']:.2f}")

                if score > best_score:
                    best_score = score
                    best_cluster = cluster
                
            if best_cluster is not None and best_score > 60:  # threshold
                print(f"✓ Following cluster with score {best_score:.1f}")
                #self.follow_cluster(best_cluster)
            else:
                print(f"✗ No person-like cluster found (best score: {best_score:.1f})")
                self.move_robot(0.0, 0.0)
        else:
            self.move_robot(0.0, 0.0)

    def score_person_from_segmentation(self, cluster):
        """Score how person-like this cluster is based on segmented depth image"""
        
        # Create segmented region for this cluster's distance
        lower_gray = cluster.grayscale_value - 10
        upper_gray = cluster.grayscale_value + 10
        segmented = cv2.inRange(self.depth_image, lower_gray, upper_gray)

        cv2.imshow("Segmented", segmented)
        cv2.waitKey(1) 
        
        # Find contours
        contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0, {'area': 0, 'aspect_ratio': 0, 'solidity': 0}
        
        # Get largest contour (assume it's the main object)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 200:  # too small, likely noise
            return 0, {'area': area, 'aspect_ratio': 0, 'solidity': 0}
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        if w == 0 or h == 0:
            return 0, {'area': area, 'aspect_ratio': 0, 'solidity': 0}
        
        # Calculate aspect ratio (height/width)
        aspect_ratio = h / w
        
        # Calculate solidity (area / convex_hull_area)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Calculate extent (area / bounding_rect_area)
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        # Score based on person-like properties
        score = 0
        
        if cluster.centroid < 2.0:
            # Close range - person appears wider
            if 1.0 <= aspect_ratio <= 2.0:
                score += 40
            elif 0.8 <= aspect_ratio <= 2.5:
                score += 25
            elif 0.6 <= aspect_ratio <= 3.0:
                score += 10
        elif cluster.centroid < 2.5:
            # Medium range
            if 1.5 <= aspect_ratio <= 3.0:
                score += 40
            elif 1.2 <= aspect_ratio <= 3.5:
                score += 25
            elif 1.0 <= aspect_ratio <= 4.0:
                score += 10
        else:
            # Far range
            if 2.0 <= aspect_ratio <= 4.0:
                score += 40
            elif 1.5 <= aspect_ratio <= 4.5:
                score += 25
            elif 1.2 <= aspect_ratio <= 5.0:
                score += 10
        
        if cluster.centroid < 2.0:
            if 10000 <= area <= 50000:
                score += 25
            elif 5000 <= area <= 60000:
                score += 15
        elif cluster.centroid < 2.5:
            if 5000 <= area <= 35000:
                score += 25
            elif 3000 <= area <= 40000:
                score += 15
        else:
            if 3000 <= area <= 15000:
                score += 25
            elif 2000 <= area <= 20000:
                score += 15
        
        # 3. Solidity score (people have some gaps but mostly solid)
        if 0.4 <= solidity <= 0.9:
            score += 20
        elif 0.3 <= solidity <= 0.95:
            score += 10
        
        # 4. Extent score (people don't fill entire bounding box perfectly)
        if 0.35 <= extent <= 0.75:
            score += 15
        elif 0.25 <= extent <= 0.85:
            score += 8
        
        metrics = {
            'area': area,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'extent': extent,
            'width': w,
            'height': h
        }
        
        return score, metrics

    def follow_cluster(self, cluster):
        error_distance = cluster.centroid - self.distance
        error_angle = cluster.avg_angle
        
        vx = self.linear_k * error_distance
        wz = self.angular_k * error_angle
        
        vx = max(min(vx, self.max_vx), -self.max_vx)
        wz = max(min(wz, self.max_wz), -self.max_wz)
        
        if cluster.centroid < self.stop:
            vx = 0.0
            wz = 0.0
        
        self.move_robot(vx, wz)

    def move_robot(self, vx, wz):
        output_msg = Twist()
        output_msg.linear.x = vx
        output_msg.angular.z = wz
        self.publisher_.publish(output_msg)

def main(args=None):
    rclpy.init(args=args)
    person_follower = PersonFollower()
    rclpy.spin(person_follower)
    person_follower.destroy_node()
    rclpy.shutdown()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
