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
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class PersonFollower(Node):

    def __init__(self):
        super().__init__('person_follower')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.last_target_angle = None
        self.last_target_distance = None
        self.target_timeout = 0

    def listener_callback(self, input_msg):
        angle_min = input_msg.angle_min
        angle_increment = input_msg.angle_increment
        ranges = input_msg.ranges

        CLUSTER_DIST = 0.12
        MIN_CLUSTER_SIZE = 4
        MAX_CLUSTER_SIZE = 15

        TARGET_DISTANCE = 1.0

        SAFE_DISTANCE = 0.35

        MAX_LINEAR_SPEED = 0.5
        MAX_ANGULAR_SPEED = 1.2

        clusters = []
        current_cluster = []

        for i, r in enumerate(ranges):
            if r == math.inf:
                if current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []
                continue

            angle = angle_min + i * angle_increment

            if abs(angle) > math.radians(30):
                if current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []
                continue

            if not current_cluster:
                current_cluster.append((r, angle))
            else:
                if abs(r - current_cluster[-1][0]) < CLUSTER_DIST:
                    current_cluster.append((r, angle))
                else:
                    clusters.append(current_cluster)
                    current_cluster = [(r, angle)]

        if current_cluster:
            clusters.append(current_cluster)

        best_cluster = None
        best_score = math.inf

        for cluster in clusters:
            if not (MIN_CLUSTER_SIZE <= len(cluster) <= MAX_CLUSTER_SIZE):
                continue

            avg_r = sum(p[0] for p in cluster) / len(cluster)
            avg_angle = sum(p[1] for p in cluster) / len(cluster)

            if self.last_target_angle is not None:
                angle_diff = abs(avg_angle - self.last_target_angle)
            else:
                angle_diff = 0.0

            score = avg_r + 0.5 * angle_diff

            if score < best_score:
                best_score = score
                best_cluster = (avg_r, avg_angle)

        if best_cluster is None:
            self.target_timeout += 1
            if self.target_timeout > 10:
                self.last_target_angle = None
            return

        target_distance, target_angle = best_cluster
        self.last_target_angle = target_angle
        self.last_target_distance = target_distance
        self.target_timeout = 0

        front_ranges = [
            r for i, r in enumerate(ranges)
            if r != math.inf and abs(angle_min + i * angle_increment) < math.radians(8)
        ]

        output = Twist()

        if front_ranges and min(front_ranges) < SAFE_DISTANCE:
            # Too close to a wall → rotate away
            output.linear.x = 0.0
            output.angular.z = 0.6
        else:
            vx = 0.6 * (target_distance - TARGET_DISTANCE)
            wz = 1.5 * target_angle

            vx = max(min(vx, MAX_LINEAR_SPEED), 0.0)
            wz = max(min(wz, MAX_ANGULAR_SPEED), -MAX_ANGULAR_SPEED)

            output.linear.x = vx
            output.angular.z = wz

        self.publisher_.publish(output)

def main(args=None):
    rclpy.init(args=args)
    person_follower = PersonFollower()
    rclpy.spin(person_follower)
    person_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
