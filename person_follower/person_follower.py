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
        qos_policy = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)

        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            qos_profile=qos_policy)

        self.linear_k = 0.55
        self.angular_k = 2.4
        self.distance = 0.5
        self.stop = 0.4
        self.max_vx = 0.15
        self.max_wz = 0.8
        self.max_dist = 2.0
        self.start = -15
        self.end = 15

    def listener_callback(self, input_msg):
        angle_min = input_msg.angle_min
        angle_increment = input_msg.angle_increment
        ranges = list(input_msg.ranges)

        best_range = None
        best_i = None

        for i in range(self.start, self.end):
            r = ranges[i]
            if not math.isfinite(r) or r > self.max_dist:
                continue
            if best_range is None or r < best_range:
                best_range = r
                best_i = i

        vx = 0.0
        wz = 0.0

        if best_range is not None:
            error_distance = best_range - self.distance
            error_angle = (best_i * angle_increment + angle_min)

            vx = self.linear_k * error_distance
            wz = self.angular_k * error_angle

            vx = max(min(vx, self.max_vx), -self.max_vx)
            wz = max(min(wz, self.max_wz), -self.max_wz)

            if best_range < self.stop:
                vx = 0.0
                wz = 0.0

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


if __name__ == '__main__':
    main()
