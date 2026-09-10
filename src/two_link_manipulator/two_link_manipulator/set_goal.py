import sys
import rclpy
from rclpy.node import Node
from custom_interfaces.srv import Position2D

"""
USAGE IN CLI: ros2 run two_link_manipulator set_goal_node 'x' 'y' 
"""

class SetGoalNode(Node):
    def __init__(self):
        super().__init__('set_goal_node')
        self.cli = self.create_client(Position2D, 'set_goal')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Position2D.Request()

    def send_request(self, x, y):
        self.req.x = x
        self.req.y = y
        return self.cli.call_async(self.req)

def main():
    rclpy.init()

    node = SetGoalNode()
    future = node.send_request(float(sys.argv[1]), float(sys.argv[2]))
    rclpy.spin_until_future_complete(node, future)
    response = future.result()

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




