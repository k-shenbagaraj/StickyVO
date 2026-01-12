from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('stickyvo_ros')
    params_file = os.path.join(pkg_share, 'config', 'blackfly-5mm.yaml')

    return LaunchDescription([
        Node(
            package='stickyvo_ros',
            executable='stickyvo_node',
            name='stickyvo_node',
            output='screen',
            parameters=[params_file],
        )
    ])
