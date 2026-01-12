# StickyVO

A high-performance Visual Odometry (VO) system using **LightGlueStick** for robust feature matching and tracking, integrated with **ROS 2**.


## Prerequisites

- Ubuntu 22.04+
- ROS 2 (Humble / Iron / Jazzy)
- PyTorch (with CUDA support)
- OpenCV
- Eigen
- Ceres

## Installation

### Clone the Repository

```bash
git clone <your-repo-url>
cd StickyVO
```

### Set Up Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Build ROS 2 Workspace

```bash
colcon build --symlink-install
```

## Usage

### Source the Workspace

```bash
source install/setup.bash
```

### Launch the VO Node

```bash
ros2 launch stickyvo_ros stickyvo.launch.py
```