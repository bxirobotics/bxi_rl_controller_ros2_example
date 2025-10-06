# bxi_rl_controller_ros2_example

## Introduction
This repository contains a framework of developing controllers for BXI robots, including:
* A reinforcement learning based sample controller program;    
* A Mujoco simulator based on ROS2;    
* The BXI Hardware envrionment based on ROS2    
The binary ROS2 packages of ROS2 environment and mujoco can be find here:[`bxi_ros2_pkg`](https://github.com/bxirobotics/bxi_ros2_pkg)      
*  The binary ROS2 package `bxi_ros2_pkg/` directory：
1. `communication`：the robot communication package, including custom communication package formats.    
2. `description`: contains the robot description files, including urdf, xml and the meshe files.    
3. `mujoco`: Mujoco simulator based on ROS2. Controller programs a recommened to be verified in Mujoco before deploy to the robot hardware.    
4. `hardware`: The robot hardware package. This node publishes all sensor data of the robot and receives control commands.     
5. `hardware_arm`: The hardware control package for the upper-body-only version of the robot. This node publishes information about the robot's upper-body arms and receives control commands.
* Control program `src/` directory：：
1. `src/bix_example`: demo of the initialization process and basic message receiving and sending functions.    
2. `src/bix_example_py`: a demo of learming based control policy.    
3. `remote_controller`: reads the Xbox/PS input and publish the commands. Work with both the robot hardware and the simulation environment.   
4. `src/bix_example_py_arm`: a demo of upper body control.     

## Usage
### Descripiton Files
1. `elf2-trunk`:Elf2 v1，`base link` is `trunk`(torso)，elf2-trunk_dof12: only 12 leg joints are free joints. dof25: all joints are free.
2. `elf2-ankle`:Elf2 v2: ankles torques are increased `50Nm`.
3. `elf2-foot`: Elf2 v3: sole changed: 1 shaping to human foot for wearing shoes. 2 symmetric soles change to asymmetric, like human feet.(deprecated)
4. `elf2-footv4`: Elf2 v4: sole change : sole shape change back to regular rectangle with oval ends. Asymmetric sole.   
5. `elf2-arm`: arms only.    
6. `elf3_prev_dof20` : Elf3 preview version with dof20.

### Switch between hardware and simulation environment
1. `hw` is short for`hardware`，all `launch` files with suffix `hw` are to launch real hardware. Please use them carefully.      
2. The simulation environment and the robot hardware share the same control program. You only need to apply different launch files to switch between simulation and hardware. Topics for simulation code are with the `simulation/` prefix, while topics for the hardware are with the `hardware/` prefix. For details, please refer to the topic parameter settings in src/example.    
3. The robot in the simulation environment is initialized with a virtual suspension. After startup, the suspension needs to be released. While suspension-related signals are ignored when operating on robot hardware).     
4. There is a global odometer topic `odm` in the simulation env, while this topic is not available in `hardware` environment.    
5. There is `touch_sensor` in the simulation env, while the real foot touch sensor is under developement. Although `hardware` publishes touch forces, they are rough estimates. For higher precision requirements, estimation can be performed using the ground contact state estimation algorithm in quadruped robots.       

### System Environment Setup   
1. `Ubuntu 22.04`，with ROS2 version `humble`. `mujoco` requires `libglfw3-dev`.       
2. Copy `./script/bxi-dev.rules` to `/etc/udev/rules.d/`
3. To set up remote controller auto-start, edit `./script/ros_elf_launch.service`, copy to `/etc/systemd/system/`, and used the `systemctl` tool to enable the auto-start service.  

### Startup Process
In both simulation and hardware, the motors are in a disabled state when started, and all parameters are uncontrollable. The startup process consists of two steps:
1. Enable position control of the motors. The motors can implement position control by setting three parameters:`pos kp kd`,    
2. Enable all control parameters. The motors can be set with `pos vel tor kp kd`,    
For startup examples, please refer to src/bxi_example_py.    

### Running a demo control program 
1. Copy the ROS2 binary packages[`bxi_ros2_pkg`](https://github.com/bxirobotics/bxi_ros2_pkg) to /opt/bxi/bxi_ros2_pkg , activate it：
   `source /opt/bxi/bxi_ros2_pkg/setup.bash` . Run it as `root` on robot hardware.         
2. In bxi_rl_controller_ros2_example directory, run `colcon build` to compile all sources in `./src` director. When compilation is done，run `source ./install/setup.bash` to activate the environment of current packages.        
3. Run whole body demos：
* `ros2 launch bxi_example_py example_launch.py` : start simulation + controller program(learning based)        
* `ros2 launch bxi_example_py example_launch_hw.py` start robot hardware + control policy 
4. Run upper body control demos：
* `ros2 launch bxi_example_arm example_launch.py`: start simulation + upper body controller program(C++ version) 
* `ros2 launch bxi_example_arm example_launch_hw.py`: start robot hardware + upper body controller program(C++ version) 
* `ros2 launch bxi_example_py_arm example_launch.py`: start simulation + upper body controller program(python version) 
* `ros2 launch bxi_example_py_arm example_launch_hw.py`: start robot hardware + upper body controller program(python version)     

### Tips for control program
1. The control commands in the topic must be sent in the specified joint order. The order of joints refer to the example `src/bix_example`    
2. Both the simulation environment and the hardware robot have an out-of-control protection. The protection is triggered if control commands are lost for more than 100ms. Once triggered, the motors will be disabled, and the system must be reinitialized before it can be used again. 

### Hardware Protection
In addition to communication timeout protection, the hardware node also includes torque protection, overspeed protection, and position protection.
1. There is an error counter built into the hardware node. When the error count reaches `1000`, the motor will exit the enabled state.     
2. The error counter logic: increases the error count by `50` if receive a motor speed overrun ; increases the error count by `100` if receive a torque overrun ; decreases the error count by `1` if receive normal motor messages, with a minimum value of `0`.     
3. When the position overrun protection is triggered, the error count is not increased. The overrun direction control will be disabled, the motor can only rotate in the opposite direction.     
4. Please contact us to get the detailed overrun values. It is not recommended to modify them unless necessary.     

## Notes
Large-sized robots may pose risks. Check instructions carefully before operation!     
All control programs must go through simulation before deploying on robot hardwares.     
Press the stop button immediately if any abnormality occurs!      

