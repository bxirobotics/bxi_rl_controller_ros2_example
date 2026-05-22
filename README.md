# bxi_rl_controller_ros2_example

## Overview
This repository contains a framework of developing controllers for BXI robots, including:
* A sample controller program that deploys reinforcement learning based control policies;      
* A Mujoco simulator based on ROS2;    
* The BXI Hardware envrionment based on ROS2;  
* A ROS2 node of remote controller taking joystick/keyboard input;   

## Package structure    
The binary ROS2 packages of ROS2 environment and mujoco can be find here:[`bxi_ros2_pkg`](https://github.com/bxirobotics/bxi_ros2_pkg)      
*  In the binary ROS2 package `bxi_ros2_pkg/` directory：
1. `communication`：the robot communication package, including custom communication package formats.    
2. `description`: contains the robot description files, including urdf, xml and the meshe files.    
3. `mujoco`: Mujoco simulator based on ROS2. Controller programs a recommened to be verified in Mujoco before deploy to the robot hardware.    
4. `hardware`: The robot hardware package. This node publishes all sensor data of the robot and receives control commands.     
5. `hardware_arm`: The hardware control package for the upper-body-only version of the robot. This node publishes information about the robot's upper-body arms and receives control commands.    
    
The [`bxi_ros2_pkg`](https://github.com/bxirobotics/bxi_ros2_pkg) is a unified framework for both simulation and hardare:    
ROS2 structure simulation:    ![ROS2 structure simulation](docs/ROS2_structure_simulation.png)    
ROS2 structure hardware:    ![ROS2 structure hardware](docs/ROS2_structure_hardware.png)

* The `src/` directory :         
1. `src/bix_example_py_elf3`: a demo of learming based control policy elf3.    
2. `remote_controller`: reads the joystick/keyboard input and publish the commands. Work with both the robot hardware and the simulation environment.   

## Descripiton Files(URDF)    
1. `elf3_dof29` : Elf3 of dof29
2. `elf3_dof31` : Elf3 of dof31 (2 on head)    
For USD or XML format please refer to: [unofficical models](https://github.com/MelodyAI/TienKung-Lab-bxi/tree/main/legged_lab/assets/elf3_lite)

## Instructions    
### 中文扩展文档 / Wiki
新增遥控器/输入控制器、添加机器人业务状态、配置状态转移、过渡行为、`on_bind(ctx)` 状态订阅和 `get_cmd_vel(ctx)` 速度处理，请参考项目 Wiki：

- GitHub Wiki：<https://github.com/bxirobotics/bxi_rl_controller_ros2_example/wiki>

### Switch between hardware and simulation environment
1. `hw` is short for`hardware`，all `launch` files with suffix `hw` are to launch real hardware. **Please use them carefully**.      
2. The simulation environment and the robot hardware share the same control program. You only need to apply different launch files to switch between simulation and hardware. Topics for simulation code are with the `simulation/` prefix, while topics for the hardware are with the `hardware/` prefix. For details, please refer to the topic parameter settings in src/example.    
3. The robot in the simulation environment is initialized with a virtual suspension. After startup, the suspension needs to be released. While suspension-related signals are ignored when operating on robot hardware).     
4. There is a global odometer topic `odm` in the simulation env, while this topic is not available in `hardware` environment.     

### System Environment Setup   
1. `Ubuntu 22.04`，with ROS2 version `humble`. `mujoco` requires `libglfw3-dev`.       
2. Copy `./script/bxi-dev.rules` to `/etc/udev/rules.d/`
3. To set up remote controller auto-start, edit `./script/ros_elf_launch.service`, copy to `/etc/systemd/system/`, and used the `systemctl` tool to enable the auto-start service.  

### Running a demo control program in simulator
1. Pull the ROS2 binary packages[`bxi_ros2_pkg`](https://github.com/bxirobotics/bxi_ros2_pkg) to /opt/bxi/bxi_ros2_pkg :    
```
mkdir /opt/bxi/ ; cd /opt/bxi/
git clone https://github.com/bxirobotics/bxi_ros2_pkg.git
```     
then activate it(**Run it as `root` on robot hardware**):    
```
source /opt/bxi/bxi_ros2_pkg/setup.bash
```              
2. In bxi_rl_controller_ros2_example directory, run `bash build.sh` to compile all sources in `./src` director. When compilation is done，run `source ./install/setup.bash` to activate the environment of current packages.        
3. Run whole body control policy：
`ros2 launch bxi_example_py_elf3 example_launch_demo.py` : start simulation + controller program(learning based)    
4. Bring up keyboard control node:     
`ros2 launch remote_controller remote_conroller_launch_keyboard.py'    

### Running a demo control program in hardware
0. Login as root;
1. Pull the ROS2 binary packages[`bxi_ros2_pkg`](https://github.com/bxirobotics/bxi_ros2_pkg) to /opt/bxi/bxi_ros2_pkg :    
```
mkdir /opt/bxi/ ; cd /opt/bxi/
git clone https://github.com/bxirobotics/bxi_ros2_pkg.git
```     
then activate it(**Run it as `root` on robot hardware**):    
```
source /opt/bxi/bxi_ros2_pkg/setup.bash
```              
2. In bxi_rl_controller_ros2_example directory, run `bash build.sh` to compile all sources in `./src` director. When compilation is done，run `source ./install/setup.bash` to activate the environment of current packages.   
3. Run whole body control policy：    
`ros2 launch bxi_example_py_elf3 example_launch_demo_hw.py` start robot hardware + control policy  

### Tips for running a control program
1. The control commands in the topic must be sent in the specified joint order. The order of joints refer to the example `src/bix_example`    
2. Both the simulation environment and the hardware robot have an out-of-control protection. The protection is triggered if control commands are lost for more than 100ms. Once triggered, the motors will be disabled, and the system must be reinitialized before it can be used again. 

### Startup Process
In both simulation and hardware, the motors are in a disabled state when started, and all parameters are uncontrollable. The startup process consists of two steps:
1. Enable position control of the motors. The motors can implement position control by setting three parameters:`pos kp kd`. As of Elf2 hardware, when received the 1st `reset` command, it's joints rotate to zero position and keep it for 10 seconds.              
2. Enable all control parameters. The motors can be set with `pos vel tor kp kd`. As of Elf2 hardware, when received the 2nd `reset` command, it's joints start taking control input.       
For startup examples, please refer to src/bxi_example_py.   

### Hardware Protection
In addition to communication timeout protection, the hardware node also includes torque protection, overspeed protection, and position protection.
1. There is an error counter built into the hardware node. When the error count reaches `1000`, the motor will exit the enabled state.     
2. The error counter logic: increases the error count by `50` if receive a motor speed overrun ; increases the error count by `100` if receive a torque overrun ; decreases the error count by `1` if receive normal motor messages, with a minimum value of `0`.     
3. When the position overrun protection is triggered, the error count is not increased. The overrun direction control will be disabled, the motor can only rotate in the opposite direction.     
4. Please contact us to get the detailed overrun values. It is not recommended to modify them unless necessary.     

## Important Notes
Large-sized robots may pose risks. Check instructions carefully before operation!     
All control programs must go through simulation before deploying on robot hardwares.     
Press the stop button immediately if any abnormality occurs!      
