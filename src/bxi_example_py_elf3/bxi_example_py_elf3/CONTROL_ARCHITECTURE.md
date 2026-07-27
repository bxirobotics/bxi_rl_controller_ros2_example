# Elf3 测试控制结构

## 公共模块

```text
control/elf3.py       29 轴唯一顺序、名义姿态、PD 参数、软件位置限制
control/remote.py     C++ 翻转型手柄和瞬时按键的统一边沿处理
control/trajectory.py 轨迹解析、有限值/维度/位置校验、循环播放和动态诊断
```

`bxi_example_test_wire.py` 和 `bxi_example_vibration.py` 都从这些模块读取关节定义，
避免以后修改关节顺序时两个测试节点不一致。

## 控制模式

```text
/motion_commands
        |
        +--> X / btn_9  --> 吊挂跑步：播放 / 暂停轨迹
        |
        +--> Y / btn_10 --> 振动测试：启动预检 -> 启动振动 -> 停止振动
        |
        v
bxi_example_suspended_tests（模式互斥、唯一 ActuatorCmds 发布者）
        |
        +--> simulation/actuators_cmds
        +--> hardware/actuators_cmds
```

推荐使用组合入口 `bxi_example_suspended_tests.py`。两种模式共用初始化、安全
状态机和执行器发布者。原来的独立跑步、独立振动入口仍保留用于单项调试。

振动模式仍可通过 ROS 服务控制；手柄只是同一个内部状态机的另一个入口，不会
绕过初始化、逐关节预检、反馈、位置边界或发布者冲突检查。

## 扩展新测试功能

新增控制模式时优先复用公共关节定义和 `RemoteButtonEdge`。新节点应至少满足：

1. 两阶段 reset 必须确认成功；
2. 发布前检查命令长度、有限值和关节位置限制；
3. 实机运动期间检查反馈和控制命令时效；
4. 检测同一 `actuators_cmds` 话题上的其他发布者；
5. 遥控、服务和自动流程最终进入同一个状态转换函数；
6. 安全故障锁存后不能通过再次按键直接恢复。
