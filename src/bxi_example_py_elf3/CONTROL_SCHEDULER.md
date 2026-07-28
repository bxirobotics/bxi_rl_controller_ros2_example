# 框架控制调度

`RobotControlRuntime` 使用独立的绝对时间线程，以固定 50 Hz 时间轴调用
`RobotControlFramework.update()`。它同时拥有 Framework 生命周期、低频 Mod
维护、线程安全、调度统计和逐条超时告警。ROS 平台适配器只负责输入快照、启动步骤和
电机帧输出，不再通过 ROS Timer 驱动动作或维护 Framework。

配置位于 `config/elf3_state_machine.yaml` 的 `control_runtime` 分段：

```yaml
control_runtime:
  period_sec: 0.02
  compute_budget_sec: 0.002
  deadline_tolerance_sec: 0.001
  maintenance_hz: 5.0
  statistics_interval_sec: 60.0
  deadline_warning_interval_sec: 1.0
  maintenance_guard_sec: 0.005
  python_switch_interval_sec: 0.001
  spin_wait_us: -1
  cpu_affinity: -1
  realtime_priority: 0
```

- `period_sec`：统一控制周期，所有状态和过渡共用。
- `compute_budget_sec`：单轮控制计算的目标耗时预算。它只用于统计预算超限，
  不会改变 20 ms 推理释放时间轴；硬截止时间始终是本轮释放后的下一个周期。
- `deadline_tolerance_sec`：释放唤醒或完成时间超过对应截止点后的额外容差。
- `cpu_affinity`：`-1` 表示不绑核，非负值表示控制线程绑定的 CPU。
- `realtime_priority`：`0` 表示普通调度，`1..99` 尝试启用
  `SCHED_FIFO`；权限不足时记录警告并继续使用普通调度。
- `maintenance_hz`：Runtime 自有低频线程的 Mod 进程监管频率，不进入 50 Hz
  控制路径。
- `statistics_interval_sec`：周期性 INFO 调度统计的输出间隔。
- `deadline_warning_interval_sec`：逐轮时间异常的合并告警间隔。默认 1 秒，
  无论这段时间内出现多少异常，最多输出一条汇总 WARNING。
- `maintenance_guard_sec`：距离下一次控制唤醒不足该时间时，跳过本轮
  状态快照或 Mod 维护，避免反向阻塞控制线程。
- `python_switch_interval_sec`：Python GIL 切换间隔。默认 1 ms，降低
  ROS Python 回调造成的控制线程唤醒长尾。
- `spin_wait_us`：末段忙等时长，单位微秒。`-1` 表示关闭；非负值表示先正常
  睡眠，在距离绝对释放点剩余该时长时切换为忙等。例如 `200` 表示最后
  200 微秒忙等。该值必须小于一个控制周期；忙等会短暂占用一个 CPU 核心并
  持有 Python GIL，建议只配置数百微秒并配合控制线程绑核使用。

控制线程以每 20 ms 一个绝对释放点调用一次 Framework；完成时间不会被用来推导下一次
释放点，因此推理耗时的短期波动不会形成累计漂移。启动阶段也沿用同一时间轴。
单轮轻微越过下一释放点时，下一轮会立即补上并在后续短周期恢复原时间轴；只有已经
跨过完整的额外释放点时才计为跳过，避免无谓地少执行一次推理。
带历史观测的策略在状态准备阶段直接用当前观测填充历史缓冲，仅执行一次预热推理；
不会再在一个控制周期内集中执行 `history_len * 2` 次推理。

`statistics_interval_sec` 只控制 INFO 统计摘要的输出周期。摘要包含唤醒延迟、
完整控制周期耗时，以及自上次成功输出以来的控制周期、预算超限、deadline miss 和
跳过周期增量。

每次 deadline miss 都由控制线程轻量放入队列，再由现有 maintenance 线程合并统计，
不会在 50 Hz 控制线程内执行日志 I/O。默认每秒最多打印一条汇总 WARNING，分别给出
调度唤醒延迟超限次数、周期完成截止超限次数和最大值，并根据异常类型给出初步判定。
一次输出失败会保留汇总，并在后续维护周期重试。deadline miss 不自动切换状态。
状态信息话题的 `control_timing` 字段提供累计计数和最后一次 miss。

时间统计日志使用正式、统一的中文技术术语。时序异常汇总说明调度唤醒延迟、周期执行
耗时、周期完成截止超限及初步判定；周期性能统计给出异常比例、预算超限次数、跳过周期
数、P99 指标和最大值。

Runtime 和 Scheduler 的 INFO、WARNING、ERROR 使用彼此独立的 Python 调用位置，
兼容 `rclpy` 对同一日志调用位置固定 severity 的要求；进入高负载状态后从 INFO
切换为 WARNING 不会再被 ROS 日志系统拒绝。

`control_runtime` 在构造 `RobotControlFramework` 前从系统配置中移除，因此不会
参与状态图或 Mod 配置合并。仿真、真机以及未来的非 ROS 平台适配器共用同一运行时。

`bxi_example_py_elf3_mjlab` 保持原有调度，不受这些参数影响。
