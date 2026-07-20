## 按键说明

1. 按键映射对照图片：![手柄](./ps4_key_map.png "keymap") 
2. 其中AXIS为线性按键，箭头方向为正方向，值范围:-32767 - 32767，除扳机键L2/R2外初始值均为0  
3. 其中BT为常规按键，按下为1，松开为0
4. 方向键属于线性按键，但只有两个值，图示AXIS两端的键分别为最小和最大
5. L2/R2扳机键初始值为-32767，按下后线性增加到最大值，同时扳机键是一个复合按键，按下后同时触发一个常规按键，复合关系看图

## 多设备自动选择

`remote_controller` 将每个 `sources.<设备名>` 视为一个候选输入设备，并且严格只激活一个。候选设备按 `priority` 选择；高优先级设备连续可用 `promote_stable_ms` 后会自动抢占低优先级设备。

```yaml
inputs:
  selection:
    scan_interval_ms: 100       # 驱动可用性扫描周期
    promote_stable_ms: 500      # 高优先级设备稳定多久才允许抢占

sources:
  gamepad:
    type: joystick
    device: /dev/input/by-id/usb-...-joystick
    priority: 50
    ready_timeout_ms: 1000
    loss_timeout_ms: 300
    cooldown_ms: 1000
    signals: { ... }

  keyboard:
    type: keyboard
    priority: 0
    signals: { ... }
```

切换顺序固定为：停止旧驱动、清除旧设备所有信号、发布一次零运动命令、启动新驱动、等待新驱动声明就绪、再允许正常输入。切换期间会抑制所有 edge 输出；因此必须在切换后先松开、再按下 `start/stop` 等边沿按键，避免状态继承造成误触发。设备连续不可用超过 `loss_timeout_ms` 后会被断开，并在 `cooldown_ms` 内不参与再次选择。

`signals.*.timeout_ms` 是单个数据字段的过期保护，只适用于 UDP/TCP 等持续流式输入；它不用于手柄或 CRSF 的设备断连判断。默认手柄配置不设置它，断连统一由驱动的 `is_available()` 和 `loss_timeout_ms` 处理。

`--driver <type>` 和 `--keyboard` 保留为调试过滤器：它们只允许相应类型的候选设备参与选择。正常部署不要传该参数，让 YAML 优先级生效。

### 添加编译内置的新驱动

新驱动通常继承 `InputDriverBase`（或直接实现 `InputDriver`），并用 `register_input_driver_factory("类型名", factory)` 注册。`InputDriverBase::set_signal()` 已处理与 mapper 的同步。`is_available()` 必须非阻塞；驱动可按自身协议定义可用性。对 CRSF，建议仅在设备打开且近期收到校验通过的完整通道帧时返回可用。`is_ready()` 只有在已收到驱动认为足够安全的完整状态快照时才返回 `true`。

例如，编译进 `crsf` 驱动后可加入以下候选项。通道到 `crsf.vx` 等逻辑信号的复杂映射由 C++ 驱动实现，YAML 仅声明它们供 `controls` 使用：

```yaml
sources:
  crsf_remote:
    type: crsf
    device: /dev/ttyCRSF
    priority: 100
    ready_timeout_ms: 1000
    loss_timeout_ms: 300
    cooldown_ms: 1000
    baudrate: 420000             # 设备特定标量参数在 InputDeviceConfig.options 中
    signals:
      crsf.vx: {from: crsf.vx}
      crsf.vy: {from: crsf.vy}
      crsf.yaw: {from: crsf.yaw}
      crsf.start: {from: crsf.start}
      crsf.stop: {from: crsf.stop}
```

在 `controls` 中为这些 signal 添加源即可。尚未编译的 `type` 不会导致节点启动失败：节点会记录 warning、忽略该候选项，并自动使用下一个可用设备；没有任何可用设备时保持安全停止。
