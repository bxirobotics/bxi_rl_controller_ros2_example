from bxi_example_py_elf3.utils import sonic_connection


class _FakeRouteSocket:
    def __init__(self, address=None, connect_error=None):
        self.address = address
        self.connect_error = connect_error
        self.closed = False

    def connect(self, _target):
        if self.connect_error is not None:
            raise self.connect_error

    def getsockname(self):
        return (self.address, 12345)

    def close(self):
        self.closed = True


def test_detect_preferred_ipv4_uses_default_route(monkeypatch):
    route_socket = _FakeRouteSocket("192.168.88.164")
    monkeypatch.setattr(sonic_connection.socket, "socket", lambda *_: route_socket)

    assert sonic_connection.detect_preferred_ipv4() == "192.168.88.164"
    assert route_socket.closed


def test_detect_preferred_ipv4_falls_back_to_hostname(monkeypatch):
    route_socket = _FakeRouteSocket(connect_error=OSError("no route"))
    monkeypatch.setattr(sonic_connection.socket, "socket", lambda *_: route_socket)
    monkeypatch.setattr(sonic_connection.socket, "gethostname", lambda: "elf3-32")
    monkeypatch.setattr(
        sonic_connection.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (2, 2, 17, "", ("127.0.1.1", 0)),
            (2, 2, 17, "", ("192.168.88.164", 0)),
        ],
    )

    assert sonic_connection.detect_preferred_ipv4() == "192.168.88.164"
    assert route_socket.closed


def test_prepare_sonic_runtime_config_adds_ip_without_mutating_yaml_data():
    source = {
        "states": {
            "sonic_teleop": {
                "behavior": "SonicTeleopState",
                "manifest": {
                    "confirm_message": "进入后保持站立并完成PICO校准"
                }
            },
            "sonic_teleop_gripper": {
                "behavior": "SonicTeleopState",
                "manifest": {"confirm_message": "松开trigger后进入夹爪模式"},
            },
        }
    }

    runtime, address, message = sonic_connection.prepare_sonic_runtime_config(
        source, "192.168.88.104"
    )

    assert address == "192.168.88.104"
    assert message == "机器人IP：192.168.88.104（PICO连接此地址）"
    assert runtime["states"]["sonic_teleop"]["manifest"]["confirm_message"] == (
        "机器人IP：192.168.88.104（PICO连接此地址）；"
        "进入后保持站立并完成PICO校准"
    )
    assert runtime["states"]["sonic_teleop_gripper"]["manifest"][
        "confirm_message"
    ] == (
        "机器人IP：192.168.88.104（PICO连接此地址）；"
        "松开trigger后进入夹爪模式"
    )
    assert source["states"]["sonic_teleop"]["manifest"]["confirm_message"] == (
        "进入后保持站立并完成PICO校准"
    )
    assert source["states"]["sonic_teleop_gripper"]["manifest"][
        "confirm_message"
    ] == "松开trigger后进入夹爪模式"


def test_prepare_sonic_runtime_config_fails_open_without_network():
    source = {
        "states": {
            "sonic_teleop": {
                "behavior": "SonicTeleopState",
                "manifest": {"confirm_message": "完成PICO校准"},
            }
        }
    }

    runtime, address, message = sonic_connection.prepare_sonic_runtime_config(
        source, "127.0.0.1"
    )

    assert address == "127.0.0.1"
    assert message == "机器人IP：未检测到，请检查机器人网络"
    assert runtime["states"]["sonic_teleop"]["manifest"]["confirm_message"] == (
        "机器人IP：未检测到，请检查机器人网络；完成PICO校准"
    )
