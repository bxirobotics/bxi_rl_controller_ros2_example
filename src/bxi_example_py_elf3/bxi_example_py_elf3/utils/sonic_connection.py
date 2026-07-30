import copy
import ipaddress
import socket
from typing import Any, Dict, Optional, Tuple


_ROUTE_PROBE = ("192.0.2.1", 9)
_UNAVAILABLE_TEXT = "机器人IP：未检测到，请检查机器人网络"


def _usable_ipv4(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return bool(
        address.version == 4
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_unspecified
    )


def detect_preferred_ipv4() -> Optional[str]:
    """Return the IPv4 selected by the default route, with a hostname fallback.

    Connecting a UDP socket does not send traffic; it only asks the kernel which
    local address it would use for that route.  The TEST-NET address avoids a
    dependency on a public DNS or Internet service.
    """

    route_socket = None
    try:
        route_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        route_socket.connect(_ROUTE_PROBE)
        address = str(route_socket.getsockname()[0])
        if _usable_ipv4(address):
            return address
    except OSError:
        pass
    finally:
        if route_socket is not None:
            route_socket.close()

    try:
        addresses = socket.getaddrinfo(
            socket.gethostname(),
            None,
            family=socket.AF_INET,
            type=socket.SOCK_DGRAM,
        )
    except OSError:
        return None

    for address_info in addresses:
        address = str(address_info[4][0])
        if _usable_ipv4(address):
            return address
    return None


def sonic_connection_message(robot_ipv4: Optional[str]) -> str:
    if robot_ipv4 and _usable_ipv4(robot_ipv4):
        return f"机器人IP：{robot_ipv4}（PICO连接此地址）"
    return _UNAVAILABLE_TEXT


def prepare_sonic_runtime_config(
    config: Dict[str, Any], robot_ipv4: Optional[str] = None
) -> Tuple[Dict[str, Any], Optional[str], str]:
    """Copy a state-machine config and add the current robot IP to SONIC UI text."""

    runtime_config = copy.deepcopy(config)
    if robot_ipv4 is None:
        robot_ipv4 = detect_preferred_ipv4()
    message = sonic_connection_message(robot_ipv4)

    states = runtime_config.get("states") or {}
    for sonic_state in states.values():
        if not isinstance(sonic_state, dict):
            continue
        if sonic_state.get("behavior") != "SonicTeleopState":
            continue
        manifest = sonic_state.get("manifest")
        if isinstance(manifest, dict):
            instructions = str(manifest.get("confirm_message") or "").strip()
            manifest["confirm_message"] = (
                f"{message}；{instructions}" if instructions else message
            )

    return runtime_config, robot_ipv4, message
