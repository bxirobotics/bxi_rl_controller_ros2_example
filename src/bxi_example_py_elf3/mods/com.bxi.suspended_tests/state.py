"""Button states used by the combined suspended test node."""

from bxi_example_py_elf3.framework.control_state import ButtonControlState


class SuspendedRunningState(ButtonControlState):
    name = "running"
    button = "X"
    message_field = "btn_9"
    source = "remote X button"

    def is_active(self, node):
        return node.run_enabled

    def is_pending(self, node):
        return node.pending_mode == "running"

    def handle_button(self, node, request_received_at):
        with node.state_lock:
            if node.safety_fault:
                node.get_logger().error(
                    "X rejected after a latched safety fault; restart required"
                )
                return
            if node.reset_stage != 2:
                node.get_logger().warning(
                    "X ignored because robot initialization is incomplete"
                )
                return
            if self.is_active(node):
                node._stop_running_locked(self.source)
                return
            if (
                node.test_enabled
                or node.joint_test_running
                or node.limb_test_running
            ):
                node.get_logger().warning(
                    "X rejected while vibration or joint test is active; "
                    "stop the active mode first"
                )
                return
            if node.returning_to_center or node.pending_mode:
                node.get_logger().warning(
                    "X ignored while a mode transition is in progress"
                )
                return
            node._prepare_running_locked(self.source)


class SuspendedVibrationState(ButtonControlState):
    name = "vibration"
    button = "Y"
    message_field = "btn_10"
    source = "remote Y button"

    def is_active(self, node):
        return node.test_enabled

    def is_pending(self, node):
        return node.pending_mode in ("vibration", "vibration_start")

    def handle_button(self, node, request_received_at):
        with node.state_lock:
            if node.safety_fault:
                node.get_logger().error(
                    "Y rejected after a latched safety fault; restart required"
                )
                return
            if node.reset_stage != 2:
                node.get_logger().warning(
                    "Y ignored because robot initialization is incomplete"
                )
                return
            if self.is_active(node):
                node._stop_test(self.source)
                return
            if node.limb_test_running:
                node.get_logger().warning(
                    "Y rejected while the A-key joint test is active; press A "
                    "to stop it first"
                )
                return
            if node.joint_test_running:
                node._cancel_joint_rotation_test(self.source)
                return
            if self.is_pending(node):
                node.pending_mode = ""
                node._queue_diagnostic_log(
                    "info",
                    "pending vibration start cancelled by remote Y button",
                )
                return
            if (
                node.returning_to_center
                and node.return_owner == "limb_test_stop"
            ):
                node.pending_mode = "vibration"
                node._queue_diagnostic_log(
                    "info",
                    "vibration requested during A-key safe return; vibration "
                    "will start automatically after the zero pose is reached",
                )
                return
            if node.returning_to_center or node.pending_mode:
                node.get_logger().warning(
                    "Y ignored while a mode transition is in progress"
                )
                return
        node._request_vibration_start(self.source, request_received_at)


class SuspendedLimbTestState(ButtonControlState):
    name = "whole_body_joint_test"
    button = "A"
    message_field = "btn_7"
    source = "remote A button"

    def is_active(self, node):
        return node.limb_test_running

    def is_pending(self, node):
        return node.pending_mode == "limb_test"

    def handle_button(self, node, request_received_at):
        with node.state_lock:
            if node.safety_fault:
                node.get_logger().error(
                    "A rejected after a latched safety fault; restart required"
                )
                return
            if node.reset_stage != 2:
                node.get_logger().warning(
                    "A ignored because robot initialization is incomplete"
                )
                return
            if self.is_active(node):
                node._stop_limb_test_locked(self.source)
                return
            if node.test_enabled or node.joint_test_running:
                node.get_logger().warning(
                    "A rejected while vibration is active; stop it with Y "
                    "first"
                )
                return
            if node.returning_to_center or node.pending_mode:
                node.get_logger().warning(
                    "A ignored while a mode transition is in progress"
                )
                return
            node._prepare_limb_test_locked(self.source)
