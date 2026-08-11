"""Small state primitives for button-driven motor control."""


class ButtonControlState:
    """Base class for a control mode triggered by one remote button."""

    name = ""
    button = ""
    message_field = ""
    source = ""

    def is_active(self, node):
        return False

    def is_pending(self, node):
        return False

    def handle_button(self, node, request_received_at):
        raise NotImplementedError
