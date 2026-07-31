"""Non-blocking ZMQ subscriber that keeps only the latest message (CONFLATE mode)."""

# Modified by BXI Robotics in 2026 to make ZMQ shutdown deterministic.

import zmq


class ZMQPoller:
    """Simple ZMQ subscriber for sporadic non-blocking reads."""

    def __init__(self, host: str = "localhost", port: int = 5555, topic: str = ""):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.connect(f"tcp://{host}:{port}")
        self._topic = topic

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_data(self):
        """Get latest data or None if no data available."""
        if self._socket.poll(timeout=0):
            data = self._socket.recv(zmq.NOBLOCK)
            if data is None:
                print("ZMQPoller: no data received")
                return None

            # Strip topic prefix
            return data[len(self._topic) :]

        print("ZMQPoller: no data available")
        return None

    def close(self):
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

