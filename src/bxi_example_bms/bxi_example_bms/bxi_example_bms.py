import rclpy
from rclpy.node import Node
import communication.msg as bxiMsg

class BmsSubscriber(Node):
    def __init__(self):
        super().__init__('bxi_example_bms')
        self.subscription = self.create_subscription(bxiMsg.BatteryStates,'battery_states',self.listener_callback,10)

    def listener_callback(self, msg):
        print("voltage: ", msg.voltage)
        print("current: ", msg.current)
        print("soc: ", msg.soc) # %
        print("battery_temperature: ", msg.battery_temperature)

def main(args=None):
    rclpy.init(args=args)
    bms_subscriber = BmsSubscriber()
    rclpy.spin(bms_subscriber)
    bms_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()