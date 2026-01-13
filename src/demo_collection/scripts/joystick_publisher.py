#!/usr/bin/env python3
"""Joystick control for demo recording"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Joy


class JoystickPublisher(Node):
    def __init__(self):
        super().__init__('joystick_publisher')
        self.publisher = self.create_publisher(String, '/demo_commands', 10)
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        
        # Button state tracking (to detect button press, not hold)
        self.prev_buttons = []
        
        print("\n" + "="*50)
        print("JOYSTICK CONTROL")
        print("Button A (0): toggle gripper")
        print("Button X (2): start recording")
        print("Button Y (3): stop recording")
        print("="*50 + "\n")

    def joy_callback(self, msg):
        # Initialize previous button states if first message
        if not self.prev_buttons:
            self.prev_buttons = [0] * len(msg.buttons)
            return
        
        # Check for button presses (rising edge: 0 -> 1)
        for i, (curr, prev) in enumerate(zip(msg.buttons, self.prev_buttons)):
            if curr == 1 and prev == 0:  # Button just pressed
                self.handle_button_press(i)
        
        # Update previous state
        self.prev_buttons = list(msg.buttons)

    def handle_button_press(self, button_index):
        """Map button index to command"""
        command = None
        
        if button_index == 0:  # Button A
            command = 'g'
            print("[Joystick] Toggle gripper (A)")
        elif button_index == 2:  # Button X
            command = 'r'
            print("[Joystick] Start recording (X)")
        elif button_index == 3:  # Button Y
            command = 's'
            print("[Joystick] Stop recording (Y)")
        
        if command:
            msg = String()
            msg.data = command
            self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = JoystickPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
