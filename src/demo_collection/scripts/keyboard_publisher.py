#!/usr/bin/env python3
"""Keyboard control for demo recording"""

import sys
import termios
import tty
import select
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class KeyboardPublisher(Node):
    def __init__(self):
        super().__init__('keyboard_publisher')
        self.publisher = self.create_publisher(String, '/demo_commands', 10)
        print("\n" + "="*50)
        print("KEYBOARD CONTROL")
        print("g: toggle gripper | r: record | s: stop | q: quit")
        print("="*50 + "\n")

    def publish_command(self, cmd):
        msg = String()
        msg.data = cmd
        self.publisher.publish(msg)
        print(f"[{cmd}] sent")


def get_key():
    """Get single keypress from stdin without buffering"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)  # Use cbreak mode instead of raw
        ch = sys.stdin.read(1)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardPublisher()
    
    try:
        while rclpy.ok():
            # Check for input with short timeout
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                key = get_key()
                k = key.lower()
                
                if k in ['g', 'r', 's']:
                    node.publish_command(k)
                elif k == 'q':
                    node.publish_command('q')
                    print("Quitting...")
                    break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()