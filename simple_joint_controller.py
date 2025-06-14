#!/usr/bin/env python

"""
Simple Joint Controller for SO101 Robot

A user-friendly script for controlling SO101 robot joints with keyboard input.
Provides real-time feedback, error handling, and easy joint switching.

Features:
- Select any joint to control
- Use simple keyboard commands to move joints
- Real-time position feedback
- Adjustable movement increments
- Detailed error reporting with suggestions
- Clean disconnect handling

Usage:
    python simple_joint_controller.py --port /dev/ttyUSB0

Controls:
    up/w/+     : Increase joint position
    down/s/-   : Decrease joint position
    </,        : Decrease movement increment
    >/,        : Increase movement increment
    j          : Switch to different joint
    r          : Read all current positions
    h          : Show help
    q          : Quit
"""

import argparse
import logging
import time
import sys
import os
from typing import Dict, Any, Optional

# Add the lerobot package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'lerobot'))

from lerobot.common.robots.so101_follower import SO101FollowerConfig, SO101Follower

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Only show warnings/errors
logger = logging.getLogger(__name__)

class SimpleJointController:
    """
    Simple and user-friendly controller for SO101 robot joints.
    """
    
    def __init__(self, port: str = "COM7"):
        """Initialize the controller."""
        self.robot_config = SO101FollowerConfig(
            port=port,
            id="simple_controller",
            max_relative_target=30.0  # Allow reasonable movements
        )
        self.robot = SO101Follower(self.robot_config)
        
        # Available joints and their descriptions
        self.joints = {
            "shoulder_pan": "Base rotation (left/right)",
            "shoulder_lift": "Shoulder up/down",
            "elbow_flex": "Elbow bend",
            "wrist_flex": "Wrist up/down",
            "wrist_roll": "Wrist rotation",
            "gripper": "Gripper open/close"
        }
        
        # Control state
        self.current_joint: Optional[str] = None
        self.increment = 2.0
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to the robot."""
        try:
            print("🔌 Connecting to robot...")
            self.robot.connect()
            
            # Test connection by reading positions
            obs = self.robot.get_observation()
            print(f"✅ Connected! Found {len(obs)} joint sensors")
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Safely disconnect from the robot."""
        if self.connected:
            try:
                self.robot.disconnect()
                print("✅ Disconnected safely")
            except Exception as e:
                print(f"⚠️  Warning during disconnect: {e}")
            finally:
                self.connected = False
    
    def get_positions(self) -> Dict[str, float]:
        """Get current joint positions."""
        try:
            obs = self.robot.get_observation()
            return {k.replace('.pos', ''): v for k, v in obs.items() if k.endswith('.pos')}
        except Exception as e:
            print(f"❌ Error reading positions: {e}")
            return {}
    
    def move_joint(self, joint: str, position: float) -> bool:
        """
        Move a specific joint to a target position.
        
        Args:
            joint: Joint name (e.g., 'gripper')
            position: Target position
            
        Returns:
            True if successful, False otherwise
        """
        action = {f"{joint}.pos": position}
        
        print(f"📤 Moving {joint} to {position:.1f}...")
        
        try:
            # Send the command
            self.robot.send_action(action)
            
            # Wait for movement
            time.sleep(0.3)
            
            # Confirm new position
            new_positions = self.get_positions()
            if joint in new_positions:
                actual_pos = new_positions[joint]
                print(f"✅ {joint} now at {actual_pos:.1f}")
                return True
            else:
                print(f"⚠️  Could not confirm {joint} position")
                return False
                
        except Exception as e:
            print(f"❌ Movement failed: {e}")
            
            # Provide helpful error suggestions
            error_str = str(e).lower()
            if "10" in str(e):
                print("💡 Tip: Movement too large - try smaller increment")
            elif "overload" in error_str:
                print("💡 Tip: Motor overload - joint may be stuck")
            elif "packet" in error_str:
                print("💡 Tip: Communication error - check connection")
            elif "timeout" in error_str:
                print("💡 Tip: Robot not responding - may need restart")
            
            return False
    
    def select_joint(self) -> Optional[str]:
        """Let user select which joint to control."""
        print("\n" + "="*50)
        print("🎯 SELECT JOINT TO CONTROL")
        print("="*50)
        
        joint_list = list(self.joints.keys())
        for i, joint in enumerate(joint_list, 1):
            description = self.joints[joint]
            print(f"{i}. {joint:<15} - {description}")
        
        while True:
            try:
                choice = input(f"\nSelect joint (1-{len(joint_list)}): ").strip()
                
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(joint_list):
                        selected = joint_list[idx]
                        print(f"✅ Selected: {selected}")
                        return selected
                
                print(f"Please enter a number between 1 and {len(joint_list)}")
                
            except (ValueError, KeyboardInterrupt):
                print("\nCancelled")
                return None
    
    def show_controls(self):
        """Display control instructions."""
        if not self.current_joint:
            return
            
        joint_desc = self.joints.get(self.current_joint, "Unknown")
        
        print(f"\n" + "="*50)
        print(f"🎮 CONTROLLING: {self.current_joint.upper()}")
        print(f"📝 {joint_desc}")
        print(f"📏 Movement increment: ±{self.increment:.1f}")
        print("="*50)
        print("CONTROLS:")
        print("  up/w/+    Move joint up")
        print("  down/s/-  Move joint down")
        print("  </,       Smaller movements")
        print("  >/,       Larger movements")
        print("  j         Switch joint")
        print("  r         Read positions")
        print("  h         Show help")
        print("  q         Quit")
        print("="*50)
    
    def run(self):
        """Run the interactive control session."""
        print("🤖 SO101 Simple Joint Controller")
        print("="*50)
        
        # Connect to robot
        if not self.connect():
            return
        
        try:
            # Select initial joint
            self.current_joint = self.select_joint()
            if not self.current_joint:
                return
            
            # Get initial position
            positions = self.get_positions()
            if not positions:
                print("❌ Could not read robot state")
                return
            
            current_pos = positions.get(self.current_joint, 0.0)
            print(f"📍 Current {self.current_joint} position: {current_pos:.1f}")
            
            self.show_controls()
            
            # Main control loop
            print(f"\n🎮 Ready! Press keys to control {self.current_joint}")
            
            while True:
                # Show current status
                positions = self.get_positions()
                current_pos = positions.get(self.current_joint, 0.0)
                
                print(f"\n[{self.current_joint}] {current_pos:.1f} | ±{self.increment:.1f}")
                command = input("Command: ").strip().lower()
                
                if command in ['q', 'quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                elif command in ['up', 'w', '+']:
                    new_pos = current_pos + self.increment
                    print(f"⬆️  {current_pos:.1f} → {new_pos:.1f}")
                    self.move_joint(self.current_joint, new_pos)
                
                elif command in ['down', 's', '-']:
                    new_pos = current_pos - self.increment
                    print(f"⬇️  {current_pos:.1f} → {new_pos:.1f}")
                    self.move_joint(self.current_joint, new_pos)
                
                elif command in ['<', ',']:
                    self.increment = max(0.5, self.increment - 0.5)
                    print(f"📉 Increment: {self.increment:.1f}")
                
                elif command in ['>', '.']:
                    self.increment = min(20.0, self.increment + 0.5)
                    print(f"📈 Increment: {self.increment:.1f}")
                
                elif command in ['j', 'joint']:
                    new_joint = self.select_joint()
                    if new_joint:
                        self.current_joint = new_joint
                        positions = self.get_positions()
                        current_pos = positions.get(self.current_joint, 0.0)
                        print(f"🔄 Switched to {self.current_joint} (at {current_pos:.1f})")
                        self.show_controls()
                
                elif command in ['r', 'read']:
                    positions = self.get_positions()
                    print("📊 Current positions:")
                    for joint, pos in positions.items():
                        marker = " 👈" if joint == self.current_joint else ""
                        print(f"  {joint:<15}: {pos:6.1f}{marker}")
                
                elif command in ['h', 'help']:
                    self.show_controls()
                
                elif command == '':
                    # Empty input, just continue
                    continue
                
                else:
                    print(f"❓ Unknown command '{command}'. Type 'h' for help")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
        finally:
            self.disconnect()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple SO101 Joint Controller")
    parser.add_argument("--port", default="COM7",
                       help="Robot serial port (default: /dev/ttyUSB0)")
    
    args = parser.parse_args()
    
    controller = SimpleJointController(port=args.port)
    controller.run()

if __name__ == "__main__":
    main()
