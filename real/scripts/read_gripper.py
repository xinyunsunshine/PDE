"""Print gripper position in a loop."""
import time
import sys
from piper_sdk import C_PiperInterface_V2

g = C_PiperInterface_V2("can_gripper")
g.ConnectPort()
time.sleep(0.5)

print("Reading gripper state (Ctrl+C to stop)...")
try:
    while True:
        state = g.GetArmGripperMsgs()
        angle = state.gripper_state.grippers_angle
        effort = state.gripper_state.grippers_effort
        width_m = angle / 1_000_000.0
        sys.stdout.write(f"\rangle={angle}  width={width_m:.4f}m  effort={effort}   ")
        sys.stdout.flush()
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nDone.")
