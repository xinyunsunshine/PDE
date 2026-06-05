from piper_sdk import *
import time
import numpy as np

_AGILEX_FACTOR = 57324.840764 # Multiply by this to convert rad to units of 0.001deg (1000 * 180 / 3.14) for piper_sdk

if __name__ == "__main__":
    piper = C_PiperInterface_V2("can_piper")
    piper.ConnectPort()


    while True: 
        struct = piper.GetArmJointMsgs().joint_state
        qpos = np.array([getattr(struct, f"joint_{i}") for i in range(1, 7)])
        qpos = qpos / _AGILEX_FACTOR
        print(f"qpos: {qpos}")
        time.sleep(1 / 50)