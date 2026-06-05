import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
from omegaconf import DictConfig


@dataclass
class Frame:
    color: np.ndarray
    depth: Optional[np.ndarray]
    timestamp: float


class RSDevice:
    @staticmethod
    def get_device_serial_numbers() -> list[str]:
        devices = rs.context().devices
        return [d.get_info(rs.camera_info.serial_number) for d in devices]

    def __init__(self, name: str, config: DictConfig):
        self.name = name
        self.serial_number = str(config.serial)
        assert self.serial_number in self.get_device_serial_numbers(), \
            f"Serial number {self.serial_number} not found."

        self.use_depth = config.depth
        target = config.get("target_size", None)
        self.target_size = tuple(target) if target else None  # (width, height) for resize

        # Configure pipeline
        self.pipe = rs.pipeline()
        self.rs_cfg = rs.config()
        self.rs_cfg.enable_device(self.serial_number)
        width, height = config.resolution
        self.rs_cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, config.frame_rate)
        if self.use_depth: 
            self.rs_cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, config.frame_rate)

        # Start pipeline
        profile = self.pipe.start(self.rs_cfg)
        sensor = profile.get_device().query_sensors()[0]
        # Get intrinsics
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._intrinsics_K = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        self._distortion_coefficients = np.array(intrinsics.coeffs)
        
        # Set exposure settings
        # sensor.set_option(rs.option.exposure, config.get("exposure", 40000))
        # TODO: Add whitebalance and other useful settings here
        self.align = rs.align(rs.stream.color)

        # Threading state
        self._lock = threading.Lock()
        self._latest: Optional[Frame] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def K(self) -> np.ndarray: 
        """Returns the factory calibrated intrinsics matrix for the camera, K."""
        return self._intrinsics_K

    @property 
    def distortion_coefficients(self) -> np.ndarray: 
        """Returns the factory calibrated distortion coefficients for the camera, D."""
        return self._distortion_coefficients

    def start(self):
        if self._running:
            # No-op if running already 
            return
        self.calibrate_camera_to_system_offset()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def calibrate_camera_to_system_offset(self, n_samples=50):
        """Calibrate hw camera hw clock to system clock offset with n_samples samples."""
        offsets = []
        for _ in range(n_samples):
            frames = self.pipe.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            
            camera_ts = color_frame.get_timestamp() / 1000.0
            system_ts = time.time()
            
            offsets.append(system_ts - camera_ts)
        
        self.camera_to_system_offset = np.median(offsets)
        print(f"Camera to system offset: {self.camera_to_system_offset:.3f} seconds")

    def _capture_loop(self):
        while self._running:
            frames = self.pipe.wait_for_frames()
            aligned = self.align.process(frames)

            color_frame = aligned.get_color_frame()
            timestamp = color_frame.get_timestamp() / 1000.0 + self.camera_to_system_offset 

            color = np.asarray(color_frame.get_data())
            if self.target_size:
                color = cv2.resize(color, self.target_size, interpolation=cv2.INTER_LINEAR)
            else:
                color = color.copy() 

            depth = None
            if self.use_depth:
                depth = np.asarray(aligned.get_depth_frame().get_data()).copy()

            with self._lock:
                self._latest = Frame(color, depth, timestamp)

    def get_frame(self) -> Optional[Frame]:
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.pipe.stop()


class MultiRSDevice:
    def __init__(self, perception_cfg: DictConfig):
        self.devices = {
            name: RSDevice(name, cfg)
            for name, cfg in perception_cfg.items()
        }

    def start(self):
        for dev in self.devices.values():
            dev.start()

    def get_frames(self) -> dict[str, Optional[Frame]]:
        return {name: dev.get_frame() for name, dev in self.devices.items()}

    def stop(self):
        for dev in self.devices.values():
            dev.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def _rget_device(self, key: str) -> RSDevice: 
        if key not in self.devices: 
            raise ValueError(f"Device {key} not found in MultiRSDevice")
        return self.devices[key]


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("/home/user/mdpo/real/scripts/config/base.yaml")

    with MultiRSDevice(cfg.perception) as cams:
        time.sleep(0.5)  # Let cameras warm up
        frames = cams.get_frames()
        for name, frame in frames.items():
            if frame:
                print(f"{name}: {frame.color.shape}, t={frame.timestamp:.3f}")

        start_time = time.time()
