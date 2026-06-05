import time 
import numpy as np
import json
import subprocess
import threading
import cv2
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
from piper_sdk import C_PiperInterface_V2
from real.scripts.agilex_read import _AGILEX_FACTOR

class AgileXPositionController: 
    def __init__(self): 
        self.piper = C_PiperInterface_V2("can_piper")
        self.piper.ConnectPort()
        while not self.piper.EnablePiper():
            time.sleep(0.01)

    def move(self, q: np.ndarray): 
        q = np.rint(q * _AGILEX_FACTOR) # round and convert to int
        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.piper.JointCtrl(*[int(qi) for qi in q.tolist()])
        time.sleep(1)


class UncutVideoRecorder:
    """
    Continuous video recorder that streams frames directly to ffmpeg.
    Records uncut video across all episodes with text overlays indicating state.
    Saves a JSON sidecar file with full metadata.
    """

    def __init__(
        self,
        output_dir: str,
        frame_source: Callable[[], Optional[np.ndarray]],
        resolution: tuple[int, int] = (640, 480),
        fps: int = 30,
        checkpoint_path: str = None,
        agent_config: dict = None,
        inference_config: dict = None,
        task_id: Optional[int] = None,
        task_name: Optional[str] = None,
        round_num: Optional[int] = None,
        prompt_idx: Optional[int] = None,
        prompt_text: Optional[str] = None,
        expected_trials: Optional[int] = None,
    ):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._frame_source = frame_source
        self._resolution = resolution  # (width, height)
        self._fps = fps
        self._frame_interval = 1.0 / fps

        self._timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._video_path = self._output_dir / f"{self._timestamp}_uncut.mp4"
        self._json_path = self._output_dir / f"{self._timestamp}_uncut.json"

        self._state = "idle"
        self._episode = 0
        self._step = 0
        self._label = ""
        self._frame_count = 0

        # Episode metadata
        self._start_time = datetime.now()
        self._episodes = []
        self._current_episode_start_frame = 0
        self._current_episode_start_time = None

        self._checkpoint_path = checkpoint_path
        self._agent_config = agent_config
        self._inference_config = inference_config

        # Prompt-opt pipeline coordinates (optional; included in metadata when set).
        self._task_id = task_id
        self._task_name = task_name
        self._round_num = round_num
        self._prompt_idx = prompt_idx
        self._prompt_text = prompt_text
        self._expected_trials = expected_trials
        
        # Threading for background capture
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        self._proc = subprocess.Popen(
            [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-s', f'{resolution[0]}x{resolution[1]}',
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                str(self._video_path)
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        
        print(f"[Recorder] Started recording to {self._video_path}")

    def start(self):
        """Start the background capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("[Recorder] Background capture started")

    def _capture_loop(self):
        """Background thread that captures frames at fixed rate."""
        next_capture_time = time.perf_counter()
        while self._running:
            now = time.perf_counter()
            if now >= next_capture_time:
                frame = self._frame_source()
                if frame is not None:
                    self._write_frame(frame)
                next_capture_time += self._frame_interval
                # Prevent drift accumulation
                if next_capture_time < now:
                    next_capture_time = now + self._frame_interval
            else:
                sleep_time = next_capture_time - now - 0.001
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def _write_frame(self, frame: np.ndarray):
        """Add overlay and write frame to ffmpeg."""
        if self._proc.stdin.closed:
            return

        h, w = frame.shape[:2]
        exp_w, exp_h = self._resolution
        if w != exp_w or h != exp_h:
            frame = cv2.resize(frame, (exp_w, exp_h))

        frame_with_overlay = frame.copy()
        
        # Add text overlay based on current state
        self._add_overlay(frame_with_overlay)
        
        try:
            self._proc.stdin.write(frame_with_overlay.tobytes())
            with self._lock:
                self._frame_count += 1
        except BrokenPipeError:
            pass

    def _add_overlay(self, frame: np.ndarray):
        """Add text overlay to frame based on current state."""
        with self._lock:
            state = self._state
            episode = self._episode
            step = self._step
            label = self._label
        
        # Overlay settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        padding = 10
        
        if state == "idle":
            text = "IDLE - Waiting to start"
            color = (128, 128, 128)  
        elif state == "reset":
            text = f"[RESET] Uncut Rollouts: Episode {episode} - Resetting Env"
            color = (0, 165, 255)  
        elif state == "episode":
            text = f"Episode {episode} - Step {step}"
            color = (0, 255, 0)  
        elif state == "pause":
            if label:
                text = f"[WAIT] Uncut Rollouts: Episode {episode} - {label.upper()}"
            else:
                text = f"[WAIT] Uncut Rollouts: Episode {episode} - Labeling"
            color = (0, 255, 255)  
        else:
            text = state
            color = (255, 255, 255)
        
        # Draw background rectangle for text visibility
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(
            frame,
            (padding, padding),
            (padding + text_w + 10, padding + text_h + baseline + 10),
            (0, 0, 0),
            -1
        )
        
        cv2.putText(
            frame,
            text,
            (padding + 5, padding + text_h + 5),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
        
        # Draw frame counter
        frame_text = f"Frame: {self._frame_count}"
        (fw, fh), _ = cv2.getTextSize(frame_text, font, 0.5, 1)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (w - fw - 20, h - fh - 20), (w - 5, h - 5), (0, 0, 0), -1)
        cv2.putText(frame, frame_text, (w - fw - 15, h - 10), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    def set_state(self, state: str, episode: int = None, step: int = None, label: str = None):
        """Update the overlay state (thread-safe)."""
        with self._lock:
            self._state = state
            if episode is not None:
                self._episode = episode
            if step is not None:
                self._step = step
            if label is not None:
                self._label = label

    def start_episode(self, episode_id: int):
        """Mark the start of a new episode."""
        with self._lock:
            self._current_episode_start_frame = self._frame_count
            self._current_episode_start_time = datetime.now()
        self.set_state("episode", episode=episode_id, step=0, label="")

    def end_episode(self, episode_id: int, label: str, steps: int):
        """Mark the end of an episode and record metadata."""
        with self._lock:
            episode_data = {
                "episode_id": episode_id,
                "start_frame": self._current_episode_start_frame,
                "end_frame": self._frame_count,
                "start_time": self._current_episode_start_time.isoformat() if self._current_episode_start_time else None,
                "end_time": datetime.now().isoformat(),
                "steps": steps,
                "label": label,
            }
            self._episodes.append(episode_data)
        self.set_state("pause", episode=episode_id, label=label)

    def close(self):
        """Stop recording and save metadata."""
        print("[Recorder] Stopping...")
        
        # Stop the background capture thread
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        
        # Close ffmpeg
        if self._proc.stdin and not self._proc.stdin.closed:
            self._proc.stdin.close()
        self._proc.wait(timeout=10.0)
        
        self._save_metadata()
        
        print(f"[Recorder] Saved video to {self._video_path}")
        print(f"[Recorder] Saved metadata to {self._json_path}")

    def _save_metadata(self):
        """Save the metadata JSON sidecar file."""
        end_time = datetime.now()
        
        # Stats
        successes = sum(1 for ep in self._episodes if ep["label"] == "success")
        failures = sum(1 for ep in self._episodes if ep["label"] == "failure")
        resets = sum(1 for ep in self._episodes if ep["label"] == "reset")
        total = len(self._episodes)

        valid = successes + failures
        metadata = {
            "task_id": self._task_id,
            "task_name": self._task_name,
            "round_num": self._round_num,
            "prompt_idx": self._prompt_idx,
            "prompt_text": self._prompt_text,
            "recording_info": {
                "video_file": self._video_path.name,
                "start_time": self._start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": (end_time - self._start_time).total_seconds(),
                "total_frames": self._frame_count,
                "fps": self._fps,
                "resolution": list(self._resolution),
                "expected_trials": self._expected_trials,
            },
            "checkpoint": {
                "path": self._checkpoint_path,
            },
            "agent_config": self._agent_config,
            "inference_config": self._inference_config,
            "summary": {
                "total_episodes": total,
                "successes": successes,
                "failures": failures,
                "resets": resets,
                "success_rate": successes / valid if valid > 0 else 0.0,
            },
            "episodes": self._episodes,
        }
        
        with open(self._json_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
