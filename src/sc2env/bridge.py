"""Communication bridge between the SAGE agent and the optional Web API.

event_queue carries episode events; control_queue carries pause/resume/stop;
status_queue carries periodic state updates; history_queue carries episode
records; param_update_queue and param_confirm_queue carry parameter changes
and acknowledgements. Standalone evaluations disable outbound publication."""

import time
from multiprocessing import Queue, Event
from typing import Any, Dict, List, Optional


class GameBridge:
    def __init__(self, publish=True):
        # Standalone evaluations have no API consumer. Publishing large episode
        # payloads would block multiprocessing queue feeders during shutdown.
        # The agent writes its authoritative episode logs directly to disk.
        self.publish = publish
        self.event_queue: Queue = Queue()
        self.control_queue: Queue = Queue(maxsize=1)
        self.status_queue: Queue = Queue(maxsize=100)
        self._stop_event = Event()
        self._run_episode_event = Event()
        self.history_queue: Queue = Queue(maxsize=50000)
        self.param_update_queue: Queue = Queue(maxsize=1)
        self.param_confirm_queue: Queue = Queue(maxsize=1)

    # ---- Agent → API ----

    def put_event(self, event_dict: Dict[str, Any]) -> None:
        if not self.publish:
            return
        try:
            self.event_queue.put_nowait(event_dict)
        except Exception:
            pass

    def get_events(self) -> List[Dict[str, Any]]:
        events = []
        try:
            while True:
                events.append(self.event_queue.get_nowait())
        except Exception:
            pass
        return events

    # Agent -> API: incremental status updates.

    def update_status(self, **kwargs) -> None:
        if not self.publish:
            return
        try:
            self.status_queue.put_nowait(kwargs)
        except Exception:
            pass

    def drain_status_updates(self) -> List[Dict[str, Any]]:
        updates = []
        try:
            while True:
                updates.append(self.status_queue.get_nowait())
        except Exception:
            pass
        return updates

    # ---- API → Agent (control only) ----

    def send_control(self, cmd: str) -> None:
        try:
            if not self.control_queue.empty():
                self.control_queue.get_nowait()
            self.control_queue.put_nowait(cmd)
        except Exception:
            pass

    def check_control(self) -> Optional[str]:
        try:
            return self.control_queue.get_nowait()
        except Exception:
            return None

    def wait_until_resumed(self, poll_interval: float = 0.1) -> Optional[str]:
        self.update_status(paused=True)
        resume_cmd = None
        while not self._stop_event.is_set():
            cmd = self.check_control()
            if cmd in ("resume", "stop", "step"):
                resume_cmd = cmd
                break
            time.sleep(poll_interval)
        self.update_status(paused=False)
        return resume_cmd

    def request_stop(self) -> None:
        self._stop_event.set()
        self.send_control("stop")

    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    def set_run_episode(self) -> None:
        self._run_episode_event.set()

    def check_run_episode(self) -> bool:
        if self._run_episode_event.is_set():
            self._run_episode_event.clear()
            return True
        return False

    # Agent -> API: batched episode history.

    def put_history(self, episode_data: Dict[str, Any]) -> None:
        if not self.publish:
            return
        try:
            self.history_queue.put_nowait(episode_data)
        except Exception:
            try:
                self.history_queue.get_nowait()
                self.history_queue.put_nowait(episode_data)
            except Exception:
                pass

    def get_histories(self) -> List[Dict[str, Any]]:
        result = []
        try:
            while True:
                result.append(self.history_queue.get_nowait())
        except Exception:
            pass
        return result

    # ---- Param Confirm (Agent → API) ----

    def confirm_params(self, trial_number: int) -> None:
        if not self.publish:
            return
        try:
            if not self.param_confirm_queue.empty():
                self.param_confirm_queue.get_nowait()
            self.param_confirm_queue.put_nowait(trial_number)
        except Exception:
            pass

    def check_param_confirm(self) -> Optional[int]:
        try:
            return self.param_confirm_queue.get_nowait()
        except Exception:
            return None
