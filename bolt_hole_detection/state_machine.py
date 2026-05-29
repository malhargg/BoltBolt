from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class SystemState(str, Enum):
    IDLE = "IDLE"
    WAITING_FOR_WINDOW = "WAITING_FOR_WINDOW"
    CAPTURING = "CAPTURING"
    PROCESSING = "PROCESSING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


_VALID_TRANSITIONS: dict[SystemState, set[SystemState]] = {
    SystemState.IDLE: {SystemState.WAITING_FOR_WINDOW, SystemState.STOPPED, SystemState.ERROR},
    SystemState.WAITING_FOR_WINDOW: {SystemState.CAPTURING, SystemState.STOPPED, SystemState.ERROR},
    SystemState.CAPTURING: {SystemState.PROCESSING, SystemState.WAITING_FOR_WINDOW, SystemState.STOPPED, SystemState.ERROR},
    SystemState.PROCESSING: {SystemState.PAUSED, SystemState.WAITING_FOR_WINDOW, SystemState.STOPPED, SystemState.ERROR},
    SystemState.PAUSED: {SystemState.PROCESSING, SystemState.STOPPED, SystemState.ERROR},
    SystemState.STOPPED: {SystemState.IDLE, SystemState.WAITING_FOR_WINDOW},
    SystemState.ERROR: {SystemState.STOPPED, SystemState.IDLE},
}


@dataclass(frozen=True)
class StateTransition:
    previous: SystemState
    current: SystemState
    reason: str


class StateMachine:
    def __init__(self, initial_state: SystemState = SystemState.IDLE) -> None:
        self._state = initial_state
        self._lock = threading.RLock()
        self._listeners: list[Callable[[StateTransition], None]] = []

    @property
    def state(self) -> SystemState:
        with self._lock:
            return self._state

    def add_listener(self, listener: Callable[[StateTransition], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def transition_to(self, new_state: SystemState, reason: str = "") -> StateTransition:
        with self._lock:
            previous = self._state
            if previous == new_state:
                transition = StateTransition(previous, new_state, reason)
                return transition
            if new_state != SystemState.ERROR and new_state not in _VALID_TRANSITIONS[previous]:
                raise ValueError(f"Invalid state transition: {previous.value} -> {new_state.value}")
            self._state = new_state
            transition = StateTransition(previous, new_state, reason)
            listeners = list(self._listeners)
        for listener in listeners:
            listener(transition)
        return transition

    def is_running_state(self) -> bool:
        return self.state in {SystemState.WAITING_FOR_WINDOW, SystemState.CAPTURING, SystemState.PROCESSING, SystemState.PAUSED}
