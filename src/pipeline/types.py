from __future__ import annotations

from dataclasses import dataclass, field


RESOLUTION = 192
CHARTER_NAME = "Guitar H Isolation"


@dataclass(frozen=True)
class NoteEvent:
    start_s: float
    end_s: float
    midi_pitch: int
    velocity: float = 1.0


@dataclass
class ChartNote:
    tick: int
    fret: int
    sustain: int = 0


@dataclass
class TempoMap:
    bpm: float
    resolution: int = RESOLUTION

    def time_to_tick(self, seconds: float) -> int:
        return int(round(seconds * (self.bpm / 60.0) * self.resolution))


@dataclass
class SongMeta:
    name: str
    artist: str
    album: str = ""
    genre: str = ""
    year: str = ""
    duration_ms: int = 0


@dataclass
class ChartData:
    expert: list[ChartNote] = field(default_factory=list)
    hard: list[ChartNote] = field(default_factory=list)
    medium: list[ChartNote] = field(default_factory=list)
    easy: list[ChartNote] = field(default_factory=list)
