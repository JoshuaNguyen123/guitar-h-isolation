from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from src.pipeline.run import run_pipeline
from src.pipeline.util import default_output_dir, parse_filename_metadata

AUDIO_TYPES = [
    ("Audio files", "*.mp3 *.wav *.flac *.ogg *.m4a"),
    ("All files", "*.*"),
]

STAGE_COPY = {
    "Checking tools": "Checking FFmpeg and setup",
    "Starting": "Starting",
    "Downloading model": "Downloading the guitar model from Hugging Face",
    "Separate": "Isolating guitar from the mix",
    "Transcribe": "Hearing the guitar notes",
    "Detecting tempo": "Finding the tempo",
    "Chart": "Building the Clone Hero chart",
    "Package": "Writing the song folder",
    "Done": "Done",
}


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Guitar H Isolation")
        self.geometry("720x780")
        self.minsize(640, 700)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.configure(fg_color="#1c1914")

        self._busy = False
        self._prefetching = False
        self._output_path: Path | None = None
        self._events: queue.Queue[tuple] = queue.Queue()
        self._user_set_output = False
        self._elapsed_s = 0
        self._pulse = 0
        self._stage_name = "Ready"
        self._stage_fraction = 0.0

        self._build()
        self.after(80, self._poll_events)
        self.after(1000, self._tick_wait)
        self.after(250, self._start_prefetch)

    def start_demo(self) -> None:
        root = Path(__file__).resolve().parents[2]
        fixture = root / "tests" / "fixtures" / "guitar_chords.mp3"
        if not fixture.is_file():
            messagebox.showerror("Demo", f"Missing test clip:\n{fixture}")
            return
        self.file_var.set(str(fixture))
        self.name_var.set("Acoustic Chords")
        self.artist_var.set("Frederic Jacquot")
        self._user_set_output = True
        self.output_var.set(str(root / "e2e_output" / "Live Demo"))
        self._log("Live demo: using the short public guitar clip.")
        self._start()

    def _build(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=28, pady=(28, 8))
        ctk.CTkLabel(
            header,
            text="GUITAR H ISOLATION",
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
            text_color="#e8c36a",
        ).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text="Turn an MP3 into a Clone Hero guitar chart.",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            text_color="#c4b8a0",
        ).pack(anchor="w", pady=(4, 0))

        body = ctk.CTkFrame(self, fg_color="#262218", corner_radius=12)
        body.pack(fill="both", expand=True, padx=28, pady=(12, 20))

        self._add_label(body, "Audio file")
        file_row = ctk.CTkFrame(body, fg_color="transparent")
        file_row.pack(fill="x", padx=18, pady=(4, 10))
        self.file_var = tk.StringVar()
        self.file_entry = ctk.CTkEntry(
            file_row,
            textvariable=self.file_var,
            placeholder_text="Choose an MP3…",
            height=36,
        )
        self.file_entry.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(
            file_row,
            text="Browse",
            width=100,
            height=36,
            fg_color="#c4892a",
            hover_color="#a87120",
            text_color="#1c1914",
            command=self._browse_file,
        ).pack(side="left", padx=(10, 0))

        meta = ctk.CTkFrame(body, fg_color="transparent")
        meta.pack(fill="x", padx=18, pady=(4, 8))
        left = ctk.CTkFrame(meta, fg_color="transparent")
        right = ctk.CTkFrame(meta, fg_color="transparent")
        left.pack(side="left", fill="x", expand=True, padx=(0, 8))
        right.pack(side="left", fill="x", expand=True, padx=(8, 0))

        self._add_label(left, "Song name", inner=True)
        self.name_var = tk.StringVar()
        self.name_var.trace_add("write", self._on_name_change)
        ctk.CTkEntry(left, textvariable=self.name_var, height=36).pack(fill="x")

        self._add_label(right, "Artist", inner=True)
        self.artist_var = tk.StringVar()
        ctk.CTkEntry(right, textvariable=self.artist_var, height=36).pack(fill="x")

        self._add_label(body, "Output folder")
        out_row = ctk.CTkFrame(body, fg_color="transparent")
        out_row.pack(fill="x", padx=18, pady=(4, 16))
        self.output_var = tk.StringVar()
        ctk.CTkEntry(out_row, textvariable=self.output_var, height=36).pack(
            side="left", fill="x", expand=True
        )
        ctk.CTkButton(
            out_row,
            text="Browse",
            width=100,
            height=36,
            fg_color="#3a3428",
            hover_color="#4a4334",
            command=self._browse_output,
        ).pack(side="left", padx=(10, 0))

        self.generate_btn = ctk.CTkButton(
            body,
            text="Generate",
            height=46,
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            fg_color="#c4892a",
            hover_color="#a87120",
            text_color="#1c1914",
            command=self._start,
        )
        self.generate_btn.pack(fill="x", padx=18, pady=(4, 12))

        self.wait_banner = ctk.CTkLabel(
            body,
            text="Ready. Pick a song and click Generate.",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
            text_color="#e8c36a",
            wraplength=620,
            justify="left",
        )
        self.wait_banner.pack(anchor="w", padx=18, pady=(4, 0))
        self.stage_var = tk.StringVar(value="Ready")
        ctk.CTkLabel(
            body,
            textvariable=self.stage_var,
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color="#c4b8a0",
        ).pack(anchor="w", padx=18, pady=(4, 0))
        self.progress = ctk.CTkProgressBar(
            body,
            height=22,
            progress_color="#e8c36a",
            fg_color="#3a3428",
        )
        self.progress.pack(fill="x", padx=18, pady=(8, 12))
        self.progress.set(0)

        self.log = ctk.CTkTextbox(
            body,
            height=160,
            fg_color="#1c1914",
            text_color="#d9cbb0",
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.log.pack(fill="both", expand=True, padx=18, pady=(0, 12))
        self._log("Pick an MP3 and click Generate.")

        actions = ctk.CTkFrame(body, fg_color="transparent")
        actions.pack(fill="x", padx=18, pady=(0, 18))
        self.open_btn = ctk.CTkButton(
            actions,
            text="Open folder",
            height=36,
            fg_color="#3a3428",
            hover_color="#4a4334",
            state="disabled",
            command=self._open_folder,
        )
        self.open_btn.pack(side="left")
        self.copy_btn = ctk.CTkButton(
            actions,
            text="Copy path",
            height=36,
            fg_color="#3a3428",
            hover_color="#4a4334",
            state="disabled",
            command=self._copy_path,
        )
        self.copy_btn.pack(side="left", padx=(10, 0))
        self.copy_hint = ctk.CTkLabel(actions, text="", text_color="#e8c36a")
        self.copy_hint.pack(side="left", padx=(12, 0))

    def _add_label(self, parent, text: str, inner: bool = False) -> None:
        ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color="#c4b8a0",
        ).pack(anchor="w", padx=0 if inner else 18, pady=(10 if not inner else 0, 2))

    def _log(self, message: str) -> None:
        self.log.insert("end", message + "\n")
        self.log.see("end")

    def _browse_file(self) -> None:
        path = filedialog.askopenfilename(title="Choose audio", filetypes=AUDIO_TYPES)
        if not path:
            return
        self.file_var.set(path)
        title, artist = parse_filename_metadata(Path(path))
        self.name_var.set(title)
        self.artist_var.set(artist)
        if not self._user_set_output:
            self.output_var.set(str(default_output_dir(title)))

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Choose output folder")
        if not path:
            return
        self._user_set_output = True
        self.output_var.set(path)

    def _on_name_change(self, *_args) -> None:
        if self._user_set_output:
            return
        name = self.name_var.get().strip()
        if name:
            self.output_var.set(str(default_output_dir(name)))

    def _start(self) -> None:
        if self._busy:
            return
        src = self.file_var.get().strip()
        name = self.name_var.get().strip()
        artist = self.artist_var.get().strip()
        dest = self.output_var.get().strip()
        if not src:
            messagebox.showerror("Missing file", "Choose an audio file first.")
            return
        if not Path(src).is_file():
            messagebox.showerror("Missing file", "That audio file does not exist.")
            return
        if not dest:
            messagebox.showerror("Missing folder", "Choose an output folder.")
            return
        if not name:
            name = Path(src).stem
            self.name_var.set(name)

        self._busy = True
        self._elapsed_s = 0
        self._pulse = 0
        self._stage_name = "Starting"
        self._stage_fraction = 0.02
        self._output_path = None
        self.generate_btn.configure(
            state="disabled",
            text="Working... do not close this window",
            fg_color="#6b5420",
        )
        self.open_btn.configure(state="disabled")
        self.copy_btn.configure(state="disabled")
        self.copy_hint.configure(text="")
        self.progress.set(0.02)
        self._set_wait_text("Starting", 0.02)
        self._log(f"Generating chart for {name}. This can take several minutes.")

        thread = threading.Thread(
            target=self._worker,
            args=(Path(src), Path(dest), name, artist or "Unknown"),
            daemon=True,
        )
        thread.start()

    def _worker(self, src: Path, dest: Path, name: str, artist: str) -> None:
        def progress(stage: str, fraction: float) -> None:
            self._events.put(("progress", stage, fraction))

        try:
            output = run_pipeline(src, dest, name, artist, progress=progress)
            self._events.put(("done", output))
        except Exception as exc:
            self._events.put(("error", str(exc)))

    def _poll_events(self) -> None:
        try:
            while True:
                item = self._events.get_nowait()
                kind = item[0]
                if kind == "progress":
                    _kind, stage, fraction = item
                    self._stage_name = stage
                    self._stage_fraction = max(0.0, min(1.0, float(fraction)))
                    self._set_wait_text(stage, self._stage_fraction)
                    self._log(f"{self._friendly_stage(stage)} ({int(self._stage_fraction * 100)}%)")
                elif kind == "prefetch":
                    self._on_prefetch(item[1])
                elif kind == "done":
                    output: Path = item[1]
                    self._finish_ok(output)
                elif kind == "error":
                    self._finish_error(item[1])
        except queue.Empty:
            pass
        self.after(80, self._poll_events)

    def _start_prefetch(self) -> None:
        self._prefetching = True
        self.wait_banner.configure(
            text="Preparing the guitar model in the background. You can pick a song now."
        )
        self._log("Checking Hugging Face cache / downloading the guitar model if needed.")
        thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        thread.start()

    def _prefetch_worker(self) -> None:
        from src.pipeline.hf import prefetch_separator_model

        try:
            status = prefetch_separator_model()
            self._events.put(("prefetch", status))
        except Exception as exc:
            self._events.put(("prefetch", f"error:{exc}"))

    def _on_prefetch(self, status: str) -> None:
        self._prefetching = False
        if self._busy:
            return
        if status == "cached":
            self.wait_banner.configure(text="Ready. Guitar model is already on disk.")
            self._log("Guitar model is cached. Generate will skip the Hugging Face download.")
        elif status == "downloaded":
            self.wait_banner.configure(
                text="Ready. Guitar model finished downloading. Generate will be faster now."
            )
            self._log("Finished downloading the guitar model. Later songs skip this step.")
        elif status.startswith("error:"):
            self.wait_banner.configure(
                text="Ready. The model will download when you click Generate."
            )
            self._log(f"Background model download did not finish: {status[6:]}")

    def _friendly_stage(self, stage: str) -> str:
        return STAGE_COPY.get(stage, stage)

    def _set_wait_text(self, stage: str, fraction: float) -> None:
        friendly = self._friendly_stage(stage)
        minutes, seconds = divmod(self._elapsed_s, 60)
        clock = f"{minutes}:{seconds:02d}"
        if self._busy:
            if stage == "Downloading model" or (
                self._prefetching and stage in ("Starting", "Separate")
            ):
                self.wait_banner.configure(
                    text="Downloading the guitar model. This is one-time. Later songs reuse it."
                )
            else:
                self.wait_banner.configure(
                    text="Still working. Long songs can take 10+ minutes. Do not close this window."
                )
            self.stage_var.set(
                f"{friendly}  ·  {int(fraction * 100)}%  ·  running {clock}"
            )
        else:
            self.wait_banner.configure(text="Ready. Pick a song and click Generate.")
            self.stage_var.set(friendly)
        self.progress.set(max(0.0, min(1.0, fraction)))

    def _tick_wait(self) -> None:
        if self._busy:
            self._elapsed_s += 1
            self._pulse = (self._pulse + 1) % 16
            pulse = (self._pulse / 16.0) * 0.07
            shown = min(0.97, self._stage_fraction + pulse)
            self._set_wait_text(self._stage_name, shown)
        self.after(1000, self._tick_wait)

    def _reset_generate_btn(self) -> None:
        self.generate_btn.configure(
            state="normal",
            text="Generate",
            fg_color="#c4892a",
        )

    def _finish_ok(self, output: Path) -> None:
        self._busy = False
        self._output_path = output
        self._stage_name = "Done"
        self._stage_fraction = 1.0
        self._reset_generate_btn()
        self.open_btn.configure(state="normal")
        self.copy_btn.configure(state="normal")
        self.wait_banner.configure(text="Finished. Your Clone Hero folder is ready.")
        self.stage_var.set(f"Done  ·  took {self._elapsed_s // 60}:{self._elapsed_s % 60:02d}")
        self.progress.set(1)
        self._log(f"Wrote Clone Hero folder:\n{output}")
        self._log("Copy that folder into Clone Hero Songs, then Scan Songs.")

    def _finish_error(self, message: str) -> None:
        self._busy = False
        self._reset_generate_btn()
        self.wait_banner.configure(text="Something went wrong. Check the log below.")
        self.stage_var.set("Failed")
        self._log(f"Error: {message}")
        messagebox.showerror("Generate failed", message)

    def _open_folder(self) -> None:
        if self._output_path and self._output_path.exists():
            os.startfile(self._output_path)  # type: ignore[attr-defined]

    def _copy_path(self) -> None:
        if not self._output_path:
            return
        self.clipboard_clear()
        self.clipboard_append(str(self._output_path))
        self.copy_hint.configure(text="Copied")


def run_app(demo: bool = False) -> None:
    app = App()
    if demo:
        app.after(400, app.start_demo)
    app.mainloop()
