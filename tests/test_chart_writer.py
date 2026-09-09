from src.pipeline.chart_writer import render_chart
from src.pipeline.package import render_song_ini
from src.pipeline.types import ChartData, ChartNote, SongMeta, TempoMap


def _sample_meta() -> SongMeta:
    return SongMeta(
        name='Test "Song"',
        artist="The Band",
        album="Demo",
        genre="rock",
        year="2024",
        duration_ms=183000,
    )


def _sample_charts() -> ChartData:
    return ChartData(
        expert=[
            ChartNote(tick=192, fret=0, sustain=0),
            ChartNote(tick=192, fret=2, sustain=0),
            ChartNote(tick=384, fret=3, sustain=96),
        ],
        hard=[ChartNote(tick=192, fret=0, sustain=0)],
        medium=[ChartNote(tick=192, fret=0, sustain=0)],
        easy=[ChartNote(tick=192, fret=0, sustain=0)],
    )


def test_render_chart_has_required_sections_and_tempo():
    text = render_chart(_sample_meta(), TempoMap(bpm=120.0), _sample_charts())
    assert "[Song]" in text
    assert "[SyncTrack]" in text
    assert "[ExpertSingle]" in text
    assert "[HardSingle]" in text
    assert "[MediumSingle]" in text
    assert "[EasySingle]" in text
    assert "Resolution = 192" in text
    assert "0 = B 120000" in text
    assert "0 = TS 4" in text
    assert "MusicStream = \"song.ogg\"" in text
    assert "GuitarStream = \"guitar.ogg\"" in text


def test_render_chart_writes_notes():
    text = render_chart(_sample_meta(), TempoMap(bpm=140.5), _sample_charts())
    assert "192 = N 0 0" in text
    assert "192 = N 2 0" in text
    assert "384 = N 3 96" in text
    assert "0 = B 140500" in text


def test_render_chart_escapes_quotes():
    text = render_chart(_sample_meta(), TempoMap(bpm=120.0), _sample_charts())
    assert 'Name = "Test \'Song\'"' in text


def test_render_song_ini_fields():
    text = render_song_ini(_sample_meta())
    assert text.startswith("[song]\n")
    assert "name = Test \"Song\"" in text
    assert "artist = The Band" in text
    assert "charter = Guitar H Isolation" in text
    assert "diff_guitar = 2" in text
    assert "song_length = 183000" in text
