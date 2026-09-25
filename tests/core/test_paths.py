"""Migration guard for historical report inputs."""

from acoustic_freeform.core import paths


def test_historical_input_resolves_from_outside_repository(tmp_path, monkeypatch):
    moved = tmp_path / "artifacts" / "studies" / "S04" / "case" / "source.npz"
    moved.parent.mkdir(parents=True)
    moved.write_bytes(b"saved source")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(paths, "REPOSITORY", tmp_path)
    monkeypatch.setattr(
        paths,
        "_relocations",
        lambda: (("artifacts/old", "artifacts/studies/S04"),),
    )
    assert paths.data_path("artifacts/old/case/source.npz") == moved
    assert paths.data_path("artifacts/studies/S04/case/source.npz") == moved
