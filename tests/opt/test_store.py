import tomllib

from scripts.opt import configs
from scripts.opt.store import Store, canonical_toml, config_id_of, stable_id

LLAMA = ["/b/llama-server", "-m", "/m/Swift-IQ4_XS.gguf", "--host", "0.0.0.0", "--port", "8080", "-np", "2", "-c", "204800",
         "-ctk", "q8_0", "-ctv", "q8_0", "--spec-draft-n-max", "5", "-ub", "128", "--tensor-split", "1,1"]


def test_canonical_toml_roundtrips_through_tomllib():
    canon = {"engine": "llama", "model": {"label": "IQ4_XS", "sha256": "ab"}, "flags": {"np": 2, "ctk": "q8_0", "fa": True,
             "json": {"a": [1, 2]}}, "env": {}, "x": 1.5}
    back = tomllib.loads(canonical_toml(canon))
    assert back["flags"]["np"] == 2 and back["flags"]["fa"] is True and back["x"] == 1.5
    assert back["model"] == {"label": "IQ4_XS", "sha256": "ab"}
    assert "env" not in back  # empty tables are not part of the identity


def test_canonical_toml_is_order_independent():
    a = {"b": 1, "a": {"y": 1, "x": 2}}
    b = {"a": {"x": 2, "y": 1}, "b": 1}
    assert canonical_toml(a) == canonical_toml(b)
    assert config_id_of(a) == config_id_of(b)


def test_config_id_ignores_volatile_flags_and_tracks_real_ones():
    base = configs.from_command(LLAMA, None, build_sha="a" * 40)
    moved = configs.from_command([c if c != "8080" else "9999" for c in LLAMA], None, build_sha="a" * 40)
    assert base["config_id"] == moved["config_id"]
    bigger = configs.from_command([c if c != "204800" else "102400" for c in LLAMA], None, build_sha="a" * 40)
    assert bigger["config_id"] != base["config_id"]
    assert configs.from_command(LLAMA, None, build_sha="b" * 40)["config_id"] != base["config_id"]
    assert base["np"] == 2 and base["ctx_per_slot"] == 102400 and base["k"] == 5 and base["ub"] == 128
    assert tomllib.loads(base["canonical_toml"])["flags"]["ctk"] == "q8_0"


def test_store_upsert_is_idempotent_and_flush_atomic(tmp_path):
    s = Store(tmp_path / "d")
    row = {"id": "r1", "v": 1, "provenance": {"summary_path": "a/x.json", "summary_sha256": "s1"}}
    assert s.upsert("measurements", row) == "added"
    assert s.upsert("measurements", dict(row)) == "same"
    s.flush()
    before = (tmp_path / "d" / "measurements.jsonl").read_bytes()
    s2 = Store(tmp_path / "d")
    assert s2.upsert("measurements", dict(row)) == "same"
    s2.flush()
    assert (tmp_path / "d" / "measurements.jsonl").read_bytes() == before
    assert not list((tmp_path / "d").glob("*.tmp"))


def test_archive_copy_merges_path_not_rows(tmp_path):
    s = Store(tmp_path)
    a = {"id": "r", "v": 1, "provenance": {"summary_path": "results/a.json", "summary_sha256": "1"}}
    b = {"id": "r", "v": 1, "provenance": {"summary_path": "results/imports/a.json", "summary_sha256": "2"}}
    s.upsert("measurements", a)
    assert s.upsert("measurements", b) == "updated"
    row = s.tables["measurements"]["r"]
    assert row["provenance"]["summary_path"] == "results/a.json"
    assert row["provenance"]["also_at"] == ["results/imports/a.json"]
    changed = {"id": "r", "v": 2, "provenance": {"summary_path": "results/a.json", "summary_sha256": "1"}}
    assert s.upsert("measurements", changed) == "updated" and s.tables["measurements"]["r"]["v"] == 2


def test_corrupt_db_row_is_rejected(tmp_path):
    import pytest
    from scripts.opt.store import Reject
    (tmp_path / "measurements.jsonl").write_text('{"id": "a"}\n{not json\n')
    with pytest.raises(Reject):
        Store(tmp_path)


def test_stable_id_depends_on_every_part():
    assert stable_id("a", "b") == stable_id("a", "b")
    assert stable_id("a", "b") != stable_id("a", "c") != stable_id("ab", "")


def test_table_written_by_another_tool_loads_by_natural_key_and_is_not_rewritten(tmp_path):
    import pytest
    from scripts.opt.store import Reject
    lev = tmp_path / "levers.jsonl"
    lev.write_text('{"lever_id": "a", "axis": "REL"}\n')
    s = Store(tmp_path)
    assert "a" in s.tables["levers"]
    s.upsert("measurements", {"id": "m", "provenance": {}})
    s.flush()
    assert lev.read_text() == '{"lever_id": "a", "axis": "REL"}\n'
    lev.write_text('{"axis": "REL"}\n')
    s = Store(tmp_path)  # a row without lever_id loads read-only instead of crashing the DB
    assert len(s.tables["levers"]) == 1
    s.upsert("measurements", {"id": "m2", "provenance": {}})
    s.flush()
    assert lev.read_text() == '{"axis": "REL"}\n'
