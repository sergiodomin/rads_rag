import pytest

from radiology_rag.artifacts import artifact_stage


def test_stage_publication_keeps_output_and_cleans_only_own_temporary_files(tmp_path):
    sibling = tmp_path / "keep.txt"
    sibling.write_text("keep", encoding="utf-8")
    with artifact_stage(tmp_path) as stage:
        (stage / "result.txt").write_text("result", encoding="utf-8")
        stage.rename(tmp_path / "published")
    assert (tmp_path / "published/result.txt").read_text() == "result"
    with pytest.raises(RuntimeError), artifact_stage(tmp_path) as stage:
        (stage / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("incomplete")
    assert sibling.read_text() == "keep"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt", "published"]


def test_stage_prefix_cannot_escape_its_parent(tmp_path):
    with pytest.raises(ValueError), artifact_stage(tmp_path, prefix="../escape-"):
        pytest.fail("Should not create an escaping stage")
