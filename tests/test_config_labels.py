import pytest

from preprocessing.config import ALL_CLASSES, load_config
from preprocessing.labels import (build_labels, canonical_class_name, labels_mismatches,
                                  load_labels, write_labels)


def test_config_loading_defaults_and_hash(tmp_path):
    cfg = load_config(processed_dir=tmp_path / "p", raw_dir=tmp_path / "r")
    assert len(cfg.classes) == 26 and cfg.seed == 42
    assert cfg.manifest_path("train").name == "train.csv"
    assert cfg.config_hash() == cfg.with_overrides().config_hash()
    assert cfg.config_hash() != cfg.with_overrides(seed=7).config_hash()
    # path / worker changes must not alter the behavioural hash
    assert cfg.config_hash() == cfg.with_overrides(workers=1).config_hash()


def test_config_rejects_bad_values(tmp_path):
    with pytest.raises(ValueError):
        load_config(split_ratios=(0.5, 0.5, 0.5))
    with pytest.raises(ValueError):
        load_config(class_set="30")
    with pytest.raises(ValueError):
        load_config(raw_dir=tmp_path, processed_dir=tmp_path / "inside")


def test_label_mapping_26_and_29(tmp_path):
    cfg = load_config(processed_dir=tmp_path / "p", raw_dir=tmp_path / "r")
    l26 = build_labels(cfg)
    assert l26["classes"][0] == "A" and l26["class_to_idx"]["Z"] == 25
    assert set(l26["excluded_classes"]) == {"del", "nothing", "space"}
    l29 = build_labels(cfg.with_overrides(class_set="29"))
    assert l29["num_classes"] == 29 and l29["excluded_classes"] == {}
    assert all(l29["class_to_idx"][c] == l26["class_to_idx"][c] for c in l26["classes"])
    assert l29["class_to_idx"]["space"] == 28


def test_canonical_names():
    assert canonical_class_name("a") == "A"
    assert canonical_class_name("DEL") == "del"
    assert canonical_class_name("Nothing") == "nothing"
    assert canonical_class_name("train") is None
    assert canonical_class_name("AB") is None


def test_write_and_mismatch(tmp_path):
    cfg = load_config(processed_dir=tmp_path / "p", raw_dir=tmp_path / "r")
    write_labels(cfg)
    labels = load_labels(cfg.labels_path)
    assert labels_mismatches(labels, cfg) == []
    assert labels_mismatches(labels, cfg.with_overrides(class_set="29"))
    assert set(ALL_CLASSES) >= set(labels["classes"])
