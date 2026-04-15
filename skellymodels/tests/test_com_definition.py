"""Tests for center of mass definition types and YAML loading."""

from pathlib import Path

import pytest

from skellymodels.core.biomechanics.com_definition import (
    CoMDefinition,
    SegmentCoMParameters,
)
from skellymodels.core.biomechanics import load_com_from_yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
DE_LEVA_PATH = CONFIGS_DIR / "center_of_mass" / "human_body_de_leva.yaml"


# ============================================================
# SegmentCoMParameters
# ============================================================


class TestSegmentCoMParameters:
    def test_basic_construction(self) -> None:
        seg = SegmentCoMParameters(
            rigid_body="right_upper_arm",
            com_length_ratio=0.436,
            mass_fraction=0.028,
        )
        assert seg.distal is None

    def test_with_str_distal_override(self) -> None:
        seg = SegmentCoMParameters(
            rigid_body="pelvis",
            distal="some_keypoint",
            com_length_ratio=0.5,
            mass_fraction=0.1,
        )
        assert seg.distal == "some_keypoint"

    def test_with_list_distal_override(self) -> None:
        seg = SegmentCoMParameters(
            rigid_body="thorax",
            distal=["kp_a", "kp_b"],
            com_length_ratio=0.5,
            mass_fraction=0.1,
        )
        assert seg.distal == ["kp_a", "kp_b"]

    def test_with_dict_distal_override(self) -> None:
        seg = SegmentCoMParameters(
            rigid_body="thorax",
            distal={"kp_a": 0.6, "kp_b": 0.4},
            com_length_ratio=0.5,
            mass_fraction=0.1,
        )
        assert isinstance(seg.distal, dict)

    def test_rejects_ratio_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="com_length_ratio"):
            SegmentCoMParameters(
                rigid_body="x", com_length_ratio=1.5, mass_fraction=0.1
            )

    def test_rejects_negative_mass_fraction(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            SegmentCoMParameters(
                rigid_body="x", com_length_ratio=0.5, mass_fraction=-0.1
            )


# ============================================================
# CoMDefinition
# ============================================================


class TestCoMDefinition:
    def test_valid_construction(self) -> None:
        com = CoMDefinition(
            skeleton_name="test",
            source="Test 2024",
            segments={
                "a": SegmentCoMParameters(rigid_body="rb_a", com_length_ratio=0.5, mass_fraction=0.6),
                "b": SegmentCoMParameters(rigid_body="rb_b", com_length_ratio=0.5, mass_fraction=0.4),
            },
        )
        assert com.source == "Test 2024"

    def test_rejects_empty_segments(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CoMDefinition(
                skeleton_name="test", source="x", segments={}
            )

    def test_rejects_mass_fraction_sum_too_low(self) -> None:
        with pytest.raises(ValueError, match="sum to approximately 1.0"):
            CoMDefinition(
                skeleton_name="test", source="x",
                segments={
                    "a": SegmentCoMParameters(rigid_body="rb_a", com_length_ratio=0.5, mass_fraction=0.3),
                    "b": SegmentCoMParameters(rigid_body="rb_b", com_length_ratio=0.5, mass_fraction=0.3),
                },
            )

    def test_rejects_mass_fraction_sum_too_high(self) -> None:
        with pytest.raises(ValueError, match="sum to approximately 1.0"):
            CoMDefinition(
                skeleton_name="test", source="x",
                segments={
                    "a": SegmentCoMParameters(rigid_body="rb_a", com_length_ratio=0.5, mass_fraction=0.7),
                    "b": SegmentCoMParameters(rigid_body="rb_b", com_length_ratio=0.5, mass_fraction=0.7),
                },
            )


# ============================================================
# YAML loading — De Leva
# ============================================================


class TestDeLevaCOMYAML:
    def test_loads_successfully(self) -> None:
        com = load_com_from_yaml(DE_LEVA_PATH)
        assert com.skeleton_name == "human_body"
        assert com.source == "De Leva 1996"
        assert len(com.segments) == 14

    def test_mass_fractions_sum_to_one(self) -> None:
        com = load_com_from_yaml(DE_LEVA_PATH)
        total = sum(seg.mass_fraction for seg in com.segments.values())
        assert abs(total - 1.0) < 0.02

    def test_all_distal_are_none(self) -> None:
        """De Leva uses all simple 2-keypoint segments, no distal overrides."""
        com = load_com_from_yaml(DE_LEVA_PATH)
        for seg_name, seg in com.segments.items():
            assert seg.distal is None, f"Segment '{seg_name}' has unexpected distal override"

    def test_spot_check_values(self) -> None:
        com = load_com_from_yaml(DE_LEVA_PATH)
        head = com.segments["head"]
        assert head.rigid_body == "skull"
        assert head.com_length_ratio == 0.5
        assert head.mass_fraction == 0.081

        right_thigh = com.segments["right_thigh"]
        assert right_thigh.com_length_ratio == 0.433
        assert right_thigh.mass_fraction == 0.1
