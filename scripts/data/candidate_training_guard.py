"""Fail-closed action checks for A3 candidate artifacts.

This guard is deliberately independent from the legacy formal-v2 exporter: a
candidate can be audited and copied to a handoff directory, but cannot be
mistaken for a qualified training export by a caller that invokes this module.
"""
from __future__ import annotations

from typing import Any


def check_action(manifest: dict[str, Any], action: str) -> tuple[bool, str]:
    state = manifest.get("content_state")
    if state != "CANDIDATE_NOT_FROZEN":
        return False, "candidate manifest state is not explicitly CANDIDATE_NOT_FROZEN"
    if action in {"train", "remote_sft", "product_sft", "release"}:
        return False, "candidate artifacts require a qualified export and a separately authorized run"
    if action in {"audit", "local_rebuild", "handoff"}:
        return True, "candidate action is preparation-only"
    return False, f"unknown action {action!r} fails closed"


def require_preparation_action(manifest: dict[str, Any], action: str) -> None:
    allowed, reason = check_action(manifest, action)
    if not allowed:
        raise PermissionError(reason)
