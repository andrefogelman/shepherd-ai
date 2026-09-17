# under-test: vcs_core._clonefile_carrier
"""B1c — the APFS clonefile CarrierBackend (macOS member of the reversibility axis).

Backend-level coverage: CoW clone of the base, clone-vs-base diff (add/modify/delete +
exec bit), child-scope parent-relative diff + commit-into-parent, discard, and push
materialization to the real workspace. macOS-gated (APFS clonefile; Linux uses fuse/kernel).
"""

from __future__ import annotations

import os
import sys

import pytest
from vcs_core._clonefile_carrier import ClonefileCarrierBackend
from vcs_core._substrate_runtime import CarrierBackend

_macos = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="clonefile is macOS/APFS; the Linux carriers are fuse/kernel overlay",
)


def _make(tmp_path, seed=None):
    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    for rel, content in (seed or {"a.txt": b"A", "sub/b.txt": b"B"}).items():
        target = base / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return ClonefileCarrierBackend(workspace=ws, state_root=tmp_path / "state", base_lowerdir=base)


@_macos
def test_clonefile_conforms_and_clones_base_into_ground(tmp_path) -> None:
    backend = _make(tmp_path)
    assert isinstance(backend, CarrierBackend)  # structurally satisfies the carrier protocol
    backend.create_layer("ground", parent_scope_id=None)
    wp = backend.working_path("ground")
    assert (wp / "a.txt").read_bytes() == b"A"
    assert (wp / "sub" / "b.txt").read_bytes() == b"B"
    assert backend.has_layer("ground")
    assert backend.diff_layer("ground") == []  # a fresh clone equals its base


@_macos
def test_clonefile_diff_detects_add_modify_delete_and_exec_bit(tmp_path) -> None:
    backend = _make(tmp_path)
    backend.create_layer("ground", parent_scope_id=None)
    backend.write_file("ground", "c.txt", b"C")  # add
    backend.write_file("ground", "a.txt", b"A2", mode=0o100755)  # modify + exec bit
    backend.delete_file("ground", "sub/b.txt")  # delete
    diff = {path: (content, mode) for path, content, mode in backend.diff_layer("ground")}
    assert diff["c.txt"] == (b"C", 0o100644)
    assert diff["a.txt"] == (b"A2", 0o100755)
    assert diff["sub/b.txt"] == (None, 0)


@_macos
def test_clonefile_child_scope_diff_is_parent_relative_and_commits(tmp_path) -> None:
    backend = _make(tmp_path)
    backend.create_layer("ground", parent_scope_id=None)
    backend.create_layer("task", parent_scope_id="ground")
    backend.write_file("task", "a.txt", b"EDITED")  # change vs ground (== base)
    assert backend.diff_layer("task") == [("a.txt", b"EDITED", 0o100644)]  # only the child's delta
    backend.commit_layer("task", into_scope_id="ground")
    assert backend.read_file("ground", "a.txt") == b"EDITED"  # applied into ground
    assert not backend.has_layer("task")  # child discarded on commit


@_macos
def test_clonefile_discard_drops_clone_leaves_ground(tmp_path) -> None:
    backend = _make(tmp_path)
    backend.create_layer("ground", parent_scope_id=None)
    backend.create_layer("task", parent_scope_id="ground")
    assert backend.has_layer("task")
    backend.discard_layer("task")
    assert not backend.has_layer("task")
    assert backend.has_layer("ground")  # ground untouched by a child discard


@_macos
def test_clonefile_push_materializes_diff_to_real_workspace(tmp_path) -> None:
    backend = _make(tmp_path)
    backend.create_layer("ground", parent_scope_id=None)
    backend.write_file("ground", "out.txt", b"OUT")
    backend.write_file("ground", "a.txt", b"A-NEW")
    backend.push_layer("ground")
    ws = tmp_path / "ws"
    assert (ws / "out.txt").read_bytes() == b"OUT"  # the delta materialized to the real workspace
    assert (ws / "a.txt").read_bytes() == b"A-NEW"
    assert backend.diff_layer("ground") == []  # ground reset to base after push


@_macos
def test_clonefile_diff_skips_symlinks(tmp_path) -> None:
    """`ClonefileCarrierBackend` HERDA de `CopyCarrierBackend`, então herda também
    o `_scan_dir` que pula symlink em vez de recusá-lo. Este teste existe para
    dizer isso explicitamente: quando o comportamento do pai mudou, este arquivo
    continuou afirmando o contrato antigo e ficou vermelho sem ninguém olhar.

    A perda é a mesma do carrier de cópia: um symlink criado durante um run NÃO
    entra no changeset."""
    backend = _make(tmp_path)
    backend.create_layer("ground", parent_scope_id=None)
    (backend.working_path("ground") / "link").symlink_to("a.txt")
    backend.write_file("ground", "c.txt", b"C")

    diff = {path: (content, mode) for path, content, mode in backend.diff_layer("ground")}
    assert diff == {"c.txt": (b"C", 0o100644)}


@_macos
def test_clonefile_diff_still_rejects_exotic_kinds(tmp_path) -> None:
    """Fifo, socket e device continuam levantando — o skip vale só para symlink."""
    from vcs_core import UnsupportedOverlayEntryError

    backend = _make(tmp_path)
    backend.create_layer("ground", parent_scope_id=None)
    os.mkfifo(backend.working_path("ground") / "pipe")
    with pytest.raises(UnsupportedOverlayEntryError):
        backend.diff_layer("ground")


@_macos
def test_clonefile_child_of_unmaterialized_parent_clones_the_base(tmp_path) -> None:
    """B3c-3 regression: a child whose parent never materialized a clone of its
    own (ground, in normal operation) clones the BASE snapshot — "a child scope
    clones its parent's working tree" holds transitively. Before the fix the
    child cloned an empty tree, so a body saw none of the pre-existing files."""
    backend = _make(tmp_path)
    # No create_layer("ground") — the parent works directly on the base.
    backend.create_layer("run", parent_scope_id="ground")
    wp = backend.working_path("run")
    assert (wp / "a.txt").read_bytes() == b"A"
    assert (wp / "sub" / "b.txt").read_bytes() == b"B"
    assert backend.diff_layer("run") == []  # the fresh child equals its effective base


@_macos
def test_clonefile_commit_into_missing_destination_materializes_full_tree(tmp_path) -> None:
    """B3c-3 regression: committing a child into a destination with no clone
    materializes the destination as a FULL tree first — never a delta-only dir
    a later fork would mistake for the whole workspace."""
    backend = _make(tmp_path)
    backend.create_layer("run", parent_scope_id="ground")
    backend.write_file("run", "new.txt", b"NEW")
    backend.commit_layer("run", into_scope_id="ground")
    ground = backend.working_path("ground")
    assert (ground / "new.txt").read_bytes() == b"NEW"  # the delta
    assert (ground / "a.txt").read_bytes() == b"A"  # ... and the full base tree
    # A subsequent fork composes on the merged state.
    backend.create_layer("run2", parent_scope_id="ground")
    wp2 = backend.working_path("run2")
    assert (wp2 / "new.txt").read_bytes() == b"NEW"
    assert (wp2 / "a.txt").read_bytes() == b"A"
