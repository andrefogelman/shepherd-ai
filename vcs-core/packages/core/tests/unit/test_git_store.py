"""Tests for pygit2 tree-building helpers."""

from __future__ import annotations

import pygit2
import pytest
from vcs_core.git_store import (
    build_dual_tree,
    build_effect_meta_tree,
    build_tree,
    create_commit_with_recovery,
    create_or_update_reference,
    diff_workspace_trees,
    insert_tree_entry,
    loose_object_recovery_count,
    read_effect_json,
    reset_loose_object_recovery_counter,
    set_reference_target,
)


def test_build_tree_flat_files(tmp_path) -> None:
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    tree_oid = build_tree(repo, None, [("a.py", b"hello"), ("b.py", b"world")])
    tree = repo.get(tree_oid)
    assert len(tree) == 2
    assert repo.get(tree["a.py"].id).data == b"hello"
    assert repo.get(tree["b.py"].id).data == b"world"


def test_build_tree_nested_dirs(tmp_path) -> None:
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    tree_oid = build_tree(repo, None, [("src/main.py", b"code"), ("src/lib/util.py", b"util")])
    tree = repo.get(tree_oid)
    src = repo.get(tree["src"].id)
    assert repo.get(src["main.py"].id).data == b"code"
    lib = repo.get(src["lib"].id)
    assert repo.get(lib["util.py"].id).data == b"util"


def test_build_tree_incremental(tmp_path) -> None:
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    base = build_tree(repo, None, [("a.py", b"v1"), ("b.py", b"v1")])
    updated = build_tree(repo, base, [("a.py", b"v2")])
    tree = repo.get(updated)
    assert repo.get(tree["a.py"].id).data == b"v2"
    assert repo.get(tree["b.py"].id).data == b"v1"


def test_build_tree_delete(tmp_path) -> None:
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    base = build_tree(repo, None, [("a.py", b"v1"), ("b.py", b"v1")])
    updated = build_tree(repo, base, [("a.py", None)])
    tree = repo.get(updated)
    assert len(tree) == 1
    assert repo.get(tree["b.py"].id).data == b"v1"


def test_build_dual_tree(tmp_path) -> None:
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    ws = build_tree(repo, None, [("file.py", b"code")])
    meta = build_effect_meta_tree(repo, {"type": "Test", "substrate": "test"})
    root = build_dual_tree(repo, ws, meta)
    tree = repo.get(root)
    assert "workspace" in [e.name for e in tree]
    assert "meta" in [e.name for e in tree]


def test_read_effect_json(tmp_path) -> None:
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    ws = build_tree(repo, None, [])
    meta = build_effect_meta_tree(repo, {"type": "Test", "substrate": "marker"})
    root = build_dual_tree(repo, ws, meta)
    sig = pygit2.Signature("test", "test@test")
    oid = repo.create_commit(None, sig, sig, "test\n", root, [])
    commit = repo.get(oid)
    effect = read_effect_json(repo, commit)
    assert effect["type"] == "Test"
    assert effect["substrate"] == "marker"


def test_diff_workspace_trees(tmp_path) -> None:
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    old = build_tree(repo, None, [("a.py", b"v1")])
    new = build_tree(repo, old, [("a.py", b"v2"), ("b.py", b"new")])
    changes = diff_workspace_trees(repo, old, new)
    statuses = {c.path: c.status for c in changes}
    assert statuses["a.py"] == "modified"
    assert statuses["b.py"] == "added"


def test_build_tree_respects_executable_mode(tmp_path) -> None:
    """build_tree stores executable filemode when a 3-tuple is provided."""
    from vcs_core.git_store import walk_workspace_tree

    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    tree_oid = build_tree(
        repo,
        None,
        [
            ("regular.py", b"print()"),
            ("script.sh", b"#!/bin/sh\necho hi", 0o100755),
        ],
    )
    files = walk_workspace_tree(repo, tree_oid)
    by_name = {path: mode for path, _oid, mode in files}
    assert by_name["regular.py"] == pygit2.GIT_FILEMODE_BLOB  # 100644
    assert by_name["script.sh"] == pygit2.GIT_FILEMODE_BLOB_EXECUTABLE  # 100755


def test_build_tree_normalizes_user_facing_filemode(tmp_path) -> None:
    from vcs_core.git_store import walk_workspace_tree

    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    tree_oid = build_tree(repo, None, [("script.sh", b"#!/bin/sh", 100755)])

    assert walk_workspace_tree(repo, tree_oid)[0][2] == pygit2.GIT_FILEMODE_BLOB_EXECUTABLE


def test_build_tree_rejects_invalid_filemode_before_pygit2(tmp_path) -> None:
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)

    with pytest.raises(ValueError, match="Git filemode must be 100644 or 100755"):
        build_tree(repo, None, [("script.sh", b"#!/bin/sh", 123)])


def test_build_tree_drops_a_top_level_dot_git_entry(tmp_path) -> None:
    """libgit2 refuses a tree entry literally named ".git" (CVE-2014-9390's
    still-enforced hardening) — a workspace's own .git dir surfacing as a
    change used to raise pygit2.GitError out of insert_tree_entry; it must
    now be dropped instead, silently, like git's own worktree scan never
    yielding .git as trackable content."""
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    tree_oid = build_tree(
        repo, None,
        [("a.py", b"hello"), (".git/config", b"[core]\n"), (".git/HEAD", b"ref: refs/heads/main\n")],
    )
    tree = repo.get(tree_oid)
    assert [e.name for e in tree] == ["a.py"]
    assert repo.get(tree["a.py"].id).data == b"hello"


def test_build_tree_drops_a_nested_dot_git_entry(tmp_path) -> None:
    """The common real-world case: a git-sourced dependency (an Elixir
    mix.exs {:git, ...} dep is a typical trigger) plants its own .git
    several levels under a directory that isn't itself special."""
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    tree_oid = build_tree(
        repo, None,
        [
            ("deps/some_pkg/mix.exs", b"defmodule; end\n"),
            ("deps/some_pkg/.git/config", b"[core]\n"),
        ],
    )
    tree = repo.get(tree_oid)
    some_pkg = repo.get(repo.get(tree["deps"].id)["some_pkg"].id)
    assert [e.name for e in some_pkg] == ["mix.exs"]
    assert repo.get(some_pkg["mix.exs"].id).data == b"defmodule; end\n"


def test_build_tree_drops_a_dot_git_delete_change_too(tmp_path) -> None:
    """A delete change under .git must be dropped the same way an insert is
    — it can never have been legitimately present in a prior tree either,
    since inserting it would have raised at that point too."""
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    tree_oid = build_tree(repo, None, [(".git/config", None), ("a.py", b"hello")])
    tree = repo.get(tree_oid)
    assert [e.name for e in tree] == ["a.py"]


@pytest.fixture
def _clean_recovery_counter():
    reset_loose_object_recovery_counter()
    yield
    reset_loose_object_recovery_counter()


class _FakeTreeBuilder:
    """Minimal TreeBuilder stand-in. Pygit2's TreeBuilder.insert is read-only,
    so we inject a fake that the recovery helper can drive."""

    def __init__(self, *, fail_count: int) -> None:
        self.fail_count = fail_count
        self.calls: list[tuple[str, pygit2.Oid, int]] = []

    def insert(self, path: str, oid: pygit2.Oid, mode: int) -> None:
        self.calls.append((path, oid, mode))
        if len(self.calls) <= self.fail_count:
            raise pygit2.GitError("failed to insert entry: invalid object specified")


@pytest.mark.usefixtures("_clean_recovery_counter")
def test_insert_tree_entry_recovers_once_and_increments_counter(tmp_path) -> None:
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    blob = repo.create_blob(b"recovery payload")
    fake_tb = _FakeTreeBuilder(fail_count=1)

    insert_tree_entry(repo, fake_tb, "effect.json", blob, pygit2.GIT_FILEMODE_BLOB)  # type: ignore[arg-type]

    assert len(fake_tb.calls) == 2
    assert loose_object_recovery_count().get("tree entry 'effect.json'") == 1


@pytest.mark.usefixtures("_clean_recovery_counter")
def test_insert_tree_entry_no_recovery_on_healthy_path(tmp_path) -> None:
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    blob = repo.create_blob(b"healthy payload")
    tb = repo.TreeBuilder()

    insert_tree_entry(repo, tb, "effect.json", blob, pygit2.GIT_FILEMODE_BLOB)

    assert loose_object_recovery_count() == {}


@pytest.mark.usefixtures("_clean_recovery_counter")
def test_insert_tree_entry_raises_when_recovery_does_not_resolve(tmp_path) -> None:
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    blob = repo.create_blob(b"persistent failure")
    fake_tb = _FakeTreeBuilder(fail_count=2)

    with pytest.raises(pygit2.GitError, match="not visible while inserting tree entry"):
        insert_tree_entry(repo, fake_tb, "effect.json", blob, pygit2.GIT_FILEMODE_BLOB)  # type: ignore[arg-type]


@pytest.mark.usefixtures("_clean_recovery_counter")
def test_create_commit_with_recovery_passes_through_on_healthy_path(tmp_path) -> None:
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    blob = repo.create_blob(b"hello\n")
    tb = repo.TreeBuilder()
    tb.insert("README", blob, pygit2.GIT_FILEMODE_BLOB)
    tree_oid = tb.write()
    sig = pygit2.Signature("probe", "probe@example.test")

    commit_oid = create_commit_with_recovery(repo, None, sig, sig, "msg", tree_oid, [])

    assert repo.get(commit_oid) is not None
    assert loose_object_recovery_count() == {}


@pytest.mark.usefixtures("_clean_recovery_counter")
def test_create_or_update_reference_recovers_once(tmp_path) -> None:
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    blob = repo.create_blob(b"hello\n")
    tb = repo.TreeBuilder()
    tb.insert("README", blob, pygit2.GIT_FILEMODE_BLOB)
    tree_oid = tb.write()
    sig = pygit2.Signature("probe", "probe@example.test")
    commit_oid = repo.create_commit(None, sig, sig, "msg", tree_oid, [])

    state = {"failed": False}
    real_create = repo.references.create

    def flaky_create(name, target, force=False):  # type: ignore[no-untyped-def]
        if not state["failed"]:
            state["failed"] = True
            raise pygit2.GitError("target OID for the reference doesn't exist on the repository")
        return real_create(name, target, force=force)

    repo.references.create = flaky_create  # type: ignore[method-assign]
    try:
        create_or_update_reference(repo, "refs/test/recovery", commit_oid)
    finally:
        repo.references.create = real_create  # type: ignore[method-assign]

    assert state["failed"] is True
    assert loose_object_recovery_count().get("reference 'refs/test/recovery'") == 1
    # Ref should be reachable.
    assert "refs/test/recovery" in repo.references


@pytest.mark.usefixtures("_clean_recovery_counter")
def test_set_reference_target_recovers_once(tmp_path, monkeypatch) -> None:
    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    blob = repo.create_blob(b"hello\n")
    tb = repo.TreeBuilder()
    tb.insert("README", blob, pygit2.GIT_FILEMODE_BLOB)
    tree_oid = tb.write()
    sig = pygit2.Signature("probe", "probe@example.test")
    first = repo.create_commit(None, sig, sig, "first", tree_oid, [])
    second = repo.create_commit(None, sig, sig, "second", tree_oid, [first])
    repo.references.create("refs/test/moving", first)

    # `Reference.set_target` is a read-only C-extension method, so we route the
    # call through a fake references-collection. The fake's __getitem__ returns
    # an object whose set_target() fails the first time and succeeds the second.
    real_refs = repo.references
    state = {"failed": False}

    class _FakeReference:
        def set_target(self, target: pygit2.Oid) -> None:
            if not state["failed"]:
                state["failed"] = True
                raise pygit2.GitError("target OID for the reference doesn't exist on the repository")
            real_refs[ref_name].set_target(target)

    class _FakeRefs:
        def __getitem__(self, name: str) -> _FakeReference:
            return _FakeReference()

    ref_name = "refs/test/moving"
    monkeypatch.setattr(repo, "references", _FakeRefs(), raising=False)
    set_reference_target(repo, ref_name, second)
    monkeypatch.undo()

    assert state["failed"] is True
    assert loose_object_recovery_count().get("reference 'refs/test/moving'") == 1
    assert repo.references[ref_name].peel(pygit2.Commit).id == second


def test_build_tree_nested_executable(tmp_path) -> None:
    """Executable mode works through nested directory paths."""
    from vcs_core.git_store import walk_workspace_tree

    repo = pygit2.init_repository(str(tmp_path / "repo"), bare=True)
    tree_oid = build_tree(
        repo,
        None,
        [
            ("bin/run.sh", b"#!/bin/sh", 0o100755),
            ("src/lib.py", b"pass"),
        ],
    )
    files = walk_workspace_tree(repo, tree_oid)
    by_name = {path: mode for path, _oid, mode in files}
    assert by_name["bin/run.sh"] == 0o100755
    assert by_name["src/lib.py"] == 0o100644
