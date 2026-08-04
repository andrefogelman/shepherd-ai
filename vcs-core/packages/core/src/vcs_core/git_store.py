"""Low-level pygit2 helpers for tree building and commit inspection.

This module, store.py, and internal projection storage own pygit2 use.
All functions return library types, never pygit2 types.

## Loose-object dirent recovery

`.vcscore` is expected to live on a coherent local filesystem. On
filesystems with a weakly-coherent guest-side metadata cache (Podman
machine virtiofs on macOS, 9p, some NFS configurations), the dirent for
a freshly-written loose object can be invisible to the next stat from
the same process for a few milliseconds. Consumer operations
(``TreeBuilder.insert``, ``Repository.create_commit``, ``references.create``,
``references[ref].set_target``) then fail with a libgit2 ENOENT-shaped
error.

The helpers below recover from this case with a single ``os.listdir`` of
the object's parent directory, which forces a virtiofsd round-trip and
flushes the stale dirent. This is deterministic (validated in
``spikes/260426-libgit2-virtiofs/``) and ~14x cheaper than the previous
sleep-retry approach. Healthy filesystems never hit the recovery path
because the first attempt succeeds.

This is belt-and-braces for unsupported-but-encountered deployments. A
non-zero ``loose_object_recovery_count()`` after the harness fix and on
the supported deployment matrix indicates a coherency-weak filesystem
under ``.vcscore`` and should be investigated.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import pygit2

from vcs_core._pygit2_helpers import (
    lookup_path,
    require_object,
    require_tree,
    topological_commits,
    tree_entry_id,
)
from vcs_core.types import FileChange, WorkspaceChange, normalize_git_filemode

_logger = logging.getLogger(__name__)

_recovery_counter: dict[str, int] = defaultdict(int)


def loose_object_recovery_count() -> dict[str, int]:
    """Return a snapshot of how often the dirent-cache recovery has fired, by context."""
    return dict(_recovery_counter)


def reset_loose_object_recovery_counter() -> None:
    """Clear the recovery counter. Test-only entry point."""
    _recovery_counter.clear()


def _refresh_loose_object_dirent(repo: pygit2.Repository, oid: pygit2.Oid, *, context: str) -> None:
    """Force a parent-directory readdir to flush a stale dirent for ``oid``.

    Increments the recovery counter and emits a debug log entry so operators
    can observe when this fires.
    """
    _recovery_counter[context] += 1
    _logger.debug("loose-object dirent recovery: context=%s oid=%s", context, oid)
    parent = Path(repo.path) / "objects" / str(oid)[:2]
    # Force a full readdir to flush a stale virtiofs dirent cache. Consume the
    # iterator so the syscall actually completes (lazy iterdir would not).
    with contextlib.suppress(OSError):
        for _ in parent.iterdir():
            pass


def insert_tree_entry(
    repo: pygit2.Repository,
    tb: pygit2.TreeBuilder,
    path: str,
    oid: pygit2.Oid,
    mode: int,
) -> None:
    """Insert a tree entry; recover once from a stale loose-object dirent."""
    try:
        tb.insert(path, oid, mode)
        return
    except pygit2.GitError as exc:
        original = exc
    _refresh_loose_object_dirent(repo, oid, context=f"tree entry {path!r}")
    try:
        tb.insert(path, oid, mode)
    except pygit2.GitError as inner:
        msg = f"Git object {oid} not visible while inserting tree entry {path!r}: {inner}"
        raise pygit2.GitError(msg) from original


def create_or_update_reference(
    repo: pygit2.Repository,
    ref: str,
    oid: pygit2.Oid,
    *,
    force: bool = False,
) -> None:
    """Create or update a direct ref; recover once from a stale loose-object dirent."""
    try:
        repo.references.create(ref, oid, force=force)
        return
    except pygit2.GitError as exc:
        original = exc
    _refresh_loose_object_dirent(repo, oid, context=f"reference {ref!r}")
    try:
        repo.references.create(ref, oid, force=force)
    except pygit2.GitError as inner:
        msg = f"Git object {oid} not visible while creating reference {ref!r}: {inner}"
        raise pygit2.GitError(msg) from original


def set_reference_target(
    repo: pygit2.Repository,
    ref: str,
    oid: pygit2.Oid,
) -> None:
    """Set a direct ref target; recover once from a stale loose-object dirent."""
    try:
        repo.references[ref].set_target(oid)
        return
    except pygit2.GitError as exc:
        original = exc
    _refresh_loose_object_dirent(repo, oid, context=f"reference {ref!r}")
    try:
        repo.references[ref].set_target(oid)
    except pygit2.GitError as inner:
        msg = f"Git object {oid} not visible while updating reference {ref!r}: {inner}"
        raise pygit2.GitError(msg) from original


def create_commit_with_recovery(
    repo: pygit2.Repository,
    update_ref: str | None,
    author: pygit2.Signature,
    committer: pygit2.Signature,
    message: str,
    tree_oid: pygit2.Oid,
    parents: list[pygit2.Oid],
    encoding: str | None = None,
) -> pygit2.Oid:
    """Create a commit; recover once from a stale dirent on the just-written tree.

    Mirrors ``Repository.create_commit``. The recovery path forces a readdir
    on the tree OID's parent objects directory, which is the failure mode
    observed in MG-SHAKEOUT-001 (``KeyError: object not found - no match for
    id <tree>`` raised from ``create_commit`` after a fresh ``tb.write()``).
    """

    def _call() -> pygit2.Oid:
        if encoding is None:
            return repo.create_commit(update_ref, author, committer, message, tree_oid, parents)
        return repo.create_commit(update_ref, author, committer, message, tree_oid, parents, encoding)

    try:
        return _call()
    except (pygit2.GitError, KeyError) as exc:
        original: BaseException = exc
    _refresh_loose_object_dirent(repo, tree_oid, context="commit tree")
    try:
        return _call()
    except (pygit2.GitError, KeyError) as inner:
        msg = f"Tree object {tree_oid} not visible while creating commit: {inner}"
        raise pygit2.GitError(msg) from original


def _unpack_change(change: WorkspaceChange) -> tuple[str, bytes | None, int]:
    """Unpack a workspace change tuple, defaulting mode to 100644."""
    path = change[0]
    content = change[1]
    mode = normalize_git_filemode(change[2]) if len(change) > 2 else pygit2.GIT_FILEMODE_BLOB
    return path, content, mode


# libgit2's own TreeBuilder.insert refuses a path component literally named
# ".git" (case-insensitively) — the CVE-2014-9390 hardening, still enforced
# today. A workspace's own .git directory, or a nested one from a project's
# git-sourced dependency (e.g. an Elixir `mix.exs` {:git, ...} dep checked
# out under deps/<pkg>/.git), surfaces here whenever it gets swept up as a
# workspace change to snapshot — insert_tree_entry then raises. .git is VCS
# infrastructure, never trackable workspace content: a real git worktree
# scan never yields it as trackable content either, so build_tree drops any
# change under such a component before ever building a blob/subtree for it,
# rather than letting the insert fail.
def _has_reserved_git_component(path: str) -> bool:
    return any(part.casefold() == ".git" for part in path.split("/"))


def build_tree(
    repo: pygit2.Repository,
    parent_tree_oid: pygit2.Oid | None,
    changes: tuple[WorkspaceChange, ...] | list[WorkspaceChange],
) -> pygit2.Oid:
    """Apply file changes to a parent workspace tree, return new tree OID.

    Each change is (path, content) or (path, content, git_filemode)
    where content=None means delete. When mode is absent, defaults to
    100644. Paths may be nested (e.g., "src/auth.py"). Handles arbitrary
    nesting via recursive subtree construction.

    Complexity: O(changed files), not O(total files).
    """
    nested: dict[str, list[WorkspaceChange]] = defaultdict(list)
    flat: list[tuple[str, bytes | None, int]] = []

    for change in changes:
        path, content, mode = _unpack_change(change)
        if _has_reserved_git_component(path):
            continue
        if "/" in path:
            parts = path.split("/", 1)
            rest: WorkspaceChange = (
                (parts[1], content) if mode == pygit2.GIT_FILEMODE_BLOB else (parts[1], content, mode)
            )
            nested[parts[0]].append(rest)
        else:
            flat.append((path, content, mode))

    parent_tree = require_tree(repo, parent_tree_oid, context="build_tree parent tree") if parent_tree_oid else None
    tb = repo.TreeBuilder(parent_tree) if parent_tree is not None else repo.TreeBuilder()

    for path, content, mode in flat:
        if content is None:
            try:
                tb.remove(path)
            except KeyError:
                pass
            except pygit2.GitError as exc:
                if "file isn't in the tree" not in str(exc):
                    raise
        else:
            blob_oid = repo.create_blob(content)
            insert_tree_entry(repo, tb, path, blob_oid, mode)

    for dirname, sub_changes in nested.items():
        existing_oid = None
        if parent_tree:
            existing_oid = tree_entry_id(parent_tree, dirname)
        sub_tree_oid = build_tree(repo, existing_oid, sub_changes)
        insert_tree_entry(repo, tb, dirname, sub_tree_oid, pygit2.GIT_FILEMODE_TREE)

    # Downstream consumers (insert_tree_entry on this OID, create_commit_with_recovery)
    # carry their own dirent-cache recovery; no preemptive readback is needed here.
    return tb.write()


def build_effect_meta_tree(
    repo: pygit2.Repository,
    effect_meta: dict[str, Any],
) -> pygit2.Oid:
    """Create the meta/ subtree with effect.json."""
    meta_tb = repo.TreeBuilder()
    meta_blob = repo.create_blob(json.dumps(effect_meta).encode())
    insert_tree_entry(repo, meta_tb, "effect.json", meta_blob, pygit2.GIT_FILEMODE_BLOB)
    return meta_tb.write()


def build_dual_tree(
    repo: pygit2.Repository,
    workspace_tree_oid: pygit2.Oid,
    meta_tree_oid: pygit2.Oid,
) -> pygit2.Oid:
    """Create root tree with workspace/ + meta/ subtrees."""
    root_tb = repo.TreeBuilder()
    insert_tree_entry(repo, root_tb, "workspace", workspace_tree_oid, pygit2.GIT_FILEMODE_TREE)
    insert_tree_entry(repo, root_tb, "meta", meta_tree_oid, pygit2.GIT_FILEMODE_TREE)
    return root_tb.write()


def create_signature(scope_name: str = "init") -> pygit2.Signature:
    """Create a standardized commit signature."""
    return pygit2.Signature("vcscore", f"vcscore@{scope_name}", int(time.time()), 0)


def count_between(
    repo: pygit2.Repository,
    ancestor: pygit2.Commit,
    descendant: pygit2.Commit,
) -> int:
    """Count commits between ancestor and descendant (exclusive of ancestor)."""
    count = 0
    for commit in topological_commits(repo, descendant.id):
        if commit.id == ancestor.id:
            break
        count += 1
    return count


def diff_workspace_trees(
    repo: pygit2.Repository,
    old_tree_oid: pygit2.Oid,
    new_tree_oid: pygit2.Oid,
) -> list[FileChange]:
    """Diff two workspace subtrees, returning FileChange DTOs."""
    diff = repo.diff(old_tree_oid, new_tree_oid)

    status_map = {
        pygit2.GIT_DELTA_ADDED: "added",
        pygit2.GIT_DELTA_MODIFIED: "modified",
        pygit2.GIT_DELTA_DELETED: "deleted",
    }

    files: list[FileChange] = []
    for patch in diff:
        if patch is None:
            continue
        delta = patch.delta
        path = delta.new_file.path if delta.status != pygit2.GIT_DELTA_DELETED else delta.old_file.path
        files.append(FileChange(path=path, status=status_map.get(delta.status, "unknown")))
    return files


def walk_workspace_tree(
    repo: pygit2.Repository,
    tree_oid: pygit2.Oid,
    prefix: str = "",
) -> list[tuple[str, pygit2.Oid, int]]:
    """Recursively list all blobs in a tree as (path, blob_oid, filemode) triples."""
    result: list[tuple[str, pygit2.Oid, int]] = []
    tree = require_tree(repo, tree_oid, context="workspace tree walk")
    for entry in tree:
        obj = require_object(repo, entry.id)
        if isinstance(obj, pygit2.Blob):
            result.append((prefix + str(entry.name), entry.id, entry.filemode))
        elif isinstance(obj, pygit2.Tree):
            result.extend(walk_workspace_tree(repo, entry.id, prefix + str(entry.name) + "/"))
    return result


def read_effect_json(
    repo: pygit2.Repository,
    commit: pygit2.Commit,
) -> dict[str, Any]:
    """Extract meta/effect.json from a commit. Returns {} on missing/invalid."""
    try:
        meta_blob = lookup_path(repo, commit.tree, "meta/effect.json")
        if not isinstance(meta_blob, pygit2.Blob):
            return {}
        return cast("dict[str, Any]", json.loads(meta_blob.data.decode()))
    except (KeyError, json.JSONDecodeError):
        return {}
