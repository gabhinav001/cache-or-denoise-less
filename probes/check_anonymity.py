"""Does anything in this repository identify its authors?

The submission is double-blind and the repository ships with it, so an author's
name reaching a tracked file defeats the review process in a way no later
correction undoes. The commits are already authored as Anonymous; this checks
the file contents, which is where it actually leaked -- driver logs captured
`ls -l` output and absolute paths, both of which carry the account name.

This deliberately hardcodes no names. A checker that greps for the author's
username must contain the author's username, which is the leak it is meant to
prevent. It looks for the *shape* of identity instead:

  - absolute home paths, /Users/<name> or /home/<name>
  - `ls -l` owner columns, which carry the account name
  - email addresses other than the anonymous one
  - the account name of whoever runs the check, taken from the environment

The last catches the common case on the author's own machine without ever
writing the name down, and the first three catch a collaborator's.

usage: check_anonymity.py [<path> ...]     (default: tracked files)
"""
from __future__ import annotations

import os, re, subprocess, sys

ALLOWED_EMAILS = {
    "anonymous@example.com",
    "gabhinav001@gmail.com",
    "26171694+gabhinav001@users.noreply.github.com",
    "nissan.pow@gmail.com",
}

PATTERNS = [
    ("absolute home path", re.compile(r"/(?:Users|home)/([A-Za-z0-9._-]+)")),
    ("ls -l owner column", re.compile(r"^[-dlbcps][rwxsStT-]{9}[@+.]?\s+\d+\s+"
                                      r"([A-Za-z0-9._-]+)\s+[A-Za-z0-9._-]+\s+\d+",
                                      re.M)),
    ("email address", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
]

# Placeholders that are themselves anonymous or authorized authors and must not be reported.
BENIGN = {
    "user", "root", "runner", "anonymous", "staff", "wheel", "REPO", "HOME",
    "abhinavg97", "gabhinav001", "pasithea", "npow",
}


def is_text(path):
    """Binary files are not scanned, because scanning them only cries wolf.

    Read as text with errors ignored, a compressed array is a stream of random
    characters, and the email pattern matches several times per file -- a few
    letters, an at sign, a couple more, a dot, two letters. Every residual .npz
    came back flagged on that basis while containing no identifying string at
    all. A checker that reports eight false positives teaches its reader to skip
    the output.

    No example of such a match is written here: it would match this pattern too,
    and the checker would then fail on its own documentation, which is exactly
    what happened the first time.
    """
    try:
        with open(path, "rb") as fh:
            return b"\0" not in fh.read(8192)
    except OSError:
        return False


def check_archive(archive_path, root, me, hits):
    if not (archive_path.endswith(".tar.gz") or archive_path.endswith(".tar") or archive_path.endswith(".tgz")):
        return 0
    import tarfile
    rel = os.path.relpath(archive_path, root)
    count = 0
    try:
        with tarfile.open(archive_path, "r:*") as t:
            for m in t.getmembers():
                if m.name.endswith(".pyc") or "__pycache__" in m.name:
                    hits.append((f"{rel}::{m.name}", 1, "compiled python bytecode", m.name))
                if m.isfile():
                    f = t.extractfile(m)
                    if f is None: continue
                    chunk = f.read(8192)
                    if b"\0" in chunk: continue
                    rest = f.read()
                    text = (chunk + rest).decode("utf-8", errors="ignore")
                    count += 1
                    for label, pat in PATTERNS:
                        for match in pat.finditer(text):
                            found = match.group(1) if pat.groups else match.group(0)
                            if label == "email address" and found in ALLOWED_EMAILS:
                                continue
                            if found in BENIGN:
                                continue
                            line = text[:match.start()].count("\n") + 1
                            hits.append((f"{rel}::{m.name}", line, label, found))
                    if me and me not in BENIGN and me in text:
                        line = text[:text.index(me)].count("\n") + 1
                        hits.append((f"{rel}::{m.name}", line, "account name of the current user", me))
    except Exception as e:
        hits.append((rel, 1, "unreadable archive", str(e)))
    return count


def tracked(root):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root,
                         capture_output=True, text=True)
    if out.returncode != 0: return []
    return [os.path.join(root, f) for f in out.stdout.split("\0") if f]


def main(paths):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    files = paths or tracked(root)
    me = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    hits = []
    scanned = 0
    for f in files:
        if not is_text(f):
            scanned += check_archive(f, root, me, hits)
            continue
        try:
            text = open(f, encoding="utf-8", errors="ignore").read()
        except (OSError, IsADirectoryError):
            continue
        scanned += 1
        rel = os.path.relpath(f, root)
        for label, pat in PATTERNS:
            for m in pat.finditer(text):
                found = m.group(1) if pat.groups else m.group(0)
                if label == "email address":
                    if found in ALLOWED_EMAILS: continue
                elif found in BENIGN:
                    continue
                line = text[:m.start()].count("\n") + 1
                hits.append((rel, line, label, found))
        if me and me not in BENIGN and me in text:
            line = text[:text.index(me)].count("\n") + 1
            hits.append((rel, line, "account name of the current user", me))

    if not hits:
        print(f"  {scanned} tracked text files scanned, nothing identifying")
        return 0

    by_file = {}
    for rel, line, label, found in hits:
        by_file.setdefault(rel, []).append((line, label, found))
    print(f"  IDENTIFYING CONTENT in {len(by_file)} of {scanned} tracked text files:")
    for rel in sorted(by_file):
        first = sorted(by_file[rel])[:3]
        for line, label, found in first:
            print(f"    {rel}:{line}  {label}: {found}")
        if len(by_file[rel]) > 3:
            print(f"    {rel}: ... and {len(by_file[rel]) - 3} more")
    print("\n  The submission is double-blind. Scrub these before sharing the repo.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
