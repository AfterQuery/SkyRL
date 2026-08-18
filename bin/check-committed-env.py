#!/usr/bin/env python3
"""Pre-merge guard: block a commit that ADDS a .env file containing a credential.

ENABLED in this repo by .github/workflows/committed-env.yml: the self-test, and
then this check over the files a change adds or modifies -- the PR diff on a
pull request, the pushed range on a push to main. A failure here BLOCKS the
merge -- see "What to do when it fires" at the bottom.

The push trigger is not redundant with the PR one. This guard is vendored into
repos that have never had a pull request and take commits by direct push; there,
a PR-only check would never once run.

WHY THIS EXISTS
---------------
The 2026-08-12 sweep found 75 committed .env files across 25 repos, holding 89
distinct credential values -- roughly four times the transcript/Doppler channel
this programme started with. Remediation (rotation) is a separate, larger job.
This guard is the OTHER half: it stops repo #26. Without it, every credential
rotated under 1.6r can be re-leaked by the next `git add .`, and the sweep
becomes a thing we re-run forever instead of a thing we finish.

WHAT IT FAILS ON -- AND WHAT IT DELIBERATELY DOES NOT
-----------------------------------------------------
FAILS  a file whose PATH looks like a real .env AND whose CONTENT holds a value
       matching a provider credential pattern.
PASSES `.env.example`, `.env.template`, `.env.sample`, `.env.dist` -- the files
       whose entire job is to be committed.
PASSES a real .env holding only placeholders (`your-key-here`, `xxx`, ``).
PASSES a value on credscan's KNOWN_DUMMY list -- AWS publishes
       `AKIAIOSFODNN7EXAMPLE` in its own documentation, and a guard that fails
       on vendor documentation gets switched off within a week.

BOTH HALVES OF THAT ARE LOAD-BEARING. A guard that only ever passes is not
evidence of anything, and a guard that fires on `.env.example` will be disabled
by the first person it blocks -- at which point the real .env sails through too.
`--selftest` proves both directions and is the only sanctioned way to claim this
works.

WHY PATTERN EVIDENCE ONLY (and what that costs)
-----------------------------------------------
The sweep used TWO kinds of evidence: a provider pattern (34 values) and a
secret-ish variable NAME (55 values, of which 47 adjudicated real). This guard
uses pattern evidence only, on purpose -- name evidence needs a human to judge
and cannot gate a merge without blocking `DATABASE_URL=postgres://localhost/dev`
every day.

The cost is real and must be stated rather than discovered: **this guard cannot
see `AWS_SECRET_ACCESS_KEY`.** AWS publishes no prefix for secret access keys, so
no regex can ever match one -- exactly the value the sweep found sitting beside
an access key id already on the rotation list. A name-evidence WARNING is printed
for those (non-blocking) so the reviewer sees what the gate structurally cannot.
Treating a pass here as "no secrets in this commit" is the mistake this paragraph
exists to prevent.

USAGE
  check_committed_env.py --selftest              # prove it fires AND that it doesn't
  check_committed_env.py --staged                # pre-commit: what's in the index
  check_committed_env.py --diff origin/main...HEAD   # CI: what this branch adds
  check_committed_env.py path/to/.env [...]      # explicit files

EXIT  0 clean · 1 credential found (block) · 2 the guard itself failed
      Exit 2 is NOT a pass. A guard that cannot run has not cleared anything,
      and CI must treat it as a failure rather than a skip.
"""

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import credscan  # noqa: E402  -- same detector as every sweep in this programme

# A path is "a real .env" if it is .env / .env.local / .env.prd etc, but NOT one
# of the suffixes whose purpose is to be committed. Checked against the basename
# so `config/.env.local` is caught and `docs/env.md` is not.
#
# THE LEADING DOT IS REQUIRED, and the selftest is why. An earlier version made
# it optional (`^\.?env(\....)*$`) which matched `env.md` -- a documentation file
# would have blocked a merge, and a guard that blocks docs gets switched off,
# taking the real .env check with it. All 8 basenames in the 75-file corpus
# (.env, .env.local, .env.test, .env.local.backup, ...) carry the dot. A bare
# `env` with no extension is allowed as an exact match only.
#
# Narrowing the PATH rule is the safe way to fix a false positive here;
# extending TEMPLATE_SUFFIX would not be. Every suffix added to an exclusion
# list is a new place a real credential can hide -- note that `.env.md` still
# gets scanned, because fail-safe beats tidy.
ENVISH = re.compile(r"^(\.env(\.[A-Za-z0-9_-]+)*|env)$", re.I)
TEMPLATE_SUFFIX = re.compile(
    r"\.(example|examples|sample|samples|template|templates|dist|"
    r"defaults?|schema|tpl)$", re.I)

ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.*)$")

# Name evidence -- reported as a WARNING, never a failure. See module docstring.
SECRETISH = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|PRIVATE|"
                       r"AUTH|SIGNATURE|SALT|CERT|DSN|CONNECTION_STRING|WEBHOOK)", re.I)

PLACEHOLDER = re.compile(
    r"^\s*$|^(x{3,}|y{3,}|\.{3,}|-+|_+)$|"
    r"^(your|my|the|some|insert|replace|add|put|set)[-_ ]|"
    r"(here|goes[-_ ]here|placeholder|changeme|change[-_ ]me|todo|tbd|fixme|"
    r"redacted|dummy|fake|example|sample|test[-_ ]?key|not[-_ ]?a[-_ ]?real)",
    re.I)

KNOWN_DUMMY_FP = {credscan.fingerprint(v) for v in credscan.KNOWN_DUMMY}


def is_envish(path: str) -> bool:
    """True only for files that are meant to hold real values."""
    base = os.path.basename(path)
    if TEMPLATE_SUFFIX.search(base):
        return False
    return bool(ENVISH.match(base))


def scan(path: str, text: str):
    """-> (blocking findings, non-blocking name-evidence warnings)."""
    block, warn = [], []
    hits = []
    credscan.scan_text(text, path, 0, hits)
    by_line = {}
    for h in hits:
        by_line.setdefault(text[:h["offset"]].count("\n") + 1, []).append(h)

    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        m = ASSIGN.match(line)
        if not m:
            continue
        var, val = m.group(1), m.group(2).strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]

        h = (by_line.get(i) or [None])[0]
        if h:
            if h["fp"] in KNOWN_DUMMY_FP:
                continue  # vendor-published example; see docstring
            block.append((i, var, h["pattern"], h["fp"]))
        elif SECRETISH.search(var) and val and not PLACEHOLDER.search(val) \
                and len(val) >= 16:
            warn.append((i, var, len(val)))
    return block, warn


# --------------------------------------------------------------------------
# input modes
# --------------------------------------------------------------------------
def sh(args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[guard error] {' '.join(args)}\n{r.stderr.strip()}\n"
                 f"EXIT 2 -- the guard could not run. This is NOT a pass.")
    return r.stdout


def added_files(mode, ref):
    """Only files this change ADDS or MODIFIES. An existing committed .env is
    the sweep's problem, not this gate's -- failing every build on pre-existing
    debt is how a guard gets reverted on day one."""
    if mode == "staged":
        out = sh(["git", "diff", "--cached", "--name-only", "--diff-filter=AM"])
    else:
        out = sh(["git", "diff", "--name-only", "--diff-filter=AM", ref])
    return [p for p in out.splitlines() if p.strip()]


def read(path, from_ref=None):
    """Read a candidate's content -- from a git ref when one is given.

    The working tree is not always a reliable place to read from. This guard is
    vendored into repos too large to check out in full: AfterQuery-Axiom is
    12.5 GB, and a `fetch-depth: 0` checkout there was CANCELLED at the job
    timeout, leaving a check that could never once run. The fix is a blobless,
    sparse checkout -- which means the changed .env is NOT on disk.

    Reading it out of git instead makes that safe. Under a partial clone
    `git show` lazily fetches exactly the one blob it needs, so the cost is a
    single credential-sized file rather than the repository.

    The diff modes already filter to added/modified paths, so a `git show` that
    fails here is a real fault, not a deletion. It returns None and the caller
    reports it as unmeasured (exit 2), never as clean.
    """
    if from_ref:
        r = subprocess.run(["git", "show", f"{from_ref}:{path}"],
                           capture_output=True, text=True, errors="replace")
        return r.stdout if r.returncode == 0 else None
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.read()
    except (OSError, UnicodeError):
        return None


# --------------------------------------------------------------------------
# selftest -- BOTH directions, because a guard that only passes proves nothing
# --------------------------------------------------------------------------
def selftest():
    print("=== selftest ===")
    ok = True

    def case(label, path, text, want_block):
        nonlocal ok
        if not is_envish(path):
            got = False
            detail = "path not env-ish (not scanned)"
        else:
            b, _ = scan(path, text)
            got = bool(b)
            detail = f"{len(b)} finding(s)"
        good = got == want_block
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {label:44s} "
              f"expected {'BLOCK' if want_block else 'allow'}, "
              f"got {'BLOCK' if got else 'allow'} ({detail})")

    # POSITIVE -- these must fail the build. Synthetic values, never real keys.
    case("real .env with an OpenAI project key", ".env",
         "OPENAI_API_KEY=sk-proj-" + "A" * 48 + "\n", True)
    case("nested .env.local with an AWS key id", "svc/.env.local",
         "AWS_ACCESS_KEY_ID=AKIA" + "B" * 16 + "\n", True)
    case(".env.production with a Doppler CLI token", ".env.production",
         "DOPPLER_TOKEN=dp.ct." + "C" * 40 + "\n", True)
    case("credential on line 3, quoted", ".env",
         "NODE_ENV=production\nPORT=3000\nSTRIPE='sk_live_" + "D" * 30 + "'\n", True)

    # NEGATIVE -- these must NOT fail, or the guard gets switched off.
    case(".env.example with a placeholder", ".env.example",
         "OPENAI_API_KEY=your-key-here\n", False)
    case(".env.template with an empty value", ".env.template",
         "STRIPE_SECRET_KEY=\n", False)
    case("real .env, placeholders only", ".env",
         "OPENAI_API_KEY=your-key-here\nDB_PASSWORD=changeme\n", False)
    case("real .env, ordinary local config", ".env",
         "NODE_ENV=development\nDATABASE_URL=postgres://localhost:5432/dev\n", False)
    case("AWS's own published example key", ".env",
         "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n", False)
    case("a doc that merely mentions env", "docs/env.md",
         "OPENAI_API_KEY=sk-proj-" + "A" * 48 + "\n", False)
    case("commented-out credential", ".env",
         "# OPENAI_API_KEY=sk-proj-" + "A" * 48 + "\n", False)

    # The path rule must recognise every basename the real corpus actually
    # contains -- a rule tested only against invented names is a rule tested
    # against my imagination. These are the 8 distinct basenames among the 75
    # committed .env files found on 2026-08-12.
    corpus = [".env", ".env.afterqueryai", ".env.development", ".env.local",
              ".env.local.backup", ".env.miniswe", ".env.production", ".env.test"]
    missed = [b for b in corpus if not is_envish(b)]
    ok = ok and not missed
    print(f"  {'PASS' if not missed else 'FAIL'}  "
          f"{'all 8 real corpus basenames recognised':44s} "
          + ("none missed" if not missed else f"MISSED {missed}"))

    # The stated blind spot, asserted rather than assumed. If a future credscan
    # release adds an AWS-secret pattern this flips to FAIL and the docstring
    # above needs correcting -- which is the point of testing a known gap.
    b, w = scan(".env", "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI" + "/" + "K" * 27 + "\n")
    blind = not b and bool(w)
    ok = ok and blind
    print(f"  {'PASS' if blind else 'FAIL'}  "
          f"{'documented blind spot: AWS secret key':44s} "
          f"expected warn-not-block, got "
          f"{len(b)} block / {len(w)} warn")

    # The guard PRINTS context for every finding, so sanitise() is part of the
    # guard's own attack surface: a scrubber that emits key material re-commits
    # the exposure it was run to report. Truncation is the case that breaks it --
    # the context window is a fixed byte slice, so it routinely starts part-way
    # through a neighbouring credential, and a fragment has no provider prefix
    # for any pattern to anchor on. It then falls to the length floor alone,
    # which is why the floor is tested here rather than trusted.
    #
    # Found by CI on this very PR, and confirmed by measurement rather than
    # argument: at the original floor of 24, cuts leaving a 16-23 character tail
    # emitted it verbatim -- 44 leaks over 268 cuts, longest survivor 23 chars.
    fragments, worst = 0, 0
    for key in ("sk-ant-api03-" + "aB3_x-9QzR7mK2pW" * 6,
                "sk-proj-" + "T4vB8nQ2xL9pR5mK7wD3zJ6yF1sG0hC" * 2,
                "ghp_" + "aB3x9QzR7mK2pW5nJ8vB4cL6hD1yF0",
                "AIza" + "SyD-9QzR7mK2pW5nJ8vB4cL6hD1yF0sG"):
        for cut in range(len(key) - 12):          # every prefix-stripping cut
            frag = key[cut:]
            out = credscan.sanitise(f"ANTHROPIC_API_KEY_PAINT={frag} # trailing prose")
            for n in range(len(frag), 11, -1):
                if frag[:n] in out:
                    fragments += 1
                    worst = max(worst, n)
                    break
    ok = ok and not fragments
    print(f"  {'PASS' if not fragments else 'FAIL'}  "
          f"{'no truncated key fragment survives scrub':44s} "
          + ("0 leaked over every cut of 4 key shapes"
             if not fragments else
             f"{fragments} LEAK(S), longest {worst} chars"))

    # ...and the leak test must be able to SEE a leak, or its zero is vacuous.
    # A scrubber test that would pass against a no-op scrubber proves nothing.
    canary = "aB3_x-9QzR7mK2pW5nJ8vB4c"
    saw = canary in f"ANTHROPIC_API_KEY_PAINT={canary}"
    gone = canary not in credscan.sanitise(f"ANTHROPIC_API_KEY_PAINT={canary}")
    ok = ok and saw and gone
    print(f"  {'PASS' if saw and gone else 'FAIL'}  "
          f"{'...and that test can detect a leak':44s} "
          f"present before scrub={saw}, absent after={gone}")

    # The label must SURVIVE, or the scrub blinds adjudication instead of
    # protecting it -- lowering the floor must not start eating variable names.
    kept = "ANTHROPIC_API_KEY_PAINT" in credscan.sanitise(
        "ANTHROPIC_API_KEY_PAINT=aB3_x-9QzR7mK2pW5nJ8vB4c")
    ok = ok and kept
    print(f"  {'PASS' if kept else 'FAIL'}  "
          f"{'...while the variable NAME survives':44s} "
          f"SHOUT_CASE label kept={kept}")

    # --read-from only ever runs in the big repos, which is exactly where nobody
    # will notice it silently reading nothing. A `git show` of a path that is not
    # in the working tree must return the content; a bad ref must return None so
    # the caller can exit 2 rather than call an unread file clean.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        def g(*args):
            return subprocess.run(["git", "-C", td, *args],
                                  capture_output=True, text=True)
        g("init", "-q")
        g("config", "user.email", "selftest@local")
        g("config", "user.name", "selftest")
        open(os.path.join(td, ".env"), "w").write("OPENAI_API_KEY=sk-proj-" + "A" * 48 + "\n")
        g("add", "-f", ".env")
        g("commit", "-q", "-m", "x")
        os.remove(os.path.join(td, ".env"))     # gone from the working tree
        cwd = os.getcwd()
        try:
            os.chdir(td)
            from_git = read(".env", "HEAD")
            from_tree = read(".env")
            bad_ref = read(".env", "no-such-ref")
        finally:
            os.chdir(cwd)
    got = bool(from_git and "sk-proj-" in from_git)
    tree_none = from_tree is None
    bad_none = bad_ref is None
    ok = ok and got and tree_none and bad_none
    print(f"  {'PASS' if got and tree_none and bad_none else 'FAIL'}  "
          f"{'--read-from reads a file not on disk':44s} "
          f"git={'content' if got else 'EMPTY'}, "
          f"tree={'None' if tree_none else 'unexpectedly readable'}, "
          f"bad ref={'None' if bad_none else 'NOT None -- would read as clean'}")

    print(f"\n  {'SELFTEST PASSED' if ok else 'SELFTEST FAILED'} -- "
          f"the guard fires on real credentials, leaves templates alone, and "
          f"does not\n  leak key material into its own output.")
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--staged", action="store_true")
    ap.add_argument("--diff", metavar="REF")
    ap.add_argument("--selftest", action="store_true")
    # Read candidate content out of a git ref instead of the working tree, so
    # the guard can run under a blobless sparse checkout in a repo too large to
    # check out in full. See read().
    ap.add_argument("--read-from", metavar="REF", default=None)
    a = ap.parse_args()

    if a.read_from and not (a.diff or a.staged):
        ap.error("--read-from applies to --diff/--staged, not to explicit paths")

    if a.selftest:
        return selftest()

    if a.staged:
        cands = added_files("staged", None)
    elif a.diff:
        cands = added_files("diff", a.diff)
    elif a.paths:
        cands = a.paths
    else:
        ap.error("give --staged, --diff REF, --selftest, or explicit paths")

    # An explicitly-named path that does not exist is UNMEASURED, never clean.
    # Found by testing: a shell that failed to word-split its file list handed
    # this guard one multi-line blob, which is not env-shaped, so it printed
    # "OK -- no .env-shaped file" having scanned NOTHING. A guard whose happy
    # path and its did-not-run path print the same thing is worse than absent.
    # git-derived modes are exempt: git names deleted files too.
    if a.paths and not (a.staged or a.diff):
        missing = [p for p in cands if not os.path.isfile(p)]
        if missing:
            print(f"[guard] EXIT 2 -- {len(missing)} named path(s) do not exist, "
                  f"e.g. {missing[0][:80]!r}. Nothing was scanned; this is not a pass.")
            return 2

    envs = [p for p in cands if is_envish(p)]
    print(f"[guard] {len(cands)} changed file(s), {len(envs)} env-shaped")
    if not envs:
        print("[guard] OK -- no .env-shaped file added or modified")
        return 0

    failed, warned = [], []
    for p in envs:
        text = read(p, a.read_from)
        if text is None:
            # Unreadable is UNMEASURED, never clean. Same rule as every sweep.
            src = f"{a.read_from}:{p}" if a.read_from else p
            print(f"[guard] EXIT 2 -- could not read {src}. Unreadable is not clean.")
            return 2
        b, w = scan(p, text)
        for ln, var, pat, fp in b:
            failed.append((p, ln, var, pat, fp))
        for ln, var, n in w:
            warned.append((p, ln, var, n))

    for p, ln, var, n in warned:
        print(f"[guard] WARN  {p}:{ln}  {var} -- secret-shaped NAME, "
              f"{n} chars, no provider pattern. Not blocking; a human must judge. "
              f"(AWS secret keys land here and can never be pattern-matched.)")

    if not failed:
        print(f"[guard] OK -- {len(envs)} env file(s) scanned, no credential "
              f"pattern matched."
              + (f" {len(warned)} name-evidence warning(s) above." if warned else ""))
        return 0

    print(f"\n[guard] BLOCKED -- {len(failed)} credential(s) in a committed .env:\n")
    for p, ln, var, pat, fp in failed:
        # fingerprint only; the value never leaves the file
        print(f"    {p}:{ln}  {var}  [{pat}]  fp={fp}")
    print("""
    Do NOT just delete the line and re-commit. If it was ever pushed the value
    is in git history permanently and rewriting history does not un-leak it:

      1. ROTATE the credential at the provider. This is the only remedy.
      2. Put the new value in Doppler, not in a file.
      3. Add the path to .gitignore so this cannot recur.
      4. Tell #security so it lands on the rotation roster.""")
    # This guard is vendored into several repos and only one of them has a
    # SECRETS.md. Pointing a blocked developer at a file that is not there is
    # how the whole message stops being believed -- so the reference is printed
    # only where it resolves.
    if os.path.isfile("SECRETS.md"):
        print("\n    See SECRETS.md §1 and §10.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

# ---------------------------------------------------------------------------
# What to do when it fires
#
#   1. ROTATE the credential at the provider. Deleting the line does not help:
#      if the branch was ever pushed the value is in git history permanently,
#      and rewriting history does not un-leak it.
#   2. Put the new value in Doppler, never in a file.
#   3. Add the path to .gitignore so it cannot recur.
#   4. Tell #security so it lands on the rotation roster.
#
# Running it yourself, before you push:
#   python3 bin/check-committed-env.py --staged                  # the index
#   python3 bin/check-committed-env.py --diff origin/main...HEAD # your branch
#   python3 bin/check-committed-env.py --selftest                # the guard
#
# Optional local hook (advisory -- `git commit --no-verify` bypasses it):
#   .git/hooks/pre-commit  ->  python3 bin/check-committed-env.py --staged
#
# credscan.py beside this file is the SAME detector the credential audit uses.
# Keep them in step: a second, independently-written matcher would eventually
# disagree with the audit's published numbers with no way to tell which is right.
# ---------------------------------------------------------------------------
