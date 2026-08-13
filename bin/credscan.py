#!/usr/bin/env python3
"""
Credential exposure scanner for local agent transcripts (audit task 1.6).

HARD RULE: this program never emits a credential value. It emits pattern name,
file, line, offset, sha256[:12] of the value, and the value's length. Context
strings are re-scanned and every credential-shaped substring inside them is
replaced before they are written anywhere. The session running this scanner is
itself recorded to a transcript that live-syncs to Quill, so a single careless
print re-commits the exposure it is measuring.
"""
import hashlib
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Patterns.
#
# Each is deliberately anchored on a provider-issued prefix plus a minimum body
# length. Bare-entropy detection (e.g. "any 32 hex chars") is not used: it
# drowns the result set in git SHAs, docker digests and UUIDs, and an
# adjudication pass that has to reject thousands of rows stops being read.
# ---------------------------------------------------------------------------
PATTERNS = {
    "anthropic_api_key":   r"sk-ant-api[0-9]{2}-[A-Za-z0-9_\-]{80,}",
    "anthropic_admin_key": r"sk-ant-admin[0-9]{2}-[A-Za-z0-9_\-]{80,}",
    "openai_project_key":  r"sk-proj-[A-Za-z0-9_\-]{40,}",
    "openai_legacy_key":   r"sk-[A-Za-z0-9]{48}(?![A-Za-z0-9])",
    # Found only because it sat beside a GOOGLE_API_KEY in an adjudication
    # context -- it was in the corpus the whole time with no pattern to catch
    # it. Every provider absent from this table is a silent zero.
    "openrouter_key":      r"sk-or-v1-[a-f0-9]{40,}",
    "aws_access_key_id":   r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])",
    "google_api_key":      r"AIza[0-9A-Za-z_\-]{35}(?![0-9A-Za-z_\-])",
    # `ct` (CLI token) was MISSING until 2026-08-12. That is the type
    # `doppler configure` stores on a developer machine, so every sweep run
    # before this date -- 1.6's roster rebuild and the workplace exposure sweep
    # included -- was structurally blind to the most commonly leaked Doppler
    # token type. Absence of CLI tokens from those results is not evidence.
    "doppler_token":       r"dp\.(?:st|pt|ct|sa|scim|audit)\.[A-Za-z0-9_\-]{20,}",
    "github_token":        r"gh[pousr]_[A-Za-z0-9]{36,}",
    "gitlab_pat":          r"glpat-[A-Za-z0-9_\-]{20,}",
    "slack_token":         r"xox[baprse]-[A-Za-z0-9\-]{20,}",
    "stripe_secret_key":   r"[sr]k_live_[A-Za-z0-9]{24,}",
    "sendgrid_key":        r"SG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}",
    "resend_key":          r"(?<![A-Za-z0-9])re_[A-Za-z0-9]{8,}_[A-Za-z0-9]{20,}",
    "sentry_auth_token":   r"sntrys_[A-Za-z0-9_\-=]{40,}",
    "posthog_personal":    r"phx_[A-Za-z0-9]{40,}",
    "linear_api_key":      r"lin_api_[A-Za-z0-9]{40,}",
    "npm_token":           r"npm_[A-Za-z0-9]{36}",
    "twilio_api_key":      r"(?<![A-Za-z0-9])SK[0-9a-f]{32}(?![0-9a-f])",
    "private_key_block":   r"-----BEGIN [A-Z ]{0,24}PRIVATE KEY-----"
                           r"(?:[^-]|-(?!----END))*?"
                           r"-----END [A-Z ]{0,24}PRIVATE KEY-----",
}
COMPILED = {n: re.compile(p) for n, p in PATTERNS.items()}

# A bare BEGIN marker with no closing END. Tracked separately: 13 files carry a
# marker but only a couple hold a real key, so counting markers as credentials
# would inflate the result ~6x. These rows exist to be adjudicated, not counted.
MARKER_ONLY = re.compile(r"-----BEGIN [A-Z ]{0,24}PRIVATE KEY-----")

CONTEXT = 90  # chars of sanitised context kept on each side, for adjudication

# Values that are publicly documented non-credentials -- vendor examples that
# appear verbatim in their own docs and in every tutorial that copies them.
#
# These are FLAGGED, not suppressed. Dropping a match outright would also drop
# the row that proves the scanner looked there, and a denylist that silently
# eats matches is one typo away from eating a real key. The adjudication pass
# reads `dummy` and files them as third-party-sample.
KNOWN_DUMMY = {
    "AKIAIOSFODNN7EXAMPLE",
    "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "AKIAI44QH8DHBEXAMPLE",
}


# Shortest base64 body that could be a real private key. A 1024-bit RSA key in
# PKCS#8 is ~800 base64 chars; 2048-bit is ~1600.
#
# This threshold exists because BEGIN/END armour around an ELIDED body still
# satisfies the block regex. Observed in the wild: a 298-character log line
# reading `private_key: -----BEGIN PRIVATE KEY----- ... -----END PRIVATE KEY-----`
# matched as a complete key. Without this, deliberately-redacted diagnostics --
# the safe practice -- get counted as exposures and inflate the rotation list
# with keys that never actually leaked.
MIN_PEM_BODY = 200


def pem_body(value: str) -> str:
    inner = re.sub(r"-----(BEGIN|END) [A-Z ]{0,24}PRIVATE KEY-----", "", value)
    return re.sub(r"(\\+n|\s|\.{2,})", "", inner)


def normalise(name: str, value: str) -> str:
    """Canonical form to fingerprint.

    PEM bodies reach transcripts in several encodings of the same key -- raw
    newlines, JSON-escaped \\n, doubly-escaped \\\\n -- so without stripping
    them one key fingerprints as three and the rotation list triple-counts.
    """
    if name == "private_key_block":
        return re.sub(r"(\\+n|\s)", "", value)
    return value


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:12]


# Any long run of key-alphabet characters. Deliberately shape-blind.
#
# Pattern-based scrubbing alone is NOT sufficient, and assuming it was leaked
# real key material into this audit's own output. A context window is a fixed
# byte slice, so it routinely BEGINS PART-WAY THROUGH a neighbouring
# credential -- the provider prefix that every pattern anchors on sits outside
# the window, the pattern therefore does not match, and 50+ characters of a
# real key survive scrubbing and get printed.
#
# The fix cannot be a better prefix list; a truncated secret has no prefix by
# construction. It has to be entropy-shaped: redact long opaque runs whether or
# not they are recognisable. Adjudication only needs the surrounding prose
# ("ANTHROPIC_API_KEY_PAINT=", "FAKE CONTROL FILE"), never the operand.
#
# THE THRESHOLD IS PART OF THE FIX, and 24 was too high. Truncation does not
# just remove the prefix, it shortens the fragment, so the same cut that defeats
# the pattern can also drop the remainder under a length floor -- at 24, every
# cut leaving a 16-23 character tail emitted that tail verbatim. Measured over
# 268 cuts across six provider shapes: 44 leaks, longest surviving run 23 chars.
# A guard whose floor sits above the fragments truncation actually produces is
# not a guard. 12 is below any usable key remnant and still above ordinary
# prose; SHOUT_CASE identifiers are exempted separately just below, and those
# are the tokens adjudication actually needs.
OPAQUE_RUN = re.compile(r"[A-Za-z0-9+/_\-]{12,}")

# ...but an ALL-CAPS run of letters, digits and underscores is an environment
# variable name, not a value: OFFICEQA_V2_ANTHROPIC_API_KEY, not a key.
#
# These must survive scrubbing. The variable name is frequently the ONLY thing
# that distinguishes a live credential from a test fixture, so redacting it
# blinds the adjudication step and pushes real keys into "unknown". Issued
# secrets are mixed-case base64 or lowercase hex; none are SHOUT_CASE, and the
# one uppercase family that exists (AKIA...) is 20 chars and has its own rule.
IDENTIFIER = re.compile(r"^[A-Z][A-Z0-9_]*$")


def sanitise(text: str) -> str:
    """Strip every credential shape, and every long opaque run, from context.

    Context is what makes adjudication possible -- it distinguishes a real key
    from a detector regex or a redacted placeholder -- but raw context is
    exactly where a value leaks. Scrubbing is unconditional and runs in two
    stages: named shapes first (so the label survives and stays informative),
    then anything else that merely looks like secret material.
    """
    for name, rx in COMPILED.items():
        text = rx.sub(f"<{name}>", text)
    text = MARKER_ONLY.sub("<pem_marker>", text)
    text = OPAQUE_RUN.sub(
        lambda m: m.group(0) if IDENTIFIER.match(m.group(0))
        else f"<opaque:{len(m.group(0))}>", text)
    return re.sub(r"\s+", " ", text)[: CONTEXT * 2]


def scan_text(text: str, path: str, lineno: int, out: list) -> None:
    seen_spans = []
    for name, rx in COMPILED.items():
        for m in rx.finditer(text):
            value = normalise(name, m.group(0))
            if len(value) < 12:
                continue
            # An armoured but elided body is a redaction, not an exposure.
            # Recorded under its own name so the row still proves the scanner
            # looked here, and so the two classes can never be conflated.
            if name == "private_key_block" and len(pem_body(m.group(0))) < MIN_PEM_BODY:
                name = "pem_elided"
            seen_spans.append(m.span())
            lo = max(0, m.start() - CONTEXT)
            hi = min(len(text), m.end() + CONTEXT)
            out.append({
                "pattern": name,
                "file": path,
                "line": lineno,
                "offset": m.start(),
                "fp": fingerprint(value),
                "length": len(value),
                "dummy": m.group(0) in KNOWN_DUMMY,
                "context": sanitise(text[lo:m.start()] + " ~~ " + text[m.end():hi]),
            })
    # Unclosed PEM markers, only where no full block already covered them.
    for m in MARKER_ONLY.finditer(text):
        if any(s <= m.start() < e for s, e in seen_spans):
            continue
        lo = max(0, m.start() - CONTEXT)
        hi = min(len(text), m.end() + CONTEXT)
        out.append({
            "pattern": "pem_marker_only",
            "file": path,
            "line": lineno,
            "offset": m.start(),
            "fp": "",
            "length": 0,
            "context": sanitise(text[lo:m.start()] + " ~~ " + text[m.end():hi]),
        })


def record_timestamp(line: str) -> str:
    """The `timestamp` on the JSONL record that carried the match.

    Not the file's mtime. Transcripts are append-only and long-lived: a file
    last written today can carry a credential exposed six weeks ago, so mtime
    systematically reports exposures as more recent than they are, and would
    make a closed leak look active. Parsed only for lines that already matched,
    since json.loads on every line of 11 GB is not affordable.
    """
    try:
        return json.loads(line).get("timestamp", "") or ""
    except Exception:
        return ""


def scan_file(path: str) -> list:
    out = []
    try:
        with open(path, "r", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                before = len(out)
                scan_text(line, path, i, out)
                if len(out) > before:
                    ts = record_timestamp(line)
                    for r in out[before:]:
                        r["ts"] = ts
    except (OSError, MemoryError) as exc:
        out.append({"pattern": "_ERROR", "file": path, "line": 0, "offset": 0,
                    "fp": "", "length": 0, "context": type(exc).__name__})
    return out


def main() -> int:
    paths = [p.rstrip("\n") for p in sys.stdin if p.strip()]
    # A gate whose corpus is empty passes vacuously and looks identical to a
    # genuine all-clear. Refuse rather than report zero findings.
    if not paths:
        print("FATAL: zero input files", file=sys.stderr)
        return 2
    print(f"input_files={len(paths)}", file=sys.stderr)

    rows = []
    for n, p in enumerate(paths, 1):
        rows.extend(scan_file(p))
        if n % 200 == 0:
            print(f"  scanned {n}/{len(paths)} files, {len(rows)} hits",
                  file=sys.stderr)

    out_path = sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout"
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"files={len(paths)} hits={len(rows)} -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
