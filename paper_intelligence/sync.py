"""R2 sync via rclone for shared paper storage.

Auto-configures rclone on first use from 1Password credentials.
Auto-installs rclone via brew if not found.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

RCLONE_REMOTE = "r2:paper-intelligence"
LOCAL_DIR = "~/Documents/papers"
RCLONE_CONF = Path.home() / ".config" / "rclone" / "rclone.conf"

# 1Password paths for R2 credentials
OP_ITEM = "R2 Paper Intelligence"
OP_VAULT = "CLI Secrets"


def _ensure_rclone_installed() -> bool:
    """Install rclone via brew if not found."""
    if shutil.which("rclone"):
        return True
    logger.info("Installing rclone...")
    try:
        subprocess.run(
            ["brew", "install", "rclone"],
            check=True, capture_output=True, timeout=120,
        )
        return True
    except Exception:
        logger.warning("Could not install rclone — install manually: brew install rclone")
        return False


def _ensure_rclone_configured() -> bool:
    """Configure rclone for R2 from 1Password if not already set up."""
    if RCLONE_CONF.exists() and "r2" in RCLONE_CONF.read_text():
        return True

    logger.info("Configuring rclone for R2 from 1Password...")
    try:
        access_key = subprocess.check_output(
            ["op", "read", f"op://{OP_VAULT}/{OP_ITEM}/Access Key ID"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        secret_key = subprocess.check_output(
            ["op", "read", f"op://{OP_VAULT}/{OP_ITEM}/Secret Access Key"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        endpoint = subprocess.check_output(
            ["op", "read", f"op://{OP_VAULT}/{OP_ITEM}/Endpoint"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        logger.warning("Could not read R2 credentials from 1Password — skipping sync")
        return False

    RCLONE_CONF.parent.mkdir(parents=True, exist_ok=True)

    # Append r2 config (preserve existing config if any)
    existing = RCLONE_CONF.read_text() if RCLONE_CONF.exists() else ""
    if "[r2]" not in existing:
        with open(RCLONE_CONF, "a") as f:
            f.write(f"""
[r2]
type = s3
provider = Cloudflare
access_key_id = {access_key}
secret_access_key = {secret_key}
endpoint = {endpoint}
acl = private
no_check_bucket = true
""")
    return True


def _ensure_setup() -> bool:
    """Ensure rclone is installed and configured. Returns True if ready."""
    return _ensure_rclone_installed() and _ensure_rclone_configured()


def _run_rclone(args: list[str]) -> bool:
    """Run an rclone command. Returns True on success."""
    cmd = ["rclone"] + args
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return True
    except FileNotFoundError:
        logger.warning("rclone not installed — skipping sync")
        return False
    except subprocess.CalledProcessError as e:
        logger.warning(f"rclone failed: {e.stderr.strip()}")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("rclone timed out")
        return False


def pull():
    """Pull latest papers from R2 (additive only — won't delete local files)."""
    if not _ensure_setup():
        return False
    return _run_rclone([
        "copy", RCLONE_REMOTE, LOCAL_DIR,
        "--transfers", "8",
        "--checkers", "16",
    ])


def push():
    """Push local papers to R2 (additive only — won't delete remote files)."""
    if not _ensure_setup():
        return False
    return _run_rclone([
        "copy", LOCAL_DIR, RCLONE_REMOTE,
        "--transfers", "8",
        "--checkers", "16",
    ])
