"""R2 sync via rclone for shared paper storage."""

import logging
import subprocess

logger = logging.getLogger(__name__)

RCLONE_REMOTE = "r2:paper-intelligence"
LOCAL_DIR = "~/Documents/papers"


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
    return _run_rclone([
        "copy", RCLONE_REMOTE, LOCAL_DIR,
        "--transfers", "8",
        "--checkers", "16",
    ])


def push():
    """Push local papers to R2 (additive only — won't delete remote files)."""
    return _run_rclone([
        "copy", LOCAL_DIR, RCLONE_REMOTE,
        "--transfers", "8",
        "--checkers", "16",
    ])


def sync_bidirectional():
    """Bidirectional sync — both sides get all files, nothing deleted."""
    pull()
    push()
