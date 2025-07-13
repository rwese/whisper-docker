#!/usr/bin/env python3
"""
Minimal GitHub pipeline status checker for AI agent contexts
Designed to produce concise output suitable for AI agents
"""

import subprocess
import json
import sys
from datetime import datetime


def run_gh_command(cmd):
    """Run gh command and return JSON result"""
    try:
        result = subprocess.run(
            f"gh {cmd}", 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout) if result.stdout.strip() else None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, subprocess.SubprocessError):
        return None


def format_time_ago(iso_time):
    """Format time ago in compact format"""
    try:
        dt = datetime.fromisoformat(iso_time.replace('Z', '+00:00'))
        diff = datetime.now(dt.tzinfo) - dt
        
        if diff.days > 0:
            return f"{diff.days}d"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600}h"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60}m"
        else:
            return f"{diff.seconds}s"
    except:
        return "?"


def get_status_icon(status, conclusion):
    """Get compact status icon"""
    if status == "in_progress":
        return "🟡"
    elif conclusion == "success":
        return "✅"
    elif conclusion == "failure":
        return "❌"
    elif conclusion == "cancelled":
        return "⚠️"
    else:
        return "⚪"


def main():
    # Get latest workflow runs (minimal data)
    runs = run_gh_command("run list --limit 3 --json status,conclusion,workflowName,createdAt,number")
    
    if not runs:
        print("❌ Unable to fetch pipeline status")
        sys.exit(1)
    
    print("🔄 GitHub Pipeline Status:")
    
    for run in runs:
        icon = get_status_icon(run.get('status'), run.get('conclusion'))
        workflow = run.get('workflowName', 'Unknown')[:20]  # Truncate long names
        number = run.get('number', '?')
        time_ago = format_time_ago(run.get('createdAt', ''))
        status = run.get('status', 'unknown')
        
        # Compact one-line format
        print(f"{icon} #{number} {workflow} ({status}) {time_ago} ago")
    
    # Check if any runs are failing
    latest_run = runs[0] if runs else {}
    if latest_run.get('conclusion') == 'failure':
        print(f"\n⚠️  Latest run #{latest_run.get('number')} failed")
        return 1
    elif latest_run.get('status') == 'in_progress':
        print(f"\n⏳ Run #{latest_run.get('number')} in progress")
        return 0
    elif latest_run.get('conclusion') == 'success':
        print(f"\n✅ All recent runs successful")
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())