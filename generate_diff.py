import subprocess
import sys

def run_git_command(args):
    result = subprocess.run(['git'] + args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running git {' '.join(args)}: {result.stderr}")
        sys.exit(1)
    return result.stdout.strip()

def main():
    base_branch = "staged-aig-release"  # Change to "master" if your default branch is master
    target_branch = "feat/macro-frontend"
    
    print(f"Analyzing changes between '{base_branch}' and '{target_branch}'...")

    # Fetch latest branch info if needed, or rely on local refs
    stat_summary = run_git_command(['diff', '--stat', f'{base_branch}..{target_branch}'])
    name_status = run_git_command(['diff', '--name-status', f'{base_branch}..{target_branch}'])
    full_diff = run_git_command(['diff', f'{base_branch}..{target_branch}'])

    report_lines = [
        "=" * 80,
        "                      REPOSITORY CHANGES & DIFF REPORT",
        "=" * 80,
        f"Base Branch:   {base_branch}",
        f"Target Branch: {target_branch}",
        "",
        "--- 1. OVERALL STATS SUMMARY ---",
        stat_summary,
        "",
        "--- 2. FILE STATUS (Added [A], Modified [M], Deleted [D]) ---",
        name_status,
        "",
        "--- 3. DETAILED FILE-BY-FILE DIFFS ---",
        full_diff,
        "",
        "=" * 80,
        "End of Report",
        "=" * 80
    ]

    output_filename = "changes_report.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Report successfully generated and saved to: {output_filename}")

if __name__ == "__main__":
    main()
