# SPDX-FileCopyrightText: 2025-2026 Domyn
# SPDX-License-Identifier: Apache-2.0


def assert_cli_ok(result, msg="CLI command failed unexpectedly"):
    """Assert a CLI result was successful."""
    assert result.exit_code == 0, f"{msg}\n\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


def assert_cli_error(result, expected_code=1, msg="CLI command should have failed"):
    """Assert a CLI command failed as expected."""
    assert result.exit_code == expected_code, (
        f"{msg} (got exit code {result.exit_code})\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def print_cli_debug(result):
    print("🧪 STDOUT:\n", result.stdout)
    print("🧪 STDERR:\n", result.stderr)
    if result.exception:
        print("🧪 Exception:\n", result.exception)


def extract_rich_text(output: str) -> str:
    in_box = False
    content_lines = []

    for line in output.splitlines():
        # Start of Rich box
        if line.strip().startswith("╭"):
            in_box = True
            continue
        # End of Rich box
        if line.strip().startswith("╰"):
            in_box = False
            continue
        if in_box and "│" in line:
            # Extract content between vertical bars
            parts = line.split("│")
            if len(parts) >= 2:
                content = parts[1].strip()
                content_lines.append(content)

    return " ".join(content_lines)
