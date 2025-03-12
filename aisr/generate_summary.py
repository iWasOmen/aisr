#!/usr/bin/env python
"""
AISR Project Code Summary Generator

This script creates a comprehensive summary of all Python files in the current directory
and its subdirectories, generating a single text document that can be shared with others.
"""

import os
import datetime
import re


def generate_project_summary(root_dir='.', output_file='aisr_project_summary.txt'):
    """
    Generate a summary of all Python files in the project.

    Args:
        root_dir: Root directory to scan (default is current directory)
        output_file: Output file name
    """
    # Dictionary to store directory structure
    project_structure = {}

    # Get absolute path for root directory
    root_dir = os.path.abspath(root_dir)
    root_dir_name = os.path.basename(root_dir)

    # Walk through directory tree
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip hidden directories and __pycache__
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']

        # Filter for Python files
        py_files = [f for f in filenames if f.endswith('.py') and "generate_summary" not in f]

        if py_files:
            # Calculate relative path from root
            rel_path = os.path.relpath(dirpath, root_dir)
            if rel_path == '.':
                rel_path = root_dir_name
            else:
                rel_path = os.path.join(root_dir_name, rel_path)

            # Initialize directory in structure if not exists
            if rel_path not in project_structure:
                project_structure[rel_path] = []

            # Process each Python file
            for py_file in py_files:
                file_path = os.path.join(dirpath, py_file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Add file info to structure
                    project_structure[rel_path].append({
                        'filename': py_file,
                        'path': file_path,
                        'content': content
                    })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    # Generate the summary text
    with open(output_file, 'w', encoding='utf-8') as out_file:
        # Write header
        out_file.write("=" * 80 + "\n")
        out_file.write(f"AISR PROJECT CODE SUMMARY\n")
        out_file.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out_file.write("=" * 80 + "\n\n")

        # Write directory structure summary
        out_file.write("## PROJECT STRUCTURE\n\n")
        for directory, files in sorted(project_structure.items()):
            out_file.write(f"- {directory}/\n")
            for file_info in sorted(files, key=lambda x: x['filename']):
                out_file.write(f"  - {file_info['filename']}\n")
        out_file.write("\n\n")

        # Write file contents
        out_file.write("## FILE CONTENTS\n\n")
        for directory, files in sorted(project_structure.items()):
            out_file.write(f"### Directory: {directory}/\n\n")

            for file_info in sorted(files, key=lambda x: x['filename']):
                out_file.write(f"#### {file_info['filename']}\n")
                out_file.write("```python\n")
                out_file.write(file_info['content'])
                if not file_info['content'].endswith('\n'):
                    out_file.write('\n')
                out_file.write("```\n\n")

    # Calculate statistics
    total_files = sum(len(files) for files in project_structure.values())
    total_dirs = len(project_structure)
    total_lines = sum(len(file_info['content'].splitlines())
                      for files in project_structure.values()
                      for file_info in files)

    print(f"Summary generated in {output_file}")
    print(f"Processed {total_files} Python files across {total_dirs} directories")
    print(f"Total lines of code: {total_lines}")

    return output_file, total_files, total_dirs, total_lines


if __name__ == "__main__":
    output_file, total_files, total_dirs, total_lines = generate_project_summary()

    print("\nSummary complete! You can now share the generated file in your next conversation.")
    print(f"The file contains {total_lines} lines of code from {total_files} Python files.")