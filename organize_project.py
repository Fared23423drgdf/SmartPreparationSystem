import os
import shutil
from pathlib import Path

def organize_project():
    """
    Organizes the Smart Attendance System project files into a structured directory.

    This script performs the following actions:
    1. Creates necessary subdirectories if they don't already exist.
    2. Moves specified files and directories to their new locations.
    3. Handles existing files/folders by overwriting or skipping, ensuring idempotency.
    4. Prints progress messages for each operation.
    """

    print("Starting project organization...")

    # Define the base directory (where this script is run from)
    base_dir = Path.cwd()

    # Define the new project structure
    structure = {
        "backend": {
            "files": {"main.py": "app.py", "Face_recognition_demo.py": "Face_recognition_demo.py"},
            "dirs": {}
        },
        "frontend": {
            "public": {
                "files": {"attendance_dashboard.html": "attendance_dashboard.html", "Face_recognition_demo.html": "Face_recognition_demo.html", "Portfolio.html": "Portfolio.html"},
                "dirs": {}
            }
        },
        "data": {
            "known_faces": {"files": {}, "dirs": {}},
            "unknown_faces_detected": {"files": {}, "dirs": {}},
            "snapshots": {"files": {}, "dirs": {}},
            "attendance_logs": {
                "files": {"attendance_log.json": "attendance_log.json", "log.json": "log.json"},
                "dirs": {}
            }
        },
        "docs": {
            "files": {"README.md": "README.md"},
            "dirs": {}
        }
    }

    # Files that will remain in the root or are created by this script
    root_files = ["requirements.txt", "organize_project.py"]

    # --- Step 1: Create required folders ---
    print("\n1. Creating necessary directories...")
    for top_level_dir, content in structure.items():
        current_path = base_dir / top_level_dir
        if not current_path.exists():
            current_path.mkdir(parents=True)
            print(f"  - Created directory: {current_path}")
        else:
            print(f"  - Directory already exists: {current_path} (Skipping creation)")

        # Handle nested directories
        for sub_dir_name, sub_content in content.items():
            if isinstance(sub_content, dict) and "files" in sub_content: # It's a directory entry
                nested_path = current_path / sub_dir_name
                if not nested_path.exists():
                    nested_path.mkdir(parents=True)
                    print(f"  - Created nested directory: {nested_path}")
                else:
                    print(f"  - Nested directory already exists: {nested_path} (Skipping creation)")
            elif isinstance(sub_content, dict) and "public" in sub_content: # Special case for frontend/public
                public_path = current_path / "public"
                if not public_path.exists():
                    public_path.mkdir(parents=True)
                    print(f"  - Created nested directory: {public_path}")
                else:
                    print(f"  - Nested directory already exists: {public_path} (Skipping creation)")


    # --- Step 2: Move files and folders ---
    print("\n2. Moving files and directories...")

    # Helper function to move files
    def move_file(src_name, dest_path, new_name=None):
        src = base_dir / src_name
        dest_filename = new_name if new_name else src_name
        dest = dest_path / dest_filename

        if src.exists():
            if dest.exists():
                if src.is_file() and dest.is_file():
                    # Check if it's the same file (already moved)
                    if os.path.samefile(src, dest):
                        print(f"  - File already at target: {src_name} -> {dest} (Skipping)")
                        return
                elif src.is_dir() and dest.is_dir():
                    print(f"  - Directory already at target: {src_name} -> {dest} (Skipping)")
                    return
                # If target exists but is different, remove it to overwrite
                print(f"  - Target exists, overwriting: {dest}")
                if dest.is_file():
                    os.remove(dest)
                elif dest.is_dir():
                    shutil.rmtree(dest)
            
            try:
                if src.is_file():
                    shutil.move(str(src), str(dest))
                    print(f"  - Moved file: {src_name} -> {dest}")
                elif src.is_dir():
                    shutil.move(str(src), str(dest))
                    print(f"  - Moved directory: {src_name} -> {dest}")
            except Exception as e:
                print(f"  - Error moving {src_name} to {dest}: {e}")
        else:
            print(f"  - Source not found: {src_name} (Skipping)")

    # Move files and directories according to the structure
    # Backend files
    move_file("main.py", base_dir / "backend", "app.py")
    move_file("Face_recognition_demo.py", base_dir / "backend")

    # Frontend files
    move_file("attendance_dashboard.html", base_dir / "frontend" / "public")
    move_file("Face_recognition_demo.html", base_dir / "frontend" / "public")
    move_file("Portfolio.html", base_dir / "frontend" / "public")

    # Data directories and files
    move_file("known_faces", base_dir / "data")
    move_file("unknown_faces_detected", base_dir / "data")
    move_file("snapshots", base_dir / "data")
    move_file("attendance_log.json", base_dir / "data" / "attendance_logs")
    move_file("log.json", base_dir / "data" / "attendance_logs") # Move log.json as well

    # Docs files
    move_file("README.md", base_dir / "docs")

    # Create dummy requirements.txt if it doesn't exist
    requirements_path = base_dir / "requirements.txt"
    if not requirements_path.exists():
        with open(requirements_path, "w") as f:
            f.write("# Project dependencies\n")
            f.write("opencv-python\n")
            f.write("face_recognition\n")
            f.write("numpy\n")
            f.write("Flask\n")
            f.write("Flask-SocketIO\n")
            f.write("python-socketio[client]\n") # For client if needed, though usually frontend handles it
            f.write("Werkzeug\n") # Dependency for Flask
        print(f"  - Created dummy requirements.txt at: {requirements_path}")
    else:
        print(f"  - requirements.txt already exists: {requirements_path} (Skipping creation)")

    print("\nProject organization complete!")

if __name__ == "__main__":
    organize_project()
