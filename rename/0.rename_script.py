import os

target_string = "[BT-btt.com]"

root_dir = os.getcwd()
print(f"Scanning directory: {root_dir}")

for root, dirs, files in os.walk(root_dir, topdown=False):
    # Rename files
    for filename in files:
        if target_string in filename:
            new_filename = filename.replace(target_string, "").strip()
            if new_filename != filename and new_filename != "":
                old_path = os.path.join(root, filename)
                new_path = os.path.join(root, new_filename)
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed file: {filename} -> {new_filename}")
                except Exception as e:
                    print(f"Error renaming file {filename}: {e}")

    # Rename directories
    for dirname in dirs:
        if target_string in dirname:
            new_dirname = dirname.replace(target_string, "").strip()
            if new_dirname != dirname and new_dirname != "":
                old_path = os.path.join(root, dirname)
                new_path = os.path.join(root, new_dirname)
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed directory: {dirname} -> {new_dirname}")
                except Exception as e:
                    print(f"Error renaming directory {dirname}: {e}")
