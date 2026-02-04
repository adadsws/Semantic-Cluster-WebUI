import os

# Video extensions to target
extensions = {
    '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', 
    '.webm', '.m4v', '.mpg', '.mpeg', '.3gp', '.ts',
    '.rmvb', '.vob'
}

root_dir = os.getcwd()
print(f"Scanning directory for videos: {root_dir}")

for root, dirs, files in os.walk(root_dir):
    # Calculate relative path from script location
    rel_path = os.path.relpath(root, root_dir)
    
    # Skip root directory files if necessary. 
    # If rel_path is '.', we probably don't want a prefix like ".-filename".
    if rel_path == ".":
        continue

    # Create prefix, replace path separators with '-'
    prefix = rel_path.replace(os.path.sep, "-")
    
    for filename in files:
        # Check extension
        ext = os.path.splitext(filename)[1].lower()
        if ext in extensions:
            # Construct new filename
            new_filename = f"{prefix}-{filename}"
            
            old_path = os.path.join(root, filename)
            new_path = os.path.join(root, new_filename)
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed video: {os.path.join(rel_path, filename)} -> {new_filename}")
            except Exception as e:
                print(f"Error renaming {filename}: {e}")
