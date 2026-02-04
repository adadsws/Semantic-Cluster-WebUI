import os
import shutil
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import TAGS

# Define the destination folder
dest_root_name = "~no_camera_info"
dest_root = os.path.join(os.getcwd(), dest_root_name)

# Ensure destination exists
if not os.path.exists(dest_root):
    os.makedirs(dest_root)

# Supported image extensions (files PIL can likely open)
extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.webp'}

# Tags indicating camera info usually include Make or Model
CAMERA_TAGS = {'Make', 'Model'}

def has_camera_info(file_path):
    try:
        with Image.open(file_path) as img:
            exif_data = img._getexif()
            if not exif_data:
                return False
            
            # Check if any camera-specific tags exist
            found_camera_info = False
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name in CAMERA_TAGS:
                    return True
            
            return False
    except (UnidentifiedImageError, AttributeError, SyntaxError, OSError):
        # If we can't open it or it's not an image with exif support capabilities easily read, treat as no info or skip?
        # User said "Pictures without camera info". If it's not a picture or corrupt, maybe skip.
        # But if it's a valid PNG (which often has no EXIF), it counts as "no camera info".
        # Let's assume if it is an image file extension we tracked, and we can't find EXIF, it has no camera info.
        return False
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

root_dir = os.getcwd()
print(f"Scanning directory: {root_dir}")

moved_count = 0

for root, dirs, files in os.walk(root_dir):
    # Skip the destination folder itself to avoid recursion if it's inside root
    if dest_root_name in os.path.abspath(root):
        continue

    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in extensions:
            file_path = os.path.join(root, filename)
            
            # Check for camera info
            if not has_camera_info(file_path):
                # Calculate destination path maintaining structure
                # rel_path includes the directory structure relative to root
                rel_dir = os.path.relpath(root, root_dir)
                
                if rel_dir == ".":
                     dest_dir = dest_root
                else:
                    dest_dir = os.path.join(dest_root, rel_dir)
                
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                
                new_path = os.path.join(dest_dir, filename)
                
                try:
                    shutil.move(file_path, new_path)
                    print(f"Moved (No Info): {filename} -> {new_path}")
                    moved_count += 1
                except Exception as e:
                    print(f"Error moving {filename}: {e}")

print(f"Total images moved: {moved_count}")

# Cleanup empty directories
print("Cleaning up empty directories...")
for root, dirs, files in os.walk(root_dir, topdown=False):
    # Skip destination folder
    if dest_root_name in os.path.abspath(root):
        continue
    
    # Don't delete the root folder itself
    if os.path.abspath(root) == os.path.abspath(root_dir):
        continue

    # If directory is empty, delete it
    # We check os.listdir to be sure it's actually empty at this moment
    try:
        if not os.listdir(root):
            os.rmdir(root)
            print(f"Deleted empty directory: {root}")
    except Exception as e:
        print(f"Error deleting {root}: {e}")
