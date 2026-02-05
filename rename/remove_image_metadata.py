# -*- coding: utf-8 -*-
"""
去除图片元数据：标记、标题、主题、备注、作者
保留：EXIF（相机、曝光、GPS等）
"""
from pathlib import Path
import sys
from typing import Tuple, Optional

try:
    import piexif
except ImportError:
    print("请先安装 piexif: pip install piexif")
    sys.exit(1)

# 要删除的标签（Windows XP / 文档元数据）
# 0x9C9A XPKeywords   - 标记/关键词
# 0x9C9B XPComment    - 备注
# 0x9C9C XPAuthor     - 作者
# 0x9C9D XPSubject    - 主题
# 0x9C9E XPTitle      - 标题
TAGS_TO_REMOVE = {0x9C9A, 0x9C9B, 0x9C9C, 0x9C9D, 0x9C9E}

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}


def has_exif(exif_dict) -> bool:
    """检查是否有有效的 EXIF 数据"""
    if exif_dict is None:
        return False
    for ifd_name in ("0th", "Exif", "GPS", "1st"):
        ifd = exif_dict.get(ifd_name)
        if ifd and len(ifd) > 0:
            return True
    return False


def process_image(path: Path, dry_run: bool = False) -> Tuple[bool, Optional[Path]]:
    """处理单张图片，移除指定元数据，保留其他 EXIF。若有 EXIF 则在文件名后加 _exif。返回 (是否处理, 最终路径)"""
    try:
        exif_dict = piexif.load(str(path))
    except Exception as e:
        print(f"  跳过（无法读取）: {e}")
        return False, None

    if not has_exif(exif_dict):
        return False, None  # 无 EXIF，跳过

    modified = False
    for ifd_name in ("0th", "Exif", "GPS", "1st"):
        ifd = exif_dict.get(ifd_name)
        if not ifd:
            continue
        for tag in list(ifd.keys()):
            if tag in TAGS_TO_REMOVE:
                del ifd[tag]
                modified = True

    current_path = path
    if not dry_run:
        if modified:
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, str(path))
        # 若有 EXIF 且文件名不含 _exif，则重命名
        stem, suffix = path.stem, path.suffix
        if not stem.endswith("_exif"):
            new_path = path.parent / f"{stem}_exif{suffix}"
            if new_path != path and not new_path.exists():
                path.rename(new_path)
                current_path = new_path

    return True, current_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="去除图片的标记、标题、主题、备注、作者，保留 EXIF")
    parser.add_argument("path", nargs="?", default=".", help="图片路径或目录")
    parser.add_argument("-n", "--dry-run", action="store_true", help="仅预览，不实际修改")
    parser.add_argument("-r", "--recursive", action="store_true", help="递归处理子目录")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"路径不存在: {root}")
        sys.exit(1)

    files = []
    if root.is_file():
        files = [root]
    else:
        pattern = "**/*" if args.recursive else "*"
        files = [f for f in root.glob(pattern) if f.is_file() and f.suffix.lower() in SUPPORTED_EXT]

    print(f"找到 {len(files)} 个图片文件" + (" (预览模式)" if args.dry_run else ""))
    processed_count = 0
    for f in files:
        ok, final_path = process_image(f, args.dry_run)
        if ok:
            processed_count += 1
            display = final_path or f
            try:
                rel = display.relative_to(root) if root.is_dir() else display.name
            except ValueError:
                rel = display.name
            print(f"  处理: {rel}")

    print(f"完成，共处理 {processed_count} 个文件")


if __name__ == "__main__":
    main()
