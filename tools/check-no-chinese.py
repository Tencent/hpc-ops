import sys
import re
import os

chinese_pattern = re.compile(r"[\u4e00-\u9fff]")


def check_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        if chinese_pattern.search(f.read()):
            print(f"Please don't use Chinese character in code：{path}")
            return True
    return False


def main():
    extensions = {".py", ".cc", ".cu", ".h", "cuh"}
    found = False
    for root, _, files in os.walk("."):
        for f in files:
            if any(f.endswith(ext) for ext in extensions):
                if check_file(os.path.join(root, f)):
                    found = True
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
