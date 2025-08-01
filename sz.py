if __name__ == "__main__":
    from pathlib import Path
    
    COLUMN_WIDTH = 35

    sz = dict()

    # python scripts
    paths = [Path("editor.py")] + [p for p in Path("./spiritstream").rglob("*") if p.is_file() and p.suffix == ".py"]
    for file in paths:
        c = 0
        with open(file, "r") as f: t = f.readlines()
        docstring = False
        for line in [l.strip() for l in t]:
            if line.startswith("\"\"\""):
                if not line.endswith("\"\"\""): docstring = True 
                continue
            if docstring and line.endswith("\"\"\""):
                docstring = False
                continue
            if line.startswith("#") or line == "" or docstring: continue
            c += 1
        sz[file.as_posix()] = c

    # shaders
    paths = [p for p in Path("./spiritstream").rglob("*") if p.is_file() and p.suffix in [".vert", ".frag"]]
    for file in paths:
        c = 0
        with open(file, "r") as f: t = f.readlines()
        for line in [l.strip() for l in t]:
            if line.startswith("//") or line == "": continue
            else: c += 1
        sz[file.as_posix()] = c

    for f, c in sz.items(): print(f"{f:{COLUMN_WIDTH}}:{c:6}")
    print("-" * (COLUMN_WIDTH + 7))
    print(f"{'Total':{COLUMN_WIDTH}}:{sum(sz.values()):6}")
        
