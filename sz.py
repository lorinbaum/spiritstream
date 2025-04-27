from pathlib import Path
if __name__ == "__main__":

    COLUMN_WIDTH = 35

    paths = [Path("editor.py")] + [p for p in Path("./spiritstream").rglob("*") if p.is_file() and p.suffix == ".py"]
    counts = [0] * len(paths)
    for i, file in enumerate(paths):
        with open(file, "r") as f:
            t = f.readlines()
        docstring = False
        for line in [l.strip() for l in t]:
            if line.startswith("\"\"\""):
                if not line.endswith("\"\"\""): docstring = True 
                continue
            if docstring and line.endswith("\"\"\""):
                docstring = False
                continue
            if line.startswith("#") or line == "" or docstring: continue
            counts[i] += 1

    for c, f in zip(counts, paths): print(f"{f.as_posix():{COLUMN_WIDTH}}:{c:6}")
    print("-" * (COLUMN_WIDTH + 7))
    print(f"{'Total':{COLUMN_WIDTH}}:{sum(counts):6}")
        
