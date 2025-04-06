if __name__ == "__main__":
    files = ["table.py", "ttf.py", "vec.py", "dtype.py", "op.py"]
    
    counts = [0] * len(files)
    for i, file in enumerate(files):
        with open(file, "r") as f:
            t = f.readlines()
        docstring = False
        for line in [l.strip() for l in t]:
            if line.startswith("\"\"\""):
                docstring = True
                continue
            if docstring and line.endswith("\"\"\""):
                docstring = False
                continue
            if line.startswith("#") or line == "": continue
            counts[i] += 1

    for c, f in zip(counts, files): print(f"{f:10}:{c:6}")
    print("-----------------")
    print(f"Total     :{sum(counts):6}")
        
