import time, math, os
from pathlib import Path
from typing import Dict
from difflib import ndiff

class Log:
    def __init__(self, filepath):
        """Initialize log with the given file path."""
        self.filepath = Path(filepath)
        self.f = open(self.filepath, "rb")
        self.temp = open(self.filepath.parent / "temp.log", "rb")
        self.file = None # file used in the most recent logged event
        self.meta = {"author": None}
        self.cursor = None
        self.latest_timestamp = None

        """Ensure and restore log and file integrity."""
        os.makedirs(self.filepath.parent / "verify", exist_ok=True)
        self.test_meta = {}
        self.test_timestamp = 0
        self.test_file = None
        self.test_file_path = Path(self.filepath.parent / "verify" / "test.file") # this file doesn't exist, just to resolve relative paths later.
        self.escaped = False
        self.current_block = {
            "type": None,
            "start": None,
            "end": None
        }
        if (temp:=self.temp.read()) == b"":
            escaped = False
            self.text = self.f.read()
            if self.text != "":
                for i, char in enumerate(self.text):
                    if (char:=chr(char)) == "\\" and escaped == False: escaped = True
                    else:
                        if escaped == True: escaped = False
                        else:
                            if char == "#":
                                if self.current_block["type"] != "meta": self.new_block("meta", i)
                                else: self.current_block["end"] = i
                            elif char == "+":
                                if self.current_block["type"] != "elapsed": self.new_block("elapsed", i)
                                else: self.current_block["end"] = i
                            elif char == "*":
                                if self.current_block["type"] != "seek": self.new_block("seek", i)
                                else: self.current_block["end"] = i
                            elif char == "-":
                                if self.current_block["type"] != "erase": self.new_block("erase", i)
                                else: self.current_block["end"] = i
                            elif self.current_block["end"] != None: self.new_block("write", i)
                # assign end to and process last block
                self.current_block["end"] = i
                self.process_block() 
        else: raise NotImplementedError
        
        self.f.close()
        self.f = open(self.filepath, "ab")
        self.temp.close()
        self.temp = open(self.filepath.parent / "temp.log", "wb+")
        self.meta = self.test_meta
        self.cursor = self.test_file.tell()
        self.latest_timestamp = self.test_timestamp
        self.test_file.flush()

        test_path = Path(self.filepath.parent / "verify")
        real_path = Path(self.filepath.parent.parent / "home")
        test_files = [p.relative_to(test_path) for p in test_path.rglob("*") if p.is_file()]
        real_files = [p.relative_to(real_path) for p in real_path.rglob("*") if p.is_file()]
        test_files.sort()
        real_files.sort()
        for real_file_path in real_files:
            test_file_path = [p for p in test_files if p == real_file_path]
            real_file_path = real_path / real_file_path
            with open(real_file_path, "rb") as real_file:
                timestamp = math.floor(os.stat(real_file_path).st_mtime)
                if test_file_path == []: # file is new
                    self.write(real_file.read(), real_file_path, 0, {"process":"sync"}, timestamp)
                else:
                    assert len(test_file_path) == 1
                    test_file_path = test_path / test_file_path[0]
                    with open(test_file_path, "rb") as test_file:
                        cursor = 0
                        if (t_test:=test_file.read()) != (t_real:=real_file.read()):
                            diff = ndiff([chr(c) for c in t_test], [chr(c) for c in t_real])
                            for line in diff:
                                if line.startswith((" ")): cursor += 1
                                if line.startswith("-"):
                                    self.erase(1, real_file_path.relative_to(real_path).as_posix(), cursor+1, {"process":"sync"}, timestamp)
                                if line.startswith("+"):
                                    self.write(int.to_bytes(ord(line[2:]), 1, "big"), real_file_path.relative_to(real_path).as_posix(), cursor, {"process":"sync"}, timestamp)
                                    cursor += 1
        for test_file_path in [p for p in test_files if p not in real_files]: # check for deleted files
            self.erase((fsize:=os.stat(test_file_path).st_size), test_file_path, fsize, {"process":"sync"}) # timestamp will be now because who knows when this file was deleted
            # TODO: file will still exit, but will have no characters in it
        self.commit() # changes written to temp

    def write(self, text:bytes, file=None, cursor=None, meta:dict=None, timestamp:int=None):
        """Insert text at cursor."""
        self._elapsed(timestamp)
        self._seek(file, cursor)
        self._meta(meta)
        self.temp.write(text)
        self.cursor += len(text)

    def erase(self, count=1, file=None, cursor=None, meta=None, timestamp:int=None):
        """Remove count characters at cursor."""
        self._elapsed(timestamp)
        self._seek(file, cursor)
        self._meta(meta)
        self.temp.write(b"-" if count == 1 else b"-" + int.to_bytes(count, math.ceil(math.log2(count+1)/8), "big") + b'-')
        self.cursor -= count

    def commit(self):
        """Save changes to the log."""
        self.temp.flush()
        self.temp.seek(0)
        if (t:=self.temp.read()) != b"":
            self.f.write(t)
            self.f.flush()
            self.temp.seek(0)
            self.temp.truncate()
            self.temp.flush()
    
    def view(self, date):
        """Return the state as of the given date, without altering current state."""

    # internal methods
    def _seek(self, file:str, cursor:int):
        """Record a seek event if cursor position changes."""
        if file not in [None, self.file]:
            assert isinstance(file, str)
            assert Path(file).is_absolute() == False
            assert cursor >= 0 # if file changes, cursor must be provided too
            self.temp.write(b"*" + file.encode("utf-8") + b"," + int.to_bytes(cursor, math.ceil(math.log2(cursor+1)/8), "big") + b"*")
            self.file, self.cursor = file, cursor
        elif cursor not in [None, self.cursor]:
            assert cursor >= 0
            self.temp.write(b"*" + int.to_bytes(cursor, math.ceil(math.log2(cursor+1)/8), "big") + b"*")
            self.cursor = cursor
        assert self.cursor >= 0

    def _meta(self, meta:Dict[str, str]):
        """Record a meta event storing keys and values like "author":"alice", "process":"sync" or "comment":"hamedi is coming for you"."""
        if meta != None:
            event = ""
            assert isinstance(meta, dict)
            for k,v in meta.items():
                if self.meta[k] != v:
                    if isinstance(v, str): assert " " not in v
                    self.meta[k] = v
                    event += f"{k}:{v} "
            if event != "": self.temp.write(b"# " + event.encode() + b"#")
            assert self.meta["author"] != None

    def _elapsed(self, timestamp):
        """Record elapsed time since last timestamp in whole seconds if at least 1 second."""
        if timestamp == None: timestamp = math.floor(time.time())
        assert isinstance(timestamp, int) and timestamp >= self.latest_timestamp, (type(timestamp), timestamp, self.latest_timestamp)
        elapsed = timestamp - self.latest_timestamp
        if elapsed > 0:
            self.latest_timestamp = timestamp
            self.temp.write(b"+" if elapsed == 1 else b"+"+ int.to_bytes(elapsed, math.ceil(math.log2(elapsed+1)/8), "big") + b"+")
    
    def new_block(self, block_type, start):
        if block_type != "write": self.current_block["end"] = start-1
        if self.current_block["type"] != None: self.process_block()
        self.current_block = {
            "type": block_type,
            "start": start,
            "end": None
        }

    def process_block(self):
        t = (t:= self.text[self.current_block["start"]:self.current_block["end"] + 1])
        if self.current_block["type"] == "meta": self.test_meta.update({(a:=item.decode("utf-8").split(":"))[0]:a[1] for item in t.strip(b"# ").split(b" ")})
        elif self.current_block["type"] == "elapsed": self.test_timestamp += 1 if t == b"+" else int.from_bytes((t.strip(b"+ ")), "big")
        elif self.current_block["type"] == "seek":
            file_path, cursor = t.split(b",") if b"," in (t:=t.strip(b"* ")) else (b"", t)
            cursor = int.from_bytes(cursor, "big")
            if file_path != b"" and (new_path:=Path(self.test_file_path).parent / file_path.decode("utf-8")) != self.test_file_path:
                if self.test_file != None: self.test_file.close()
                self.test_file = open(new_path, "rb+" if os.path.exists(new_path) else "wb+")
                self.test_file_path = new_path
            self.test_file.seek(cursor)
        elif self.current_block["type"] == "erase":
            self.test_file.seek(self.test_file.tell() - (1 if t == b"-" else int.from_bytes((t.strip(b"- ")), "big")))
            self.test_file.truncate()
        elif self.current_block["type"] == "write":
            t = t.replace(b"\\#", b"#").replace(b"\\+", b"+").replace(b"\\*", b"*").replace(b"\\-", b"-")
            self.test_file.write(t)
        else: raise ValueError(self.current_block)
    
    def __del__(self):
        if not self.f.closed: self.f.close()
        if not self.temp.closed: self.temp.close()
        if not self.test_file.closed: self.test_file.close()
        open(self.filepath.parent / "temp.log", "w").close()  # empty temp before closing