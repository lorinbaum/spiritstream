import unittest, os, math, shutil, time
from spiritstream.log import Log
from pathlib import Path

class TestLog(unittest.TestCase):
    def setUp(self):
        self.log = None
        self.test_folder = Path(__file__).parent / "test_log"
        os.makedirs(self.test_folder / "home", exist_ok=True)
        os.makedirs(self.test_folder / "log", exist_ok=True)
    
    def test_sync(self):
        text = "Hamedi is coming for you!"
        log_text = b"+" + int.to_bytes(1745745417, 4, 'big') + b"+" + f"# author:lorinbaum process:None comment:hamedi #*file.txt,\x00*+\x02+{text}".encode("utf-8")
        with open(self.test_folder / "home/file.txt", "w") as f: f.write(text)
        with open(self.test_folder / "log/temp.log", "w") as f: f.write("")
        with open(self.test_folder / "log/omg.log", "wb") as f: f.write(log_text)
        self.log = Log(self.test_folder / "log/omg.log")
        with open(self.test_folder / "home/file.txt", "rb") as f0:
            with open(self.test_folder / "log/verify/file.txt", "rb") as f1:
                self.assertEqual(f0.read(), f1.read())
        self.assertEqual(self.log.meta, {"author": "lorinbaum", "process": "None", "comment":"hamedi"})
        self.assertEqual(self.log.latest_timestamp, 1745745417+2)
    
    def test_reconstruction(self):
        text = "Hamedi is coming for you!"
        log_text = b"+" + int.to_bytes(1745745417, 4, 'big') + b"+" + f"# author:lorinbaum process:None comment:hamedi #*file.txt,\x00*+\x02+{text[:5]}".encode("utf-8")
        with open(self.test_folder / "home/file.txt", "w") as f: f.write(text)
        with open(self.test_folder / "log/temp.log", "w") as f: f.write("")
        with open(self.test_folder / "log/omg.log", "wb") as f: f.write(log_text)
        self.log = Log(self.test_folder / "log/omg.log")
        with open(self.test_folder / "home/file.txt", "rb") as f0:
            with open(self.test_folder / "log/verify/file.txt", "rb") as f1:
                self.assertNotEqual(f0.read(), f1.read())
        timestamp = math.floor(time.time()) # time of modification
        
        self.log = Log(self.test_folder / "log/omg.log")
        with open(self.test_folder / "home/file.txt", "rb") as f0:
            with open(self.test_folder / "log/verify/file.txt", "rb") as f1:
                self.assertEqual(f0.read(), f1.read())
        self.assertEqual(self.log.meta, {"author": "lorinbaum", "process": "sync", "comment":"hamedi"})
        self.assertEqual(self.log.latest_timestamp, timestamp)

    # def test_deleted_files(self):
    #     text = "Hamedi is coming for you!"
    #     log_text = b"+" + int.to_bytes(1745745417, 4, 'big') + b"+" + f"# author:lorinbaum process:None comment:hamedi #*file.txt,\x00*+\x02+{text}".encode("utf-8")
    #     with open(self.test_folder / "home/file.txt", "w") as f: f.write("")
    #     with open(self.test_folder / "log/temp.log", "w") as f: f.write("")
    #     with open(self.test_folder / "log/omg.log", "wb") as f: f.write(log_text)
    #     self.log = Log(self.test_folder / "log/omg.log") # updates omg.log
    #     self.log = Log(self.test_folder / "log/omg.log") # recalculates state based on updated omg.log
    #     with open(self.test_folder / "home/file.txt", "rb") as f0:
    #         with open(self.test_folder / "log/verify/file.txt", "rb") as f1:
    #             self.assertEqual(f0.read(), f1.read())
    #     self.assertEqual(self.log.meta, {"author": "lorinbaum", "process": "sync", "comment":"hamedi"})
    #     self.assertEqual(self.log.latest_timestamp, math.floor(time.time()))
    
    def tearDown(self):
        del self.log
        shutil.rmtree(self.test_folder)

if __name__ == "__main__": unittest.main()