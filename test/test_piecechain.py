import unittest
from spiritstream.piecechain import PieceChain

def check_piece_chain_integrity(p:PieceChain):
    node = p.head
    prev = None
    visited = set()

    while node:
        assert id(node) not in visited # check for cycles
        visited.add(id(node))

        assert node.prev is prev
        if node.next: assert node.next.prev is node
        if node.prev: assert node.prev.next is node
        assert node.length > 0

        prev = node
        node = node.next

class Test_PieceChain(unittest.TestCase):
    def setUp(self): self.P = PieceChain("example text")
    def tearDown(self): check_piece_chain_integrity(self.P)
    
    def test_insert_start(self):
        self.P.insert(0, "h")
        self.assertEqual(self.P.text, "hexample text")

        self.P.insert(1, "amedi ")
        self.assertEqual(self.P.text, "hamedi example text")

    def test_insert_end(self):
        self.P.insert(len(self.P.text), " h")
        self.assertEqual(self.P.text, "example text h")

        self.P.insert(len(self.P.text), "amedi")
        self.assertEqual(self.P.text, "example text hamedi")

    def test_insert_middle(self):
        self.P.insert(8, "h ")
        self.assertEqual(self.P.text, "example h text")

        self.P.insert(9, "amedi")
        self.assertEqual(self.P.text, "example hamedi text")
        self.assertEqual(self.P.add, "h amedi")

    def test_insert_emtpy(self):
        self.P.insert(0, "")
        self.assertEqual(self.P.text, "example text")

    def test_delete_start(self):
        self.P.delete(0, 1)
        self.assertEqual(self.P.text, "xample text")

        self.P.delete(0, 7)
        self.assertEqual(self.P.text, "text")

    def test_delete_end(self):
        self.P.delete(len(self.P.text) - 1 , len(self.P.text))
        self.assertEqual(self.P.text, "example tex")

        self.P.delete(len(self.P.text) - 4 , len(self.P.text))
        self.assertEqual(self.P.text, "example")

    def test_delete_middle(self):
        self.P.delete(2, 7)
        self.assertEqual(self.P.text, "ex text")

    def test_delete_multiple(self):
        self.P.insert(7, " hedi")
        self.P.insert(9, "am")
        self.assertEqual(self.P.text, "example hamedi text")
        self.P.delete(7, 14)
        self.assertEqual(self.P.text, "example text")

        self.P.insert(0, "edi ")
        self.P.insert(0, "ham")
        self.assertEqual(self.P.text, "hamedi example text")
        self.P.delete(0, 7)
        self.assertEqual(self.P.text, "example text")

    def test_delete_all(self):
        self.P.delete(0, len(self.P.text))
        self.assertEqual(self.P.text, "")

        # delete all after insertion
        self.P.insert(0, "hamedi is going to delete")
        self.assertIsNone(self.P.head.next)
        self.P.delete(0, len(self.P.text))
        self.assertEqual(self.P.text, "")

    def test_delete_empty(self):
        self.P.delete(0,0)
        self.assertEqual(self.P.text, "example text")
    
    def test_delete_piece(self):
        self.P.insert(0, "hamedi ")
        self.assertEqual(self.P.text, "hamedi example text")
        self.P.delete(0, 7)
        self.assertEqual(self.P.text, "example text")

        self.P.insert(5, "hamedi ")
        self.assertEqual(self.P.text, "examphamedi le text")
        self.P.delete(5, 12)
        self.assertEqual(self.P.text, "example text")