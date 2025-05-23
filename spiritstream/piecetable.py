from dataclasses import dataclass
from typing import Literal, Optional, Tuple

@dataclass
class Piece:
    source:Literal["original", "add"]
    offset:int
    length:int
    prev:Optional["Piece"] = None
    next:Optional["Piece"] = None

class PieceTable:
    def __init__(self, text):
        self.original = text
        self.add = ""
        self.head:Optional[Piece] = Piece("original", 0, len(text), None, None)
    
    def insert(self, pos, text:str):
        if len(text) == 0: return
        if self.head == None: self.head = Piece("add", 0, len(text), None, None)
        else:
            p, o = self._find_node(pos)
            if o == p.length: # insert at end
                if p.offset + p.length == len(self.add) and p.source == "add": p.length += len(text) # piece is at the end of the add buffer already
                else:
                    np = Piece("add", len(self.add), len(text), p, p.next)
                    p.next = np
                    if np.next != None: np.next.prev = np
            elif o == 0: # insert at start
                np = Piece("add", len(self.add), len(text), p.prev, p)
                p.prev = np
                if np.prev != None: np.prev.next = np
                if p == self.head: self.head = np
            else: # split inbetween
                np = Piece("add", len(self.add), len(text), p, None)
                p2 = Piece(p.source, p.offset + o, p.length - o, np, p.next)
                p.length = o
                p.next = np
                np.next = p2
                if p2.next != None: p2.next.prev = p2
        self.add += text

    def delete(self, start:int, end:int):
        """Deletes text from start to end indices. Works like python slice, where the start index is included and the end index isn't. eg. delete(1,2) deletes only at index 1"""
        if (l:=end-start) == 0: return
        if self.head == None: raise IndexError(f"Cannot delete from empty text")
        assert l > 0
        p0, o0 = self._find_node(start)
        p1, o1 = self._find_node(end)
        if p0 == p1: # deletion contained in a single piece
            if o0 == 0:
                p0.offset += o1
                p0.length -= o1
            elif o1 == p0.length: p0.length = o0
            else:
                np = Piece(p0.source, p0.offset + o1, p0.length - o1, prev=p0, next=p0.next)
                p0.length = o0
                p0.next = np
            self._check_0_length(p0)
        else: # deletion spands multiple pieces
            p0.length = o0
            while (p := p0.next) != p1:
                p0.next = p.next
                del p
            p0.next = p1
            p1.prev = p0
            self._check_0_length(p0)
            p1.offset += o1
            p1.length -= o1
            self._check_0_length(p1)
            
    def _check_0_length(self, node):
        """Helper for self.delete that safely removes nodes with length 0"""
        if node.length == 0 and self.head != None:
            if node != self.head:
                node.prev.next = node.next
                node.next.prev = node.prev
            else:
                self.head = node.next
                if self.head != None: self.head.prev = None
            del node

    def _find_node(self, pos) -> Tuple[Piece, int]:
        """Returns piece that covers this text position and the position offset relative to the start of the piece."""
        s, head = 0, self.head
        while head and (s:=s+head.length) < pos: head=head.next
        if head == None: raise IndexError(f"Index {pos} outside of range")
        return head, head.length - (s - pos)
    
    @property
    def text(self):
        res, node = [], self.head
        while node:
            res.append(getattr(self, node.source)[node.offset:node.offset+node.length])
            node = node.next
        return "".join(res)