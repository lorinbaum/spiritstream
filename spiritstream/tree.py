from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Union, Generator, Any, Iterable, Dict, Tuple
from spiritstream.helpers import SPACES
import re

PATTERN = re.compile("|".join([
    r"^(?P<empty_line>[ \t]*?$)",
    r"^(?P<codeblock>```[\s\S]*?\n```|~~~[\s\S]*?\n~~~)",
    r"^ {0,3}(?P<horizontal_rule>\*(?:[ \t]*\*){2,}[ \t]*|_(?:[ \t]*_){2,}[ \t]*|-(?:[ \t]*-){2,}[ \t]*)$",
    r"^(?P<listitem>[ \t]*(?:\d+[\.\)]\s+|[-*+][ \t]+))",
    r"^(?P<indented_line>(?: {0,3}\t| {4,})[\s\S]*?$)",
    r"^ {0,3}(?P<heading>#+)[\t ]+",
    r"^ {0,3}(?P<blockquote_line>>+)",
    r"^ {0,3}(?P<paragraph_line>)(?!\s)",

    r"(?P<bolditalic_toggle>(?<![\*\s\\])\*\*\*(?![\*\s])|(?<![_\s\\])___(?![_\s]))", # don't know whether it's start or end
    r"(?P<bolditalic_start>(?<!\\)\*\*\*(?=[^\*\s])|(?<!\\)___(?=[^_\s]))",
    r"(?P<bolditalic_end>(?<![\*\s\\])\*\*\*|(?<![_\s\\])___)",

    r"(?P<bold_toggle>(?<![\*\s\\])\*\*(?![\*\s])|(?<![_\s\\])__(?![_\s]))", # don't know whether it's start or end
    r"(?P<bold_start>(?<!\\)\*\*(?=[^\*\s])|(?<!\\)__(?=[^_\s]))",
    r"(?P<bold_end>(?<![\*\s\\])\*\*|(?<![_\s\\])__)",

    r"(?P<italic_toggle>(?<![\*\s\\])\*(?![\*\s])|(?<![_\s\\])_(?![_\s]))", # don't know whether it's start or end
    r"(?P<italic_start>(?<!\\)\*(?=[^\*\s])|(?<!\\)_(?=[^_\s]))",
    r"(?P<italic_end>(?<![\*\s\\])\*|(?<![_\s\\])_)",

    r"(?P<strikethrough_toggle>(?<![~\s\\])~~(?=[^~\s]))", # don't know whether it's start or end
    r"(?P<strikethrough_start>(?<!\\)~~(?=[^~\s]))",
    r"(?P<strikethrough_end>(?<![~\s\\])~~)",

    r"(?P<wikilink_embed_start>(?<!\\)!\[\[(?=[^\[]))",
    r"(?P<wikilink_start>(?<!\\)\[\[(?=[^\[]))",
    r"(?P<markdownlink_embed_start>(?<!\\)!\[(?=[^\[]))",
    r"(?P<markdownlink_start>(?<!\\)\[)",
    r"(?P<markdownlink_switch>(?<!\\)\]\()",
    r"(?P<wikilink_end>(?<![\\\]])\]\])",

    r"(?P<opening_parenthesis>(?<!\\)\()",
    r"(?P<closing_parenthesis>(?<!\\)\))",

    r"(?P<double_inline_code>(?<![`\\])``(?!`))",
    r"(?P<inline_code>(?<![`\\])`(?!`))",

    r"(?P<text>[\s\S]+?)"
]), re.MULTILINE)

BLOCK_TOKENS = ["heading", "blockquote_line", "paragraph_line", "horizontal_rule", "codeblock", "empty_line", "listitem", "endoffile"]

@dataclass(frozen=True)
class Token:
    name:str
    start:int
    end:int
    data:Any = None
    def __repr__(self): return f"Token({self.name}, {self.start}, {self.end}, {self.data})"


class K(Enum): # Kind of node
    SS = auto() # top of the tree
    SCENE = auto()
    FRAME = auto()

    BODY = auto()
    H1, H2, H3, H4, H5, H6 = auto(), auto(), auto(), auto(), auto(), auto()
    BI, B, I, S = auto(), auto(), auto(), auto() # BI = BOLD+ITALIC
    A, IMG = auto(), auto()
    INLINE_CODE, DOUBLE_INLINE_CODE = auto(), auto()
    EMPTY_LINE = auto()
    P = auto()
    HR = auto()
    CB = auto() # codeblock
    BLOCKQUOTE = auto()
    _LIST, UL, OL, LI = auto(), auto(), auto(), auto()
    TEXT = auto()

INLINE_NODES = {K.BI, K.B, K.I, K.S, K.A, K.IMG, K.INLINE_CODE, K.DOUBLE_INLINE_CODE, K.TEXT}
LINKS = {K.A, K.IMG}

class Node:
    def __init__(self, k:K, parent:"Node", children:List["Node"]=None, data:Dict=None, **kwargs):
        self.k, self.parent = k, parent
        self.children = children if children is not None else []
        self.data = data if data is not None else {}
        self.data.update(kwargs)

    def __getattr__(self, name): # Called when attribute is not found normally
        if hasattr(self, "data") and name in self.data: return self.data[name]
        return None

    def __setattr__(self, name, value):
        if name in ("k", "parent", "children", "data"): super().__setattr__(name, value) # Protect core attributes
        else: self.data[name] = value
    
    def __repr__(self): return  f"\033[32m{self.k.name}\033[0m(\033[94mdata=\033[0m{self.data})"

def tokenize(text:str) -> Generator[Token, None, None]:
    text_start = None # because the non-greedy pattern matches single characters, accumulates "text" matches and stores start and end.
    for m in PATTERN.finditer(text):
        for name, value in m.groupdict().items():
            if value is not None:
                match name:
                    case "heading": data = m.group("heading").count("#") # heading level
                    case "indented_line":
                        t = m.group("indented_line").replace("\t", " " * SPACES)
                        stripped = t.lstrip(" ")
                        depth = len(t) - len(stripped)
                        data = (depth, stripped) # depth of indent
                    case "codeblock":
                        lines = m.group("codeblock").splitlines()
                        language = lines[0].strip(" `~")
                        code = lines[1:-1]
                        data = (language, code)
                    case "listitem":
                        stripped = (t:=m.group("listitem")).lstrip("\t ")
                        indent = len(t[:len(t) - len(stripped)].replace("\t", " " * SPACES))
                        if stripped[0].isnumeric(): listtype = K.OL
                        elif stripped.startswith(("+", "*", "-")): listtype = K.UL
                        else: raise AssertionError(f"'{stripped}' invalid list type")
                        digit = int(t.strip("\t .)")) if listtype == K.OL else None
                        data = (indent, listtype, digit)
                    case "blockquote_line": data = m.group("blockquote_line").count(">")
                    case "text" if text_start is None: text_start = m.start()
                    case _: data = None
                if name != "text":
                    if text_start is not None: yield Token("text", text_start, m.start()); text_start = None
                    yield Token(name, m.start(), m.end(), data)
    if text_start is not None: yield Token("text", text_start, len(text))
    yield Token("endoffile", len(text), len(text))

def treeify(tokens:Iterable[Token], head=None) -> Node:
    node = head = Node(K.BODY, None, []) if head is None else head # node I am currently adding to aka. most recent unclosed node
    for tok in tokens:
        while tok.name in BLOCK_TOKENS and node.k in INLINE_NODES:
            # Open ended links are invalid, so replaced by text nodes (merge with adjacent text nodes if any). Other inline nodes just end.
            if node.k in LINKS:
                # k = K.TEXT and then add_text effectively replaces this with new text node while allowing merging
                node.k, start, node.end = K.TEXT, node.start, tok.start 
                add_text((node:=node.parent), start, tok.start)
            else:
                if node.k is K.BI:
                    node.k, node.children = K.B, [Node(K.I, node, node.children, start = node.start + 2, end = tok.start)]
                    for n in node.children[0].children: n.parent = node.children[0]
                node.end, node = tok.start, node.parent
        if tok.name == "endoffile": node.data.setdefault("end", tok.end)
        match tok.name:
            case "heading": node = start_node(Node(getattr(K, f"H{min(tok.data, 6)}"), head, start = tok.start))
            case "empty_line": node = start_node(Node(K.EMPTY_LINE, head, start = tok.start))
            case "paragraph_line" if node.k is not K.P: node = start_node(Node(K.P, head, start = tok.start))
            case "horizontal_rule": node = start_node(Node(K.HR, head, start = tok.start))
            case "codeblock": node = start_node(Node(K.CB, head, start = tok.start, language = tok.data[0], code = tok.data[1]))
            case "indented_line" if tok.data[0] >= 4:
                if node.k is not K.CB: node = start_node(Node(K.CB, head, start = tok.start, line = [" "*(tok.data[0]-4) + tok.data[1]]))
                else: node.line.append(" "*(tok.data[0]-4) + tok.data[1])
            case "blockquote_line":
                if node.k is not K.BLOCKQUOTE: node = start_node(Node(K.BLOCKQUOTE, head, start = tok.start, depth = 1))
                while node.depth < tok.data: node = start_node(Node(K.BLOCKQUOTE, node, start = tok.start, depth = node.depth + 1))
                while node.k is K.BLOCKQUOTE and node.depth > tok.data: node = end_node(node, tok.start)
            case "listitem":
                newindent, newlisttype, digit = tok.data
                newindent //= SPACES
                listnode = node if node.k in [K._LIST, K.UL, K.OL] else node.parent if node.k is K.LI and node.parent and node.parent.k in [K._LIST, K.UL, K.OL] else None
                if listnode is None: node = listnode = start_node(Node(K._LIST, head, start = tok.start, indent = 0))
                while (previndent:=listnode.indent) < newindent:
                    if node.k is not K.LI: node = start_node(Node(K.LI, listnode, start = tok.start))
                    node = listnode = start_node(Node(K._LIST, node, start = tok.start, indent = previndent + 1))
                if node is not listnode: node = end_node(node, tok.start)
                assert node is listnode
                while (previndent:=listnode.indent) > newindent:
                    node = end_node(listnode, tok.start)
                    assert node.k is K.LI
                    listnode = node = end_node(node, tok.start)
                if listnode.k is not newlisttype:
                    if listnode.k is K._LIST: listnode.k = newlisttype
                    else:
                        node = end_node(node, tok.start)
                        node = listnode = start_node(Node(newlisttype, node, start = tok.start, indent = newindent))
                assert node is listnode and node.k is tok.data[1]
                node = start_node(Node(K.LI, node, start = tok.start, digit = digit))
            
            case "wikilink_embed_start" if node.k not in LINKS: node = start_node(Node(K.IMG, node, start = tok.start, linktype = "wiki"))
            case "wikilink_start" if node.k not in LINKS: node = start_node(Node(K.A, node, start = tok.start, linktype = "wiki"))
            case "wikilink_end" if node.k in LINKS:
                if node.linktype == "wiki":
                    if node.children: node.href = (node.children[0].start, node.children[-1].end)
                    node = end_node(node, tok.start)
                else: node.linktype = "md switch" # this token can end markdownlinks too. data is used when parsing next token to see if conditions for a full switch are satisfied
            case "markdownlink_embed_start" if node.k not in LINKS: node = start_node(Node(K.IMG, node, start = tok.start, linktype = "md"))
            case "markdownlink_start" if node.k not in LINKS: node = start_node(Node(K.A, node, start = tok.start, linktype = "md"))
            case "markdownlink_switch" if node.k in LINKS: 
                if node.linktype == "wiki": # can switch wikilinks too in cases like [[link](link) the first [ is ignored, so start is shifted
                    shift = 1 if node.k is K.A else 2 # if k.IMG, there is a "!" that is being ignored too
                    if len(node.parent.children) > 1 and (n:=node.parent.children[-2].k is K.TEXT) and n.end == node.start: n.end += shift
                    else: node.parent.children.insert(-1, Node(K.TEXT, node.parent, start = node.start, end = node.start + shift))
                    node.start += 1 if node.k is K.A else 2 # if k.IMG, there is a "!" that is being ignored too
                    node.k = K.A
                node.linktype = "md switched" 
                node.switchidx = len(node.children) # used when closing link to determine children that form url part of the link
        
            case "opening_parenthesis" if node.k in LINKS and node.linktype == "md switch": node.linktype, node.switchidx = "md switched", len(node.children)
            case "closing_parenthesis" if node.k in LINKS and node.linktype == "md switched":
                node.href = None if len(node.children) == node.switchidx else (node.children[node.switchidx].start, node.children[-1].end)
                node.children = node.children[:node.switchidx] # remove children that were part of the link
                node = end_node(node, tok.start)

            # NOTE: check Node kind because text will match newline at end of empty_line and codeblock which is useless
            case "text" if not (node.k in [K.EMPTY_LINE, K.CB] and tok.end - tok.start == 1): add_text(node, tok.start, tok.end )

            case _ if node.k not in LINKS or node.linktype == "md": # inline formatting only allowed outside of links except in the label part of a markdown link                
                match tok.name:
                    case "italic_start" if node.k is not K.BI: node = start_node(Node(K.I, node, start=tok.start))
                    case "italic_toggle"|"italic_end":
                        if node.k is K.I: node = end_node(node, tok.end)
                        elif node.k is K.BI: # replace bolditalic with opening bold and opening italic. Italic ends here
                            node.k, node.children = K.B, [Node(K.I,node, node.children, start=node.start+2, end=tok.end)]
                            for n in node.children[0].children: n.parent = node.children[0]
                        elif tok.name == "italic_toggle": node = start_node(Node(K.I, node, start=tok.start))
                        else: add_text(node, tok.start, tok.end) # it's italic end but not inside italic node. replace with text

                    case "bold_start" if node.k is not K.BI: node = start_node(Node(K.B, node, start=tok.start))
                    case "bold_toggle"|"bold_end":
                        if node.k is K.B: node = end_node(node, tok.end)
                        elif node.k is K.BI: # replace bolditalic with opening italic and opening bold. Bold ends here
                            node.k, node.children = K.I, [Node(K.B,node, node.children, start=node.start+1, end=tok.end)]
                            for n in node.children[0].children: n.parent = node.children[0]
                        elif tok.name == "bold_toggle": node = start_node(Node(K.B, node, start=tok.start))
                        else: add_text(node, tok.start, tok.end) # it's bold end but not in bold node. replace with text

                    case "bolditalic_start" if node.k not in [K.I, K.B]: node = start_node(Node(K.BI, node, start=tok.start))
                    case "bolditalic_toggle"|"bolditalic_end":
                        if node.k is K.BI:
                            node.children = [Node(K.I,node, node.children, start=node.start+2, end=tok.end-2)]
                            for n in node.children[0].children: n.parent = node.children[0]
                            node.end, node.k, node = tok.end, K.B, node.parent
                        elif node.k is K.I:
                            node = end_node(node, tok.start+1)
                            if node.k is K.B: end_node(node, tok.end)
                            else: node = start_node(Node(K.B, node, start=tok.start+1))
                        elif node.k is K.B:
                            node = end_node(node, tok.start+2)
                            if node.k is K.I: node = end_node(node, tok.end)
                            else: node = start_node(Node(K.I, node, start=tok.start+2))
                        elif tok.name == "bolditalic_toggle": node = start_node(Node(K.BI, node, start=tok.start))
                        else: add_text(node, tok.start, tok.end) # bolitalic end but not in bolitalic. replace with text

                    case "strikethrough_start": node = start_node(Node(K.S, node, start=tok.start))
                    case "strikethrough_toggle"|"strikethrough_end":
                        if node.k is K.S: node = end_node(node, tok.start)
                        elif tok.name == "strikethrough_toggle": node = start_node(Node(K.S, node, start=tok.start))
                        else: add_text(node, tok.start, tok.end) # strikethrough end but not in strikethrough. replace with text

                    case "double_inline_code":
                        if node.k is K.DOUBLE_INLINE_CODE: node = end_node(node, tok.start)
                        elif node.k is K.INLINE_CODE:
                            node = end_node(node, tok.start)
                            node = start_node(Node(K.INLINE_CODE, node, start=tok.start + 1))
                        else: node = start_node(Node(K.DOUBLE_INLINE_CODE, node, start=tok.start))

                    case "inline_code":
                        if node.k is K.INLINE_CODE: node = end_node(node, tok.start)
                        elif node.k is not K.DOUBLE_INLINE_CODE: node = start_node(Node(K.INLINE_CODE, node, start=tok.start))
                    case _: add_text(node, tok.start, tok.end) # no valid match means it should be treated as text
            case _: add_text(node, tok.start, tok.end) # no valid match means it should be treated as text
    for n in reversed(list(walk(head))): # add missing ends. They exist because block nodes add themselves to head without closing previous nodes
        if isinstance(n, Node) and not getattr(n, "end", None): n.end = n.children[-1].end
    return head

def parse(text:str, head=None) -> Node: return treeify(tokenize(text), head)

def add_text(node:Node, start:int, end:int):
    """Merges with existing overlapping TEXT nodes if any or inserts a new text node. Does not fix existing adjacent text outside of start, end"""
    if start != end:
        overlaps = [(i, n) for (i, n) in enumerate(node.children) if n.k is K.TEXT and not (n.end < start or n.start > end)]
        start, end = min((start, *(n.start for _, n in overlaps))), max((end, *(n.end for _, n in overlaps)))
        for _, n in overlaps: node.children.remove(n)
        node.children.insert(min((len(node.children), *(i for i,_ in overlaps))), Node(K.TEXT, node, start=start, end=end))

def start_node(child:Node) -> Node: child.parent.children.append(child); return child

def end_node(node:Node, end:int) -> Node: node.end, node = end, node.parent; return node

def walk(node, level=None, seen=None) -> Generator[Node, None, None]:
    assert id(node) not in (seen := set() if seen is None else seen) or seen.add(id(node))
    yield node if level is None else (node, level)
    for n in getattr(node, "children", []): yield from walk(n, None if level is None else level+1, seen)

def show(node): [print(f"{' '*SPACES*l}{n}") for n,l in walk(node, level=0)]

def serialize(head:Node) -> str:
    """node tree to html"""
    ret = f"<!DOCTYPE html>"
    assert head.k is K.FRAME
    text = head.text
    prevNode, prevLevel = None, -1
    for node, level in walk(head, level=0):
        if isinstance(node, Node):
            assert node.k is not K.BI, "bolditalic nodes should be replace by bold and italic nodes during parsing"
            if node.k is K.TEXT: ret += text[node.start:node.end].strip("\n ").replace("\n", "<br>")
            else:
                while prevLevel >= level:
                    ret += htmltag(prevNode, open=False)
                    assert prevNode.parent is not None
                    prevNode, prevLevel = prevNode.parent, prevLevel - 1
                ret += htmltag(node)
                prevNode = node
                prevLevel = level
    while prevNode is not head:
        ret += htmltag(prevNode, open=False)
        prevNode = prevNode.parent
    return ret + htmltag(head, open=False)

def htmltag(node:Node, open=True) -> str:
    # TODO: other special kinds, like links, codeblocks, inline code
    match node.k:
        case K.FRAME:
            return f"""<html><head><link rel="stylesheet" href="./test/test.css"/><title>{getattr(node, "title", "Unnamed frame")}</title></head>""" \
            if open else "</html>"
        case K.EMPTY_LINE: return "<br>" if open else ""
        case _: return f"<{'' if open else '/'}{node.k.name.lower()}>"

INTERNAL_LINK_TARGETS = set()
def href(node:Node, text:str) -> str:
    assert node.name in ["link", "embed"] and node.data in ["wiki", "md switched"]
    assert (n:=node.children[0]) if node.data == "wiki" else (n:=node.children[1]).name == "text"
    if (t:=text[n.start:n.end]).startswith("#"):
        assert node.name != "embed", "embeding headings not supported yet"
        INTERNAL_LINK_TARGETS.add(t:=t[1:])
    return t


CSS_PATTERN = re.compile("|".join([
    r"(?P<comment>/\*.*?\*/)",
    r"(?P<string>\"([^\"\\]|\\.)*\"|'([^'\\]|\\.)*')",
    r"(?P<number>\d\.d|\d+(?:\.\d+)?|\d*(?:\.\d+))",
    r"(?P<dimension>em|px|%|rem)",
    r"(?P<hash>#[\da-fA-F]{3,8})",
    r"(?P<at>@[\w-]+)",
    r"(?P<open_parenthesis>\()",
    r"(?P<close_parenthesis>\))",
    r"(?P<open_bracket>\[)",
    r"(?P<close_bracket>\])",
    r"(?P<open_curly>\{)",
    r"(?P<close_curly>\})",
    r"(?P<comma>,)",
    r"(?P<colon>:)",
    r"(?P<semicolon>;)",
    r"(?P<delimiter>[*/\-+~|^$=])",
    r"(?P<identity>-?[_a-zA-Z]+[_a-zA-Z0-9-]*)",
    r"(?P<space>\s+)"
]))

@dataclass
class CSSNode:
    name:str
    parent:Node
    children:List[Node] = None
    data:Any = None
    status:Optional[str] = None
    def __repr__(self): return  f"\033[32m{self.__class__.__name__}\033[0m({self.name}, \033[94mdata=\033[0m{repr(self.data)})"

class CSS(Enum):
    STYLESHEET = auto()
    RULE = auto()
    DECLARATION = auto()
    IDENTITY = auto()
    FUNCTION = auto()

class Status(Enum):
    OPENED = auto()
    COMMA = auto() # awaiting another value to add to data
    PSEUDOCLASS = auto() # awaiting pseudo class name
    CLOSED = auto()
    NAMED = auto() # property name is already set, awaiting value

def tokenizeCSS(text:str):
    head = CSSNode(CSS.STYLESHEET, None, [])
    node = head
    for m in CSS_PATTERN.finditer(text):
        for name, value in m.groupdict().items():
            if value is not None:
                match name:
                    case "identity":
                        if node.name is CSS.STYLESHEET: node = start_node(CSSNode(CSS.RULE,  node, [], (value,)))
                        elif node.name is CSS.RULE:
                            status = getattr(node, "status", None)
                            if status is Status.PSEUDOCLASS:
                                node.data = (*node.data[:-1], node.data[-1]+":" + value)
                                node.status = None
                            elif status is Status.OPENED: node = start_node(CSSNode(CSS.DECLARATION,  node, [], (value,)))
                            elif status is Status.COMMA:
                                node.data += (value,)
                                node.status = None
                            else: raise NotImplementedError(node.name, status)
                        elif node.name is CSS.DECLARATION: node = start_node(CSSNode(CSS.IDENTITY,  node, [], (value,)))
                        else: raise NotImplementedError(node.name)
                    case "at":
                        assert node.name is CSS.STYLESHEET
                        node = start_node(CSSNode(CSS.RULE,  node, [], value))
                    case "open_curly":
                        assert node.name is CSS.RULE and node.children == []
                        node.status = Status.OPENED
                    case "open_parenthesis":
                        assert node.name is CSS.IDENTITY
                        node.name = CSS.FUNCTION
                        node.status = Status.OPENED
                    case "close_parenthesis":
                        assert node.name is CSS.FUNCTION
                        node.status = Status.CLOSED
                        node = node.parent
                    case "close_curly":
                        while node.name in [CSS.FUNCTION, CSS.DECLARATION, CSS.RULE, CSS.IDENTITY]:
                            node.status = Status.CLOSED
                            node = node.parent
                        assert node.name is CSS.STYLESHEET
                    case "comma":
                        assert node.name is CSS.RULE and getattr(node, "status", None) is None
                        node.status = Status.COMMA
                    case "colon":
                        if node.name is CSS.RULE:
                            assert getattr(node, "status", None) is None
                            node.status = Status.PSEUDOCLASS
                        elif node.name is CSS.DECLARATION:
                            assert isinstance(node.data, tuple) and len(node.data) == 1
                            node.status = Status.NAMED
                        else: raise NotImplementedError(node.name)
                    case "semicolon":
                        while node.name in [CSS.FUNCTION, CSS.DECLARATION, CSS.IDENTITY]:
                            node.status = Status.CLOSED
                            node = node.parent
                        assert node.name is CSS.RULE
                    case "string":
                        if node.name is CSS.DECLARATION: assert node.status is Status.NAMED
                        elif node.name is CSS.FUNCTION: assert node.status is Status.OPENED
                        else: raise NotImplementedError(node.name)
                        node.data += (value.strip("\"\'"),)
                    case "number":
                        if node.name is CSS.DECLARATION: assert node.status is Status.NAMED
                        elif node.name is CSS.FUNCTION: assert node.status is Status.OPENED
                        else: raise NotImplementedError(node.name)
                        if value.startswith("."): value = 0 + value
                        node.data += (float(value),)
                    case "hash":
                        if node.name is CSS.DECLARATION: assert node.status is Status.NAMED
                        elif node.name is CSS.FUNCTION: assert node.status is Status.OPENED
                        else: raise NotImplementedError(node.name)
                        node.data += (value,)
                    case "dimension":
                        assert node.name is CSS.DECLARATION and node.status is Status.NAMED
                        node.data += (value,)
                    case "delimiter":
                        if node.name is CSS.STYLESHEET and value == "*": node = start_node(CSSNode(CSS.RULE,  node, [], (value,)))
                        else: raise NotImplementedError
    while node.name is not CSS.STYLESHEET:
        node.status = Status.CLOSED
        node = node.parent
    node.status = Status.CLOSED # close head
    assert node is head
    return head

@dataclass(frozen=True)
class Color:
    r:float
    g:float
    b:float
    a:float

def color_from_hex(x:int) -> Color:
    assert x <= 0xffffffff
    if x <= 0xfff: return Color(((x & 0xf00) >> 8) / 15, ((x & 0xf0) >> 4) / 15, (x & 0xf) / 15, 1.0)
    elif x <= 0xffff: return Color(((x & 0xf000) >> 12) / 15, ((x & 0xf00) >> 8) / 15, ((x & 0xf0) >> 4) / 15, (x & 0xf) / 15)
    elif x <= 0xffffff: return Color(((x & 0xff0000) >> 16) / 255, ((x & 0xff00) >> 8) / 255, (x & 0xff) / 255, 1.0)
    else: return Color(((x & 0xff000000) >> 24) / 255, ((x & 0xff0000) >> 16) / 255, ((x & 0xff00) >> 8) / 255, (x & 0xff) / 255)

def css_to_dict(head:CSSNode) -> Dict:
    css_dict = {}
    selectors = None
    k = None
    for node in walk(head):
        if node.name is CSS.RULE: selectors = node.data if isinstance(node.data, tuple) else (node.data,)
        elif node.name is CSS.DECLARATION:
            if len(node.data) == 1: k = node.data[0] # value is in either function or identity children
            else:
                # make line-height unit explicit if not given
                if node.data[0] == "line-height" and isinstance(node.data[1], (float, int)): node.data= ("line-height", (node.data[1],"em",)) 
                k, v = node.data[0], node.data[1:]
                k, v = expand_css_shorthand(k, v)
                for k0, v0 in zip(k, v):
                    v0 = v0[0] if isinstance(v0, tuple) and len(v0) == 1 else v0
                    v0 = color_from_hex(int(v0[1:], base=16)) if isinstance(v0, str) and v0.startswith("#") else v0
                    for s in selectors: css_dict.setdefault(s, {})[k0] = v0
        elif node.name in (CSS.IDENTITY, CSS.FUNCTION):
            if node.name is CSS.IDENTITY:
                k, _ = expand_css_shorthand(k, node.data[0])
                for s in selectors:
                    for k0 in k:
                        css_dict.setdefault(s, {})[k0] = node.data[0]
            else:
                assert len(node.data) == 2
                f, arg = node.data
                for s in selectors: css_dict.setdefault(s, {}).setdefault(k, {})[f] = arg
                # if not isinstance(current, list): css_dict[s][k] = [current]
                # css_dict[s][k].append(node)
    return css_dict

# NOTE: margin: 0, auto not supported because mix of identity and other
# NOTE: margin: calc(100% - 5px) not supported

def expand_css_shorthand(k, v) -> Tuple[Tuple, Tuple]:
    # process shorthands for padding and margin
    if k in ["padding", "margin"]:
        k = (k+"-top", k+"-right", k+"-bottom", k+"-left")
        v_u = [] # value - unit pairs
        for v0 in v:
            if v0 in ["em", "rem", "px"]:
                assert len(v_u) > 0 and len(v_u[-1]) == 1
                v_u[-1] = (*v_u[-1], v0)
            else:
                assert len(v_u) == 0 or len(v_u[-1]) == 2 or v_u[-1][0] == 0 or isinstance(v_u[-1][0], str)
                v_u.append((v0,))
        assert len(v_u) in [1, 2, 4]
        if len(v_u) == 1: v = (v_u[0],) * 4
        elif len(v_u) == 2: v = (v_u[0], v_u[1]) * 2
        elif len(v_u) == 4: v = v_u
        return k, v
    return (k,), (v,)

def parseCSS(text:str) -> Dict: return css_to_dict(tokenizeCSS(text))