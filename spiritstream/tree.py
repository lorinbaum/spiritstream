from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Union, Generator, Any, Iterable, Dict, Tuple
from spiritstream.helpers import SPACES
from pathlib import Path
import re, os, time

PATTERN = re.compile("|".join([
    r"^(?P<codeblock>```[\s\S]*?\n```\n|~~~[\s\S]*?\n~~~\n)",
    r"^ {0,3}(?P<horizontal_rule>\*(?:[ \t]*\*){2,}[ \t]*$|_(?:[ \t]*_){2,}[ \t]*$|-(?:[ \t]*-){2,}[ \t]*$)",
    r"^(?P<listitem>[ \t]*(?:\d+[\.\)][ \t]+|[-*+][ \t]+))",
    r"^(?P<indented_line>(?: {0,3}\t| {4,}).*)",
    r"^ {0,3}(?P<heading>#+)[\t ]+",
    r"^ {0,3}(?P<blockquote_line>>+)",
    r"(?P<new_line>\n)",
    r"^ {0,3}(?P<paragraph_line>)(?=[^\n])",

    r"(?P<emphasis>(?<!\\)\*+)",

    r"(?P<wikilink_embed_start>(?<!\\)!\[\[(?=[^\[\]]))",
    r"(?P<wikilink_start>(?<!\\)\[\[(?=[^\[\]]))", # prevent wikilinks like [[]link]], but guarantees that the link is not empty
    r"(?P<markdownlink_embed_start>(?<!\\)!\[(?=[^\[\]]))",
    r"(?P<markdownlink_start>(?<!\\)\[(?=[^\]]))",
    r"(?P<markdownlink_switch>(?<!\\)\]\()",
    r"(?P<wikilink_end>(?<![\\\]])\]\])",

    r"(?P<opening_parenthesis>(?<!\\)\()",
    r"(?P<closing_parenthesis>(?<!\\)\))",

    r"(?P<double_inline_code>(?<![`\\])``(?=[^`]))",
    r"(?P<inline_code>(?<![\\])`(?=[^`]))", # no negative lookbehind for another ` because that would already have matched with double_inline_code
]), re.MULTILINE)

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
    LINE = auto()

    ANY = auto() # used for CSS selectors

    BODY = auto()
    H1, H2, H3, H4, H5, H6 = auto(), auto(), auto(), auto(), auto(), auto()
    B, I = auto(), auto()
    A, IMG = auto(), auto()
    CODE = auto()
    EMPTY_LINE = auto()
    P = auto()
    HR = auto()
    BLOCKQUOTE = auto()
    UL, OL, LI = auto(), auto(), auto()
    TEXT = auto()

INLINE_NODES = {K.B, K.I, K.A, K.IMG, K.CODE, K.TEXT, K.LINE}
LINKS = {K.A, K.IMG}

class Node:
    def __init__(self, k:K, parent:"Node", children:List["Node"]=None, data:Dict=None, **kwargs):
        self.k, self.parent = k, parent
        if children: # for easier notation when creating nested stuff
            for c in children: c.parent = self
        self.children = children if children is not None else []
        self.data = data if data is not None else {}
        self.data.update(kwargs)

    def __getattr__(self, name): # Called when attribute is not found normally
        if hasattr(self, "data") and name in self.data: return self.data[name]
        return None

    def __setattr__(self, name, value):
        if name in ("k", "parent", "children", "data"): super().__setattr__(name, value) # Protect core attributes
        else: self.data[name] = value

    def _format(self, v):
        if isinstance(v, bool): return repr(v)
        elif isinstance(v, (int, float)): return f"{v:.2f}"
        elif isinstance(v, list): return "[" + ", ".join(f"{i:.2f}" if isinstance(i, (int, float)) else repr(i) for i in v[:10]) + (", ...]" if len(v) > 10 else "]")
        elif isinstance(v, str): return repr(v[:50] + ("..." if len(v) > 50 else "")) 
        elif isinstance(v, dict): return "dict..."
        else: return repr(v)
    
    def __repr__(self): return  f"\033[32m{self.k.name}\033[0m("+", ".join([f"\033[94m{k}=\033[0m{self._format(v)}" for k,v in self.data.items()]) + ")"

def tokenize(text:str) -> Generator[Token, None, None]:
    i = 0
    for m in PATTERN.finditer(text):
        if m.start() > i: yield Token("text", i, m.start())
        i = m.end()
        for name, value in m.groupdict().items():
            if value is not None:
                match name:
                    case "heading": data = m.group("heading").count("#") # heading level
                    case "indented_line":
                        t = m.group("indented_line").replace("\t", " " * SPACES)
                        stripped = t.lstrip(" ")
                        depth = len(t) - len(stripped)
                        data = depth # depth of indent
                    case "codeblock":
                        lines = m.group("codeblock").splitlines()
                        language = lines[0].strip(" `~")
                        data = (language, len(lines[0]))
                    case "listitem":
                        stripped = (t:=m.group("listitem")).lstrip("\t ")
                        indent = len(t[:len(t) - len(stripped)].replace("\t", " " * SPACES))
                        if stripped[0].isnumeric(): listtype = K.OL
                        elif stripped.startswith(("+", "*", "-")): listtype = K.UL
                        else: raise AssertionError(f"'{stripped}' invalid list type")
                        digit = int(t.strip("\t .)")) if listtype == K.OL else None
                        data = (indent, listtype, digit)
                    case "blockquote_line": data = m.group("blockquote_line").count(">")
                    case "emphasis":
                        data = ((s0:=text[max(0, m.start()-1):m.start()]).isspace() or s0=="", (s1:=text[m.end():m.end()+1]).isspace() or s1=="")
                    case _: data = None
                yield Token(name, m.start(), m.end(), data)
                break
    if i < len(text): yield Token("text", i, len(text))
    # HACK: cursor positioning will look for nodes with start <= idx < end so last node ends 1 beyond
    yield Token("new_line", len(text), len(text)+1) 

def treeify(tokens:Iterable[Token], head=None) -> Node:
    # TODO: BODY start=0 is not guaranteed if the text has a head
    node = head = Node(K.BODY, None, start=0) if head is None else head # node stores what I am currently adding to aka. most recently unclosed
    for tok in tokens:
        in_link = any((n.k in LINKS for n in (node, *parents(node))))
        linknode = next((n for n in (node, *parents(node)) if n.k in LINKS)) if in_link else None
        in_code = any((n.k is K.CODE for n in (node, *parents(node))))
        match tok.name:
            case "new_line":
                if node is head: node.children.append(Node(K.EMPTY_LINE, head, [Node(K.TEXT, None, start=tok.start, end=tok.end)], start=tok.start, end=tok.end))
                if (n:=next((n for n in (node, *parents(node)) if n.k in [K.BLOCKQUOTE, K.HR] or n.k is K.CODE and "block" in n.cls), None)): add_text(n, tok.start, tok.end)
                while node is not head: node = end_node(node, tok.end)
            case "heading": node = start_node(Node(getattr(K, f"H{min(tok.data, 6)}"), head, start = tok.start))
            case "paragraph_line":
                if node is head and (not node.children or node.children[-1].k is not K.P): node, start = start_node(Node(K.P, head, start = tok.start)), tok.start
                else: node, start = node.children[-1], tok.start - 1 # includeget \n
                add_text(node, start, tok.end) # add any indent captured in the paragraph line token
            case "horizontal_rule": node = start_node(Node(K.HR, node, start=tok.start, end=tok.end))
            case "codeblock": 
                node = start_node(Node(K.CODE, head, start=tok.start, end=tok.end, language=tok.data[0], cls=["block"]))
                codeblock = node
                node = start_node(Node(K.P, node, start=tok.start, end=tok.start+tok.data[1]+1, cls=["code_language"]))
                add_text(node, tok.start+3, tok.start+tok.data[1] + 1) # \n
                add_text(codeblock, tok.start + tok.data[1]+1, tok.end - 4)
                node = start_node(Node(K.P, codeblock, start=tok.end-4, end=tok.end))
                add_text(node, tok.end-1, tok.end)
                node = head # prevent merging with following blocks
            case "indented_line" if tok.data >= 4:
                while node.children: node = node.children[-1] # get latest node
                node = next((n for n in (node, *parents(node)) if n.k is K.CODE and "block" in n.cls), None)
                if node is None: node = start_node(Node(K.CODE, head, start=tok.start, cls=["block"]))
                add_text(node, tok.start + tok.data, tok.end)
            case "blockquote_line":
                while node.children: node = node.children[-1] # get latest node
                node = next((n for n in (node, *parents(node)) if n.k is K.BLOCKQUOTE), None)
                if not node: node = start_node(Node(K.BLOCKQUOTE, head, start = tok.start, depth = 1))
                while node.depth < tok.data: node = start_node(Node(K.BLOCKQUOTE, node, start = tok.start, depth = node.depth + 1))
                while node.k is K.BLOCKQUOTE and node.depth > tok.data: node = end_node(node, tok.start)
            case "listitem":
                newindent, newlisttype, digit = tok.data
                while node.children: node = node.children[-1] # get latest node
                listnode = next((n for n in (node, *parents(node)) if n.k in (K.UL, K.OL)), None)
                if listnode is None:
                    if newindent >= 4: # not a list item but an indented line
                        if not (node.k is K.CODE and "block" in node.cls): node = start_node(Node(K.CODE, head, start = tok.start, cls=["block"]))
                        add_text(node, tok.start + newindent, tok.end)
                    else:
                        node = start_node(Node(newlisttype, head, indent=newindent, start=tok.start))
                        node = start_node(Node(K.LI, node, start = tok.start, digit = digit))
                else:
                    if listnode.indent < newindent:
                        if listnode.children: node = next((n for n in (node, *parents(node)) if n.k is K.LI))
                        else: node = start_node(Node(K.LI, listnode, start = tok.start))
                        node = listnode = start_node(Node(newlisttype, node, start = tok.start, indent = newindent))
                    node = listnode
                    while listnode.indent > newindent:
                        node = end_node(listnode, tok.start)
                        assert node.k is K.LI
                        listnode = node = end_node(node, tok.start)
                    if listnode.k is not newlisttype:
                        node = end_node(node, tok.start)
                        node = listnode = start_node(Node(newlisttype, node, start = tok.start, indent = newindent))
                    node = start_node(Node(K.LI, node, start = tok.start, digit = digit))
            
            case "wikilink_embed_start" if not in_link and not in_code: node = start_node(Node(K.IMG, node, start = tok.start, linktype = "wiki"))
            case "wikilink_start" if not in_link and not in_code: node = start_node(Node(K.A, node, start = tok.start, linktype = "wiki"))
            case "wikilink_end" if in_link and not in_code:
                linknode = next((n for n in (node, *parents(node)) if n.k in LINKS))
                if linknode.linktype == "wiki":
                    if linknode.children: linknode.href = (linknode.children[0].start, linknode.children[-1].end)
                    linknode.children = []
                    if linknode.k is K.A: add_text(linknode, node.start+2, tok.start)
                    while node is not linknode.parent: node = end_node(node, tok.end)
                else: node.linktype = "md switch" # this token can end markdownlinks too. data is used when parsing next token to see if conditions for a full switch are satisfied
            case "markdownlink_embed_start" if not in_link and not in_code: node = start_node(Node(K.IMG, node, start = tok.start, linktype = "md"))
            case "markdownlink_start" if not in_link and not in_code: node = start_node(Node(K.A, node, start = tok.start, linktype = "md"))
            case "markdownlink_switch" if in_link and not in_code:
                if linknode.linktype == "wiki": # can switch wikilinks too in cases like [[link](link) the first [ is ignored, so start is shifted
                    shift = 1 if linknode.k is K.A else 2 # if k.IMG, there is a "!" that is being ignored too
                    if len(linknode.parent.children) > 1 and (n:=linknode.parent.children[-2].k is K.TEXT) and n.end == linknode.start: n.end += shift
                    else: linknode.parent.children.insert(-1, Node(K.TEXT, linknode.parent, start = linknode.start, end = linknode.start + shift))
                    linknode.start += 1 if linknode.k is K.A else 2 # if k.IMG, there is a "!" that is being ignored too
                    linknode.k = K.A
                linknode.linktype = "md switched" 
                linknode.switchidx = len(linknode.children) # used when closing link to determine children that form url part of the link
        
            case "opening_parenthesis" if in_link and not in_code and linknode.linktype == "md switch": linknode.linktype, linknode.switchidx = "md switched", len(linknode.children)
            case "closing_parenthesis" if in_link and not in_code and linknode.linktype == "md switched":
                linknode.href = None if len(linknode.children) == linknode.switchidx else (linknode.children[linknode.switchidx].start, linknode.children[-1].end)
                if linknode.k is K.A: linknode.children = linknode.children[:linknode.switchidx] # remove children that were part of the link
                else:
                    linknode.alt = (linknode.children[0].start, linknode.children[0].end)
                    linknode.children = []
                while node is not linknode.parent: node = end_node(node, tok.end)

            case _ if not in_link or node.linktype == "md": # inline formatting only allowed outside of links except in the label part of a markdown link                
                in_italic, in_bold = map(lambda k: any((n.k is k for n in (node, *parents(node)))), (K.I, K.B))
                if in_italic: italicnode = next((n for n in (node, *parents(node)) if n.k is K.I))
                if in_bold: boldnode = next((n for n in (node, *parents(node)) if n.k is K.B))
                if in_code: codenode = next((n for n in (node, *parents(node)) if n.k is K.CODE))

                match tok.name:
                    case "emphasis" if not in_code:
                        before, _after = tok.data # whether space before / after. after is recalculated based on how far along I am in the emphasis
                        emphasis = tok.end - tok.start
                        while emphasis > 0:
                            emphasisnode = next((n for n in (node, *parents(node)) if n.k in [K.I, K.B]), None)
                            if emphasisnode and not before:
                                if emphasisnode.k is K.B:
                                    if emphasis >= 2:
                                        while node is not emphasisnode: node = end_node(node, tok.end-emphasis)
                                        emphasis -= 2
                                        node = end_node(node, tok.end-emphasis)
                                    else:
                                        if italicnode:=next((n for n in (node, *parents(node)) if n.k is K.I), None):
                                            while node is not italicnode: node = end_node(node, tok.end-emphasis)
                                            emphasis -= 1
                                            node = end_node(node, tok.end-emphasis)
                                            node = start_node(Node(K.B, node, start=tok.end-emphasis)) # continue overlappping bold node
                                        else:
                                            if not _after or emphasis > 1: # can start nodes
                                                node = start_node(Node(K.I, node, start=tok.end-emphasis))
                                                emphasis -= 1
                                            else:
                                                add_text(node, tok.end-emphasis, tok.end-emphasis+1)
                                                emphasis -= 1
                                elif emphasis >= 2 and (boldnode:=next((n for n in (node, *parents(node)) if n.k is K.B), None)):
                                    while node is not boldnode: node = end_node(node, tok.end-emphasis)
                                    emphasis -= 2
                                    node = end_node(node, tok.end-emphasis)
                                    node = start_node(Node(K.I, node, start=tok.end-emphasis)) # continue overlapping italic node
                                elif emphasis >= 2 and not boldnode and ((not _after and emphasis == 2) or emphasis > 2):
                                    node = start_node(Node(K.B, node, start=tok.end-emphasis))
                                    emphasis -= 2
                                else: # K.I
                                    while node is not emphasisnode: node = end_node(node, tok.end-emphasis)
                                    emphasis -= 1
                                    node = end_node(node, tok.end-emphasis)
                            else:
                                if (not _after and emphasis == 2) or emphasis > 2:
                                    node = start_node(Node(K.B, node, start=tok.end-emphasis))
                                    emphasis -= 2
                                elif (not _after and emphasis == 1) or emphasis > 1:
                                    node = start_node(Node(K.I, node, start=tok.end-emphasis))
                                    emphasis -= 1
                                else:
                                    if emphasis >= 2:
                                        add_text(node, tok.end-emphasis, tok.end-emphasis+2)
                                        emphasis -= 2
                                    else:
                                        add_text(node, tok.end-emphasis, tok.end-emphasis+1)
                                        emphasis -= 1
                                before = False # future emphasis not preceded my space anymore

                    case "double_inline_code":
                        if in_code and codenode.type == "double":
                            while node is not codenode: node = end_node(node, tok.start)
                            node = end_node(node, tok.end)
                        elif in_code and codenode.type == "single":
                            while node is not codenode: node = end_node(node, tok.start)
                            node = end_node(node, tok.end)
                            # HACK: assigning list to cls necessary because using '"block" in node.cls' elsewhere, so expects list
                            node = start_node(Node(K.CODE, node, start=tok.start + 1, type="single", cls=[])) 
                        else: node = start_node(Node(K.CODE, node, start=tok.start, type="double", cls=[]))

                    case "inline_code" if not (in_code and codenode.type == "double"):
                        if in_code and codenode.type == "single":
                            while node is not codenode: node = end_node(node, tok.start)
                            node = end_node(node, tok.end)
                        elif not in_code: node = start_node(Node(K.CODE, node, start=tok.start, type="single", cls=[]))
                    case _: add_text(node, tok.start, tok.end) # no valid match means it should be treated as text
            case _: add_text(node, tok.start, tok.end) # no valid match means it should be treated as text
    return head

def parse(text:str, head=None) -> Node: return treeify(tokenize(text), head)

def add_text(node:Node, start:int, end:int):
    """Merges with existing overlapping TEXT nodes if any or inserts a new text node. Does not fix existing adjacent text outside of start, end"""
    if start < end:
        overlaps = [(i, n) for (i, n) in enumerate(node.children) if n.k is K.TEXT and not (n.end < start or n.start > end)]
        start, end = min((start, *(n.start for _, n in overlaps))), max((end, *(n.end for _, n in overlaps)))
        for _, n in overlaps: node.children.remove(n)
        node.children.insert(min((len(node.children), *(i for i,_ in overlaps))), Node(K.TEXT, node, start=start, end=end))

def start_node(child:Node) -> Node: child.parent.children.append(child); return child

def end_node(node:Node, end:int) -> Node:
    node.end = end
    if not node.children and node.k not in [K.IMG]: # Invalid unless allowed to have no children
        # when replacing block node, don't replace with TEXT but with P>TEXT
        if node.parent and node.parent.k is K.BODY:
            if end-node.start > 1: node.k, node.data = K.P, {"start":node.start, "end":end}
            else: node.k, node.data = K.EMPTY_LINE, {"start":node.start, "end":end}
                # HACK if this is triggered from end of file, typeset needs children to be ["node...", ""] instead of just ["node..."] so xs is correct
            if end-node.start > 1:  node.children = [Node(K.TEXT, node, start=node.start, end=end-1), Node(K.TEXT, node, start=end-1, end=end)]
            else: node.children = [Node(K.TEXT, node, start=node.start, end=end)]
        else:
            # HACK: k = K.TEXT and then add_text replaces this with new text node while allowing merging
            node.k, node.data = K.TEXT, {"start": node.start, "end":end}
    if node.k in LINKS and node.href is None: # invalid link
        prev_end = node.start
        for i, child in enumerate(node.children): # replace formatting text with text nodes
            if child.start > prev_end: node.children.insert(i, Node(K.TEXT, node, start=prev_end, end=child.start))
            prev_end = child.end
        if node.children and child.end < node.end: node.children.append(Node(K.TEXT, node, start=child.end, end=node.end))
        # remove link node entirely
        steal_children(node, node.parent)
        node.parent.children.remove(node)
    return node.parent

def check(head:Node):
    for node in walk(head):
        if node.children:
            for child in node.children: assert node is child.parent, f"{node=}, {node.children=}, {child.parent=}"
        if node.parent: assert any(node is c for c in node.parent.children)
        if node.parent and node.parent.start and node.start: assert node.parent.start <= node.start
        if node.parent and node.parent.end and node.end: assert node.parent.end >= node.end

def steal_children(parent:Node, thief:Node):
    for c in parent.children:
        c.parent = thief
        thief.children.append(c)
    parent.children = []

def parents(node:Node):
    while node.parent: yield (node:=node.parent)

def walk(node, level=None, seen=None) -> Generator[Node, None, None]:
    assert id(node) not in (seen := set() if seen is None else seen) or seen.add(id(node))
    yield node if level is None else (node, level)
    for n in getattr(node, "children", []): yield from walk(n, None if level is None else level+1, seen)

def show(node): [print(f"{' '*SPACES*l}{n}") for n,l in walk(node, level=0)]

def find(node, **kwargs) -> Node: return next((n for n in walk(node) if all((getattr(n, k, None) == v for k,v in kwargs.items()))))
def find_all(node, **kwargs) -> List[Node]: return [n for n in walk(node) if all((getattr(n, k, None) == v for k,v in kwargs.items()))]

def serialize(head:Node, csspath:Path, basepath:Path) -> str:
    """node tree to html"""
    ret = f"""<!DOCTYPE html><html><head><link rel="stylesheet" href="{os.path.relpath(str(csspath.resolve()), str(basepath.parent.resolve()))}"/>
    <title>{head.title if head.title else "Unnamed frame"}</title></head>"""
    text = head.text
    assert text is not None
    for n in walk(head) :
        if n.k is K.TEXT and n.parent.k in [K.H1,K.H2,K.H3,K.H4,K.H5,K.H6]: n.parent.ids = [sluggify(text[n.start:n.end])]
    prevNode, prevLevel = None, -1
    for node, level in walk(head, level=0):
        if node.k not in [K.LINE, K.FRAME]:
            while prevLevel >= level:
                ret += htmltag(prevNode, text, open=False)
                assert prevNode.parent is not None
                prevNode, prevLevel = prevNode.parent, prevLevel - 1
            if node.k is K.TEXT: ret += text[node.start:node.end].replace("\n", "<br>")
            else:
                ret += htmltag(node, text, basepath=basepath)
                prevNode = node
                prevLevel = level
    while prevNode is not head:
        ret += htmltag(prevNode, text, open=False)
        prevNode = prevNode.parent
    return ret + "</html>"

def htmltag(node:Node, text, open=True, basepath:Path=None) -> str:
    ret = ""
    match node.k:
        case K.EMPTY_LINE: pass
        case K.A:
            if open and not (href:=text[node.href[0]:node.href[1]]).startswith(("http://", "https://")):
                hashidx = href.rfind("#")
                if hashidx == -1: path, heading = href, ""
                else: path, heading = href[:hashidx], href[hashidx+1:]
                href = basepath.parent.joinpath(path).resolve() if path != "" else basepath.resolve()
                if href.suffix == ".md": href = href.parent.joinpath(f"{href.stem}.html")
                href = os.path.relpath(href.as_posix(), basepath.parent.as_posix()) + (f"#{sluggify(heading)}" if heading else "")
            ret += f"<a href=\"{href}\"" if open else "</a>"
        case K.IMG: ret += f"<img src=\"{basepath.parent.joinpath(text[node.href[0]:node.href[1]]).resolve()}\" />" if open else ""
        case K.HR: ret += "<hr>" if open else ""
        case _: ret += f"<{''if open else '/'}{node.k.name.lower()}{'' if open else '>'}"
    return ret if ret.endswith(">") or not open or ret == "" else ret + f"""{' class="' + ' '.join(node.cls) + '"' if node.cls else ''}\
{' id="' + ' '.join(node.ids) + '"' if node.ids else ''}>"""

def sluggify(s:str) -> str: return re.sub(r'[^a-zA-Z0-9_]+', '-', s.lower().strip())

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
    r"(?P<dot>\.)",
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
    DOT = auto() # after identity, awaiting class name

def tokenizeCSS(text:str):
    head = CSSNode(CSS.STYLESHEET, None, [])
    node = head
    for m in CSS_PATTERN.finditer(text):
        for name, value in m.groupdict().items():
            if value is not None:
                match name:
                    case "identity":
                        if node.name is CSS.STYLESHEET: node = start_node(CSSNode(CSS.RULE, node, [], [value]))
                        elif node.name is CSS.RULE:
                            status = getattr(node, "status", None)
                            if status is Status.PSEUDOCLASS:
                                node.data = (*node.data[:-1], node.data[-1]+":" + value)
                                node.status = None
                            elif status is Status.OPENED: node = start_node(CSSNode(CSS.DECLARATION, node, [], (value,)))
                            elif status is Status.COMMA:
                                node.data += (value,)
                                node.status = None
                            elif status is Status.DOT: node.data[-1] += f".{value}"
                            else: raise NotImplementedError(node.name, status, value)
                        elif node.name is CSS.DECLARATION: node = start_node(CSSNode(CSS.IDENTITY, node, [], (value,)))
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
                    case "dot":
                        if node.name is CSS.RULE: node.status = Status.DOT
                        elif node.name is CSS.STYLESHEET: node = start_node(CSSNode(CSS.RULE, node, [], [""], status=Status.DOT))
                        else: raise NotImplementedError
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
                        if node.name is CSS.STYLESHEET and value == "*": node = start_node(CSSNode(CSS.RULE, node, [], (value,)))
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
    fonts = []
    font = {}
    selectors = None
    k = None
    for node in walk(head):
        if node.name is CSS.RULE:
            if font:
                fonts.append(font)
                font = {}
            selectors = node.data if isinstance(node.data, (tuple, list)) else (node.data,)
        elif node.name is CSS.DECLARATION:
            if len(node.data) == 1: k = node.data[0] # value is in either function or identity children
            else:
                if selectors == ("@font-face",):
                    font[node.data[0]] = node.data[1]
                    continue
                # make line-height unit explicit if not given
                if node.data[0] == "line-height" and isinstance(node.data[1], (float, int)): node.data= ("line-height", (node.data[1],"em",)) 
                k, v = node.data[0], node.data[1:]
                k, v = expand_css_shorthand(k, v)
                for k0, v0 in zip(k, v):
                    v0 = v0[0] if isinstance(v0, tuple) and len(v0) == 1 else v0
                    v0 = color_from_hex(int(v0[1:], base=16)) if isinstance(v0, str) and v0.startswith("#") else v0
                    for s in selectors: css_dict.setdefault(selector_node(s), {})[k0] = v0
        elif node.name in (CSS.IDENTITY, CSS.FUNCTION):
            if node.name is CSS.IDENTITY:
                k, _ = expand_css_shorthand(k, node.data[0])
                for s in selectors:
                    for k0 in k:
                        css_dict.setdefault(selector_node(s), {})[k0] = node.data[0]
            else:
                assert len(node.data) == 2
                f, arg = node.data
                if font:
                    assert k == "src"
                    font[f] = arg
                else:
                    for s in selectors: css_dict.setdefault(selector_node(s), {}).setdefault(k, {})[f] = arg
    if font: fonts.append(font)
    return css_dict, fonts

@dataclass(frozen=True)
class Selector:
    k:K
    cls:List[str] = None
    ids:List[str] = None
    pseudocls:List[str] = None

def selector_node(selector:str) -> Selector:
    tag = m.group() if (m:=re.match(r"[a-zA-Z]+[_a-zA-Z0-9\-]*", selector)) else "any"
    cls = tuple(re.findall(r"\.(\-?[_a-zA-Z]+[_a-zA-Z0-9\-]*)", selector))
    ids = tuple(re.findall(r"#(\-?[_a-zA-Z]+[_a-zA-Z0-9\-]*)", selector))
    pseudocls = tuple(re.findall(r":([a-zA-Z]+[_a-zA-Z0-9\-\(\)]*)", selector))
    return Selector(K[tag.upper()], cls if cls else None, ids if ids else None, pseudocls if pseudocls else None)

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