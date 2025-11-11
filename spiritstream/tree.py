from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Iterator, Any, Iterable
from spiritstream.helpers import SPACES
from pathlib import Path
import re, os, time, functools

PATTERN = re.compile("|".join([
    r"^(?P<codeblock>```[\s\S]*?\n```\n|~~~[\s\S]*?\n~~~\n)",
    r"^ {0,3}(?P<horizontal_rule>\*(?:[ \t]*\*){2,}[ \t]*$|_(?:[ \t]*_){2,}[ \t]*$|-(?:[ \t]*-){2,}[ \t]*$)",
    r"^(?P<listitem>[ \t]*(?:\d+[\.\)][ \t]+|[-*+][ \t]+))",
    r"^(?P<indented_line>(?: {0,3}\t| {4,}).*)",
    r"^ {0,3}(?P<heading>#+)[\t ]+",
    r"^ {0,3}(?P<blockquote_line>>+)",
    r"(?P<new_line>\n)",
    r"^(?P<toc>(?<!\\)\[TOC\]$)",
    r"^ {0,3}(?P<paragraph_line>)(?=[^\n])",

    r"(?P<emphasis>(?<!\\)\*+)",

    r"(?P<wikilink_embed_start>(?<!\\)!\[\[(?=[^\[\]]))",
    r"(?P<wikilink_start>(?<!\\)\[\[(?=[^\[\]]))", # prevent wikilinks like [[]link]], but guarantees that the link is not empty
    r"(?P<markdownlink_embed_start>(?<!\\)!\[(?=[^\[\]]))",
    r"(?P<markdownlink_start>(?<!\\)\[(?=[^\]]))",
    r"(?P<markdownlink_switch>(?<!\\)\]\()",
    r"(?P<wikilink_end>(?<![\\\]])\]\])",

    r"(?P<closing_parenthesis>(?<!\\)\))",

    r"(?P<double_inline_code>(?<![`\\])``)",
    r"(?P<inline_code>(?<![\\])`)", # no negative lookbehind for another ` because that would already have matched with double_inline_code
]), re.MULTILINE)

@dataclass(frozen=True)
class Token:
    name:str
    text:str
    data:Any = None
    def __repr__(self): return f"Token({self.name}, {repr(self.text)}, {self.data})"


class K(Enum): # Kind of node
    SS = auto() # top of the tree
    SCENE = auto()
    FRAME = auto()
    LINE = auto()

    ANY = auto() # used for CSS selectors

    BODY = auto()
    TOC = auto()
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
    def __init__(self, k:K, parent:"Node", children:list["Node"]=None, data:dict=None, **kwargs):
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

def tokenize(text:str) -> Iterator[Token]:
    i = 0
    for m in PATTERN.finditer(text):
        if m.start() > i: yield Token("text", text[i:m.start()])
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
                yield Token(name, m.group(), data)
                break
    if i < len(text): yield Token("text", text[i:])

def treeify(tokens:Iterable[Token], head=None) -> Node:
    node = head = Node(K.BODY, None) if head is None else head # node stores what I am currently adding to aka. most recently unclosed
    for tok in tokens:
        in_code = any((n.k is K.CODE for n in (node, *parents(node))))
        in_link = any((n.k in LINKS for n in (node, *parents(node)))) and not in_code
        linknode = next((n for n in (node, *parents(node)) if n.k in LINKS)) if in_link else None
        match tok.name:
            case "new_line":
                if node is head: node = start_node(Node(K.EMPTY_LINE, head))
                add_text(node, tok.text)
                while node is not head: node = end_node(node)
            case "heading":
                node = start_node(Node(getattr(K, f"H{min(tok.data, 6)}"), head))
                add_text(node, tok.text, formatting=True)
            case "paragraph_line":
                if node is head and (not node.children or node.children[-1].k is not K.P): node = start_node(Node(K.P, head))
                else: node = node.children[-1]
                if tok.text != "": add_text(node, tok.text) # add any indent captured in the paragraph line token
            case "horizontal_rule":
                node = start_node(Node(K.HR, node))
                add_text(node, tok.text, formatting=True)
            case "codeblock": 
                node = start_node(Node(K.CODE, head, language=tok.data[0], cls=["block"]))
                codeblock = node
                node = start_node(Node(K.P, node, cls=["code_language"]))
                add_text(node, tok.text[:3], formatting=True) # ```
                add_text(node, tok.text[3:tok.data[1]+1]) # \n
                add_text(codeblock, tok.text[tok.data[1]+1:-4])
                node = start_node(Node(K.P, codeblock))
                add_text(node, tok.text[-4:-1], formatting=True)# ```
                add_text(node, tok.text[-1:]) # \n
                node = head # prevent merging with following blocks
            case "indented_line" if tok.data >= 4:
                while node.children: node = node.children[-1] # get latest node
                node = next((n for n in (node, *parents(node)) if n.k is K.CODE and "block" in n.cls), None)
                if node is None: node = start_node(Node(K.CODE, head, cls=["block"]))
                add_text(node, tok.text[:tok.data], formatting=True)
                add_text(node, tok.text[tok.data:])
            case "blockquote_line":
                while node.children: node = node.children[-1] # get latest node
                node = next((n for n in (node, *parents(node)) if n.k is K.BLOCKQUOTE), None)
                if not node: node = start_node(Node(K.BLOCKQUOTE, head, depth = 1))
                while node.depth < tok.data: node = start_node(Node(K.BLOCKQUOTE, node, depth = node.depth + 1))
                while node.k is K.BLOCKQUOTE and node.depth > tok.data: node = end_node(node)
                add_text(node, tok.text, formatting=True)
            case "listitem":
                newindent, newlisttype, digit = tok.data
                while node.children: node = node.children[-1] # get latest node
                listnode = next((n for n in (node, *parents(node)) if n.k in (K.UL, K.OL)), None)
                if listnode is None:
                    if newindent >= 4: # not a list item but an indented line
                        if not (node.k is K.CODE and "block" in node.cls): node = start_node(Node(K.CODE, head, cls=["block"]))
                        add_text(node, tok.text[:4], formatting=True)
                        add_text(node, tok.text[4:])
                    else:
                        node = start_node(Node(newlisttype, head, indent=newindent))
                        node = start_node(Node(K.LI, node, digit = digit))
                        add_text(node, tok.text, formatting=True)
                else:
                    if listnode.indent < newindent:
                        if listnode.children: node = next((n for n in (node, *parents(node)) if n.k is K.LI))
                        else: node = start_node(Node(K.LI, listnode))
                        node = listnode = start_node(Node(newlisttype, node, indent = newindent))
                    node = listnode
                    while listnode.indent > newindent:
                        node = end_node(listnode)
                        assert node.k is K.LI
                        listnode = node = end_node(node)
                    if listnode.k is not newlisttype:
                        node = end_node(node)
                        node = listnode = start_node(Node(newlisttype, node, indent = newindent))
                    node = start_node(Node(K.LI, node, digit = digit))
                    add_text(node, tok.text, formatting=True)
            
            case "toc":
                node = start_node(Node(K.TOC, head))
                add_text(node, tok.text, formatting=True)
            
            case "wikilink_embed_start" if not in_link:
                node = start_node(Node(K.IMG, node, linktype = "wiki"))
                add_text(node, tok.text, formatting=True)
            case "wikilink_start" if not in_link:
                node = start_node(Node(K.A, node, linktype = "wiki"))
                add_text(node, tok.text, formatting=True)
            case "wikilink_end" if in_link and linknode.linktype == "wiki":
                if len(linknode.children) > 1: linknode.href = "".join((rc.text for c in linknode.children[1:] for rc in walk(c) if rc.k is K.TEXT))
                else: linknode.href = ""
                if linknode.k is K.IMG: linknode.children = [Node(K.TEXT, linknode, text=linknode.children[0].text+linknode.href, cls=["formatting"])]
                else: linknode.children[1:] = [Node(K.TEXT, linknode, text=linknode.href, cls=[])]
                add_text(linknode, tok.text, formatting=True)
                node = end_node(linknode)
            case "markdownlink_embed_start" if not in_link:
                node = start_node(Node(K.IMG, node, linktype = "md"))
                add_text(node, tok.text, formatting=True)
            case "markdownlink_start" if not in_link:
                node = start_node(Node(K.A, node, linktype = "md"))
                add_text(node, tok.text, formatting=True)
            case "markdownlink_switch" if in_link and linknode.linktype == "md":
                while node is not linknode: node = end_node(node)
                add_text(node, tok.text, formatting=True)
                node.linktype = "md switched"
        
            # case "opening_parenthesis" if in_link and linknode.linktype == "md switch": linknode.linktype, linknode.switchidx = "md switched", len(linknode.children)
            case "closing_parenthesis" if in_link and linknode.linktype == "md switched":
                switch_idx = next((len(linknode.children)-1-i for i,n in enumerate(reversed(linknode.children)) if "formatting" in n.cls))
                linknode.href = "".join((rc.text for c in linknode.children[switch_idx+1:] for rc in walk(c) if rc.k is K.TEXT))
                linknode.children = linknode.children[:switch_idx+1] # remove children that were part of the link
                if linknode.k is K.IMG:
                    linknode.alt = "".join((rc.text for c in linknode.children[1:switch_idx] for rc in walk(c) if rc.k is K.TEXT))
                    linknode.children[:switch_idx+1] = [
                        Node(K.TEXT, linknode, text=linknode.children[0].text+linknode.alt+linknode.children[switch_idx].text, cls=["formatting"])]
                add_text(linknode, linknode.href+tok.text, formatting=True)
                node = end_node(linknode)

            case _ if not in_link or node.linktype == "md": # inline formatting only allowed outside of links except in the label part of a markdown link                
                in_italic, in_bold = map(lambda k: any((n.k is k for n in (node, *parents(node)))), (K.I, K.B))
                if in_italic: italicnode = next((n for n in (node, *parents(node)) if n.k is K.I))
                if in_bold: boldnode = next((n for n in (node, *parents(node)) if n.k is K.B))
                if in_code: codenode = next((n for n in (node, *parents(node)) if n.k is K.CODE))

                match tok.name:
                    case "emphasis" if not in_code:
                        before, _after = tok.data # whether space before / after. after is recalculated based on how far along I am in the emphasis
                        emphasis = len(tok.text)
                        while emphasis > 0:
                            emphasisnode = next((n for n in (node, *parents(node)) if n.k in [K.I, K.B]), None)
                            if emphasisnode and not before:
                                if emphasisnode.k is K.B:
                                    if emphasis >= 2:
                                        while node is not emphasisnode: node = end_node(node)
                                        emphasis -= 2
                                        add_text(node, tok.text[-emphasis-2:(-emphasis if -emphasis<0 else len(tok.text))], formatting=True)
                                        node = end_node(node)
                                    else:
                                        if italicnode:=next((n for n in (node, *parents(node)) if n.k is K.I), None):
                                            while node is not italicnode: node = end_node(node)
                                            emphasis -= 1
                                            add_text(node, tok.text[-emphasis-1:(-emphasis if -emphasis<0 else len(tok.text))], formatting=True)
                                            node = end_node(node)
                                            node = start_node(Node(K.B, node)) # continue overlapping bold node
                                        else:
                                            if not _after or emphasis > 1: # can start nodes
                                                node = start_node(Node(K.I, node))
                                                emphasis -= 1
                                                add_text(node, tok.text[-emphasis-1:(-emphasis if -emphasis<0 else len(tok.text))], formatting=True)
                                            else:
                                                emphasis -= 1
                                                add_text(node, tok.text[-emphasis-1:(-emphasis if -emphasis < 0 else len(tok.text))])
                                elif emphasis >= 2 and (boldnode:=next((n for n in (node, *parents(node)) if n.k is K.B), None)):
                                    while node is not boldnode: node = end_node(node)
                                    emphasis -= 2
                                    add_text(node, tok.text[-emphasis-2:(-emphasis if -emphasis<0 else len(tok.text))], formatting=True)
                                    node = end_node(node)
                                    node = start_node(Node(K.I, node)) # continue overlapping italic node
                                elif emphasis >= 2 and not boldnode and ((not _after and emphasis == 2) or emphasis > 2):
                                    node = start_node(Node(K.B, node))
                                    emphasis -= 2
                                    add_text(node, tok.text[-emphasis-2:(-emphasis if -emphasis<0 else len(tok.text))], formatting=True)
                                else: # K.I
                                    while node is not emphasisnode: node = end_node(node)
                                    emphasis -= 1
                                    add_text(node, tok.text[-emphasis-1:(-emphasis if -emphasis<0 else len(tok.text))], formatting=True)
                                    node = end_node(node)
                            else:
                                if (not _after and emphasis == 2) or emphasis > 2:
                                    node = start_node(Node(K.B, node))
                                    emphasis -= 2
                                    add_text(node, tok.text[-emphasis-2:(-emphasis if -emphasis<0 else len(tok.text))], formatting=True)
                                elif (not _after and emphasis == 1) or emphasis > 1:
                                    node = start_node(Node(K.I, node))
                                    emphasis -= 1
                                    add_text(node, tok.text[-emphasis-1:(-emphasis if -emphasis<0 else len(tok.text))], formatting=True)
                                else:
                                    if emphasis >= 2:
                                        emphasis -= 2
                                        add_text(node, tok.text[-emphasis-2:(-emphasis if -emphasis<0 else len(tok.text))])
                                    else:
                                        emphasis -= 1
                                        add_text(node, tok.text[-emphasis-1:(-emphasis if -emphasis<0 else len(tok.text))])
                                before = False # future emphasis not preceded my space anymore

                    case "double_inline_code":
                        if in_code and codenode.type == "double":
                            while node is not codenode: node = end_node(node)
                            add_text(node, tok.text, formatting=True)
                            node = end_node(node)
                        elif in_code and codenode.type == "single":
                            while node is not codenode: node = end_node(node)
                            add_text(node, tok.text[:1], formatting=True)
                            node = end_node(node)
                            # HACK: assigning list to cls necessary because using '"block" in node.cls' elsewhere, so expects list
                            node = start_node(Node(K.CODE, node, type="single", cls=[])) 
                            add_text(node, tok.text[1:], formatting=True)
                        else:
                            node = start_node(Node(K.CODE, node, type="double", cls=[]))
                            add_text(node, tok.text, formatting=True)

                    case "inline_code" if not (in_code and codenode.type == "double"):
                        if in_code and codenode.type == "single":
                            while node is not codenode: node = end_node(node)
                            add_text(node, tok.text, formatting=True)
                            node = end_node(node)
                        elif not in_code:
                            node = start_node(Node(K.CODE, node, type="single", cls=[]))
                            add_text(node, tok.text, formatting=True)
                    case _: add_text(node, tok.text) # no valid match means it should be treated as text
            case _: add_text(node, tok.text) # no valid match means it should be treated as text
    # true if file is empty or the file ends with a newline. adds an empty line so that there is a cursor position.
    if node is head: head.children.append(Node(K.EMPTY_LINE, head, [Node(K.TEXT, None, text="", cls=[])]))
    while node is not head: node = end_node(node)
    return head

def parse(text:str, head=None) -> Node: return treeify(tokenize(text), head)

def add_text(node, text:str, formatting=False):
    if node.children and node.children[-1].k is K.TEXT and (isinstance(node.children[-1].cls, list) and "formatting" in node.children[-1].cls) == formatting:
        node.children[-1].text += text
    else: node.children.append(Node(K.TEXT, node, text=text, cls=["formatting"] if formatting else []))

def start_node(child:Node) -> Node: child.parent.children.append(child); return child

def end_node(node:Node) -> Node:
    if not node.children and node.k not in [K.IMG]: node.parent.children.remove(node) # Remove nodes that can't have no children
    elif "".join((n.text for n in walk(node) if n.k is K.TEXT and "formatting" not in n.cls)) in "\n" and not any((n for n in walk(node) if n.k is K.IMG)) and node.k not in [K.IMG, K.TOC, K.EMPTY_LINE, K.HR] and not (node.k is K.CODE and "block" in node.cls): # or only have formatting text
        t = "".join((n.text for n in walk(node) if n.k is K.TEXT))
        if node.parent.k is K.BODY: node.k, node.children = K.P, [Node(K.TEXT, node, text=t, cls=[])]
        else: node.k, node.children, node.text, node.cls = K.TEXT, [], t, []
    elif node.k in LINKS and node.href is None: # invalid link
        # merge text nodes
        if node.children[0].k is K.TEXT and "formatting" in node.children[0].cls: node.children[0].cls.remove("formatting")
        for c0, c1 in zip(node.children.copy(), node.children[1:].copy()):
            if c1.k is K.TEXT and "formatting" in c1.cls: c1.cls.remove("formatting")
            if c0.k is c1.k is K.TEXT:
                c1.text = c0.text + c1.text
                node.children.remove(c0)
        # remove link node entirely
        steal_children(node, node.parent)
        node.parent.children.remove(node)
    return node.parent

def check(head:Node) -> str:
    ret = ""
    for node in walk(head):
        if node.children:
            for child in node.children: assert node is child.parent, f"{node=}, {node.children=}, {child.parent=}"
        if node.parent: assert any(node is c for c in node.parent.children)
        if node.k is K.TEXT: ret += node.text
    return ret

def steal_children(parent:Node, thief:Node):
    for c in parent.children:
        c.parent = thief
        thief.children.append(c)
    parent.children = []

def parents(node:Node):
    while node.parent: yield (node:=node.parent)

def blockparent(node:Node): return next((p for p in (node, *parents(node)) if p.k not in INLINE_NODES))

def walk(node, level=None, seen=None, reverse=False, siblings=False) -> Iterator[Node | tuple[Node, int]]:
    if siblings:
        while (p:=node.parent):
            if node is (p.children[0] if reverse else p.children[-1]): node, level = p, None if level is None else level-1
            else: yield from walk(node:=p.children[p.children.index(node) + (-1 if reverse else 1)], level, seen, reverse)
    else:
        assert id(node) not in ((seen := set()) if seen is None else seen)
        seen.add(id(node))
        yield node if level is None else (node, level)
        children = getattr(node, "children", [])
        for n in reversed(children) if reverse else children: yield from walk(n, None if level is None else level+1, seen, reverse=reverse)

def show(node): [print(f"{' '*SPACES*l}{n}") for n,l in walk(node, level=0)]

def find(node, **kwargs) -> Node: return next((n for n in walk(node) if all((getattr(n, k, None) == v for k,v in kwargs.items()))))
def find_all(node, **kwargs) -> list[Node]: return [n for n in walk(node) if all((getattr(n, k, None) == v for k,v in kwargs.items()))]

def serialize(head:Node, csspath:Path, basepath:Path) -> str:
    """node tree to html"""
    ret = f"""<!DOCTYPE html><html><head><link rel="stylesheet" href="{os.path.relpath(str(csspath.resolve()), str(basepath.parent.resolve()))}"/>
    <title>{head.title if head.title else "Unnamed frame"}</title></head>"""
    for n in walk(head) :
        if n.k in [K.H1,K.H2,K.H3,K.H4,K.H5,K.H6]:
            n.ids = [sluggify("".join((c.text for c in walk(n) if c.k is K.TEXT and "formatting" not in c.cls)))]
    prevNode, prevLevel = None, -1
    for node, level in walk(head, level=0):
        if node.k not in [K.LINE, K.FRAME]:
            while prevLevel >= level:
                ret += htmltag(prevNode, open=False)
                assert prevNode.parent is not None
                prevNode, prevLevel = prevNode.parent, prevLevel - 1
            if node.k is K.TEXT:
                if "formatting" not in node.cls:
                    if node is node.parent.children[-1] and node.parent.k not in (*INLINE_NODES, K.EMPTY_LINE, K.HR): ret += node.text.rstrip("\n").replace("\n", "<br>")
                    else: ret += node.text.replace("\n", "<br>")
            else:
                ret += htmltag(node, basepath=basepath)
                prevNode = node
                prevLevel = level
    while prevNode is not head:
        ret += htmltag(prevNode, open=False)
        prevNode = prevNode.parent
    return ret + "</html>"

def htmltag(node:Node, open=True, basepath:Path=None) -> str:
    ret = ""
    match node.k:
        case K.EMPTY_LINE: pass
        case K.A:
            href = node.href
            if open and not href.startswith(("http://", "https://")):
                hashidx = href.rfind("#")
                if hashidx == -1: path, heading = href, ""
                else: path, heading = href[:hashidx], href[hashidx+1:]
                href = basepath.parent.joinpath(path).resolve() if path != "" else basepath.resolve()
                if href.suffix == ".md": href = href.parent.joinpath(f"{href.stem}.html")
                href = os.path.relpath(href.as_posix(), basepath.parent.as_posix()) + (f"#{sluggify(heading)}" if heading else "")
            ret += f"<a href=\"{href}\"" if open else "</a>"
        case K.IMG: ret += f"<img src=\"{basepath.parent.joinpath(node.href).resolve()}\" />" if open else ""
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
    children:list[Node] = None
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

def css_to_dict(head:CSSNode) -> dict:
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
    cls:list[str] = None
    ids:list[str] = None
    pseudocls:list[str] = None

def selector_node(selector:str) -> Selector:
    tag = m.group() if (m:=re.match(r"[a-zA-Z]+[_a-zA-Z0-9\-]*", selector)) else "any"
    cls = tuple(re.findall(r"\.(\-?[_a-zA-Z]+[_a-zA-Z0-9\-]*)", selector))
    ids = tuple(re.findall(r"#(\-?[_a-zA-Z]+[_a-zA-Z0-9\-]*)", selector))
    pseudocls = tuple(re.findall(r":([a-zA-Z]+[_a-zA-Z0-9\-\(\)]*)", selector))
    return Selector(K[tag.upper()], cls if cls else None, ids if ids else None, pseudocls if pseudocls else None)

# NOTE: margin: 0, auto not supported because mix of identity and other
# NOTE: margin: calc(100% - 5px) not supported

def expand_css_shorthand(k, v) -> tuple[tuple, tuple]:
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

def parseCSS(text:str) -> dict: return css_to_dict(tokenizeCSS(text))