from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Union, Generator, Any, Iterable, Dict
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
    r"(?P<bolditalic_start>(?<!\\)\*\*\*(?![\*\s])|(?<!\\)___(?![_\s]))",
    r"(?P<bolditalic_end>(?<![\*\s\\])\*\*\*|(?<![_\s\\])___)",

    r"(?P<bold_toggle>(?<![\*\s\\])\*\*(?![\*\s])|(?<![_\s\\])__(?![_\s]))", # don't know whether it's start or end
    r"(?P<bold_start>(?<!\\)\*\*(?![\*\s])|(?<!\\)__(?![_\s]))",
    r"(?P<bold_end>(?<![\*\s\\])\*\*|(?<![_\s\\])__)",

    r"(?P<italic_toggle>(?<![\*\s\\])\*(?![\*\s])|(?<![_\s\\])_(?![_\s]))", # don't know whether it's start or end
    r"(?P<italic_start>(?<!\\)\*(?![\*\s])|(?<!\\)_(?![_\s]))",
    r"(?P<italic_end>(?<![\*\s\\])\*|(?<![_\s\\])_)",

    r"(?P<strikethrough_toggle>(?<![~\s\\])~~(?![~\s]))", # don't know whether it's start or end
    r"(?P<strikethrough_start>(?<!\\)~~(?![~\s]))",
    r"(?P<strikethrough_end>(?<![~\s\\])~~)",

    r"(?P<wikilink_embed_start>(?<!\\)!\[\[(?!\[))",
    r"(?P<wikilink_start>(?<!\\)\[\[(?!\[))",
    r"(?P<markdownlink_embed_start>(?<!\\)!\[(?!\[))",
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
INLINE_NODES = ["bolditalic", "bold", "italic", "strikethrough", "link", "embed", "double_inline_code", "inline_code"]
# BLOCK_NODES = ["heading", "blockquote", "paragraph", "horizontal_rule", "codeblock", "empty_line"] 

@dataclass(frozen=True)
class Token:
    name:str
    idx:int
    data:Any = None
    def __repr__(self): return f"Token({self.name}, {self.idx}, {self.data})"

@dataclass
class Node:
    name:Union["CSS", str]
    parent:"Node"
    children:Optional[List["Node"]] = None
    data:Any = None

    def __repr__(self): return  f"\033[32m{self.__class__.__name__}\033[0m({self.name}, \033[94mdata=\033[0m{self.data})"


@dataclass
class HTMLNode(Node):
    start:int = None
    end:int = None
    
    def __repr__(self): return  f"\033[32m{self.__class__.__name__}\033[0m({self.name}, \033[94mstart=\033[0m{self.start}, \033[94mend=\033[0m{self.end}, \033[94mdata=\033[0m{self.data})"

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
                        if stripped[0].isnumeric(): listtype = "ordered"
                        elif stripped.startswith(("+", "*", "-")): listtype = "unordered"
                        else: raise AssertionError(f"'{stripped}' invalid list type")
                        digit = int(t.strip("\t .)")) if listtype == "ordered" else None
                        data = (indent, listtype, digit)
                    case "blockquote_line": data = m.group("blockquote_line").count(">")
                    case "text" if text_start is None: text_start = m.start()
                    case _: data = None
                if name != "text":
                    if text_start is not None: yield Token("text", text_start, m.start()); text_start = None
                    yield Token(name, m.start(), data)
    if text_start is not None: yield Token("text", text_start, m.start())
    yield Token("endoffile", len(text))

def treeify(tokens:Iterable[Token], head=None) -> Node:
    node = head = HTMLNode("body", None, []) if head is None else head # node I am currently adding to aka. most recent unclosed node
    for tok in tokens:
        while tok.name in BLOCK_TOKENS and node.name in INLINE_NODES:
            # if an inline node ends a line, it will have length even if empty because it includes its formatting and a linebreak.
            # account for this before deleting empty nodes
            minlength = node_format_length(node) + (1 if tok.name != "endoffile" else 0) # no linebreak to account for at end of file
            if tok.idx - node.start > minlength and node.name not in ["link", "embed"]: node = end_node(node, tok.idx, offset=False) # unclodes inline node with content. truncate to end of line
            else:
                del node.parent.children[-1] # unclosed and empty inline node
                node = node.parent
        inlink = node.name in ["link", "embed"]
        match tok.name:
            case "heading": node = start_node(head, HTMLNode(tok.name, head, [], tok.data, tok.idx))
            case "empty_line": node = start_node(head, HTMLNode(tok.name, head, [], tok.data, tok.idx))
            case "paragraph_line" if node.name != "paragraph": node = start_node(head, HTMLNode("paragraph", head, [], tok.data, tok.idx))
            case "horizontal_rule": node = start_node(head, HTMLNode(tok.name, head, [], tok.data, tok.idx))
            case "codeblock": node = start_node(head, HTMLNode(tok.name, head, [], tok.data, tok.idx))
            case "indented_line" if tok.data[0] >= 4:
                if node.name != "codeblock": node = start_node(head, HTMLNode("codeblock", head, [], (None, [" "*(tok.data[0]-4) + tok.data[1]]), tok.idx))
                else: node.data[1].append(" "*(tok.data[0]-4) + tok.data[1])
            case "blockquote_line":
                if node.name != "blockquote": node = start_node(head, HTMLNode("blockquote", head, [], 1, tok.idx))
                while node.data < tok.data: node = start_node(node, HTMLNode("blockquote", node, [], node.data+1, tok.idx))
                while node.name == "blockquote" and node.data > tok.data: node = end_node(node, tok.idx)
            case "listitem":
                newindent = tok.data[0] // SPACES
                newlisttype = tok.data[1]
                digit = tok.data[2]
                listnode = node if node.name == "list" else node.parent if node.name == "listitem" and node.parent and node.parent.name == "list" else None
                if listnode is None: node = listnode = start_node(head, HTMLNode("list", head, [], (0, ""), tok.idx))
                while (previndent:=listnode.data[0]) < newindent:
                    if node.name != "listitem": node = start_node(listnode, HTMLNode("listitem", listnode, [], None, tok.idx))
                    node = listnode = start_node(node, HTMLNode("list", node, [], (previndent+1, ""), tok.idx))
                if node is not listnode: node = end_node(node, tok.idx)
                assert node is listnode
                while (previndent:=listnode.data[0]) > newindent:
                    node = end_node(listnode, tok.idx)
                    assert node.name == "listitem"
                    listnode = node = end_node(node, tok.idx)
                if listnode.data[1] != newlisttype:
                    if listnode.data[1] == "": listnode.data = (listnode.data[0], newlisttype)
                    else:
                        node = end_node(node, tok.idx)
                        node = listnode = start_node(node, HTMLNode("list", node, [], (newindent, newlisttype), tok.idx))
                assert node is listnode and node.data[1] == tok.data[1]
                node = start_node(node, HTMLNode(tok.name, node, [], digit, tok.idx))
            
            case "wikilink_embed_start" if not inlink: node = start_node(node, HTMLNode("embed", node, [], "wiki", tok.idx))
            case "wikilink_start" if not inlink: node = start_node(node, HTMLNode("link", node, [], "wiki", tok.idx))
            case "wikilink_end" if inlink:
                if node.data == "wiki": node = end_node(node, tok.idx)
                else: node.data = "md switch" # this token can end markdownlinks too. data is used when parsing next token to see if conditions for a full switch are satisfied
            case "markdownlink_embed_start" if not inlink: node = start_node(node, HTMLNode("embed", node, [], "md", tok.idx))
            case "markdownlink_start" if not inlink: node = start_node(node, HTMLNode("link", node, [], "md", tok.idx))
            case "markdownlink_switch" if inlink: 
                if node.data == "wiki": # can switch wikilinks too in cases like [[link](link) the first [ is ignored, so start is shifted
                    node.start += 1 if node.name == "link" else 2 # if "embed", there is a "!" that is being ignored too
                    node.name = "link"
                node.data = "md switched" 
        
            case "opening_parenthesis" if inlink and node.data == "md switch": node.data = "md switched"
            case "closing_parenthesis" if inlink and node.data == "md switched": node = end_node(node, tok.idx)

            case "text" if not (node.name in ["empty_line", "codeblock"] and tok.data - tok.idx == 1): # text will match newline at end of empty_line and codeblock which is useless
                if node.children and (n:=node.children[-1]).name == "text" and n.end == tok.idx: n.end = tok.data # merge with previous text if contiguous
                else: node.children.append(HTMLNode("text", node, [], None, tok.idx, tok.data))

            case _ if not inlink or node.data == "md": # inline formatting only allowed outside of links except in the label part of a markdown link                
                match tok.name:
                    case "italic_start":
                        if node.name != "bolditalic": node = start_node(node, HTMLNode("italic", node, [], None, tok.idx))
                    case "italic_toggle"|"italic_end":
                        if node.name == "italic": node = end_node(node, tok.idx)
                        elif node.name == "bolditalic": # replace bolditalic with opening bold and opening italic. Start of opening italic is offset accordingly. Italic ends here
                            node.name = "bold"
                            node = end_node(start_node(node, HTMLNode("italic", node, [], None, node.start+2)), tok.idx)
                        elif tok.name == "italic_toggle": node = start_node(node, HTMLNode("italic", node, [], None, tok.idx))

                    case "bold_start":
                        if node.name != "bolditalic": node = start_node(node, HTMLNode("bold", node, [], None, tok.idx))
                    case "bold_toggle"|"bold_end":
                        if node.name == "bold": node = end_node(node, tok.idx)
                        elif node.name == "bolditalic": # replace bolditalic with opening italic and opening bold. Start of opening bold is offset accordingly. Bold ends here
                            node.name = "italic"
                            node = end_node(start_node(node, HTMLNode("bold", node, [], None, node.start+1)), tok.idx)
                        elif tok.name == "bold_toggle": node = start_node(node, HTMLNode("bold", node, [], None, tok.idx))
                    
                    case "bolditalic_start" if node.name not in ["italic", "bold"]: node = start_node(node, HTMLNode("bolditalic", node, [], None, tok.idx))
                    case "bolditalic_toggle"|"bolditalic_end":
                        if node.name == "bolditalic": node = end_node(node, tok.idx)
                        elif node.name == "italic":
                            node = end_node(node, tok.idx)
                            if node.name == "bold": end_node(node, tok.idx+1)
                            else: node = start_node(node, HTMLNode("bold", node, [], None, tok.idx+1))
                        elif node.name == "bold":
                            node = end_node(node, tok.idx)
                            if node.name == "italic": node = end_node(node, tok.idx+2)
                            else: node = start_node(node, HTMLNode("italic", node, [], None, tok.idx+2))
                        elif tok.name == "bolditalic_toggle": node = start_node(node, HTMLNode("bolditalic", node, [], None, tok.idx))

                    case "strikethrough_start": node = start_node(node, HTMLNode("strikethrough", node, [], None, tok.idx))
                    case "strikethrough_toggle"|"strikethrough_end":
                        if node.name == "strikethrough": node = end_node(node, tok.idx)
                        elif tok.name == "strikethrough_toggle": node = start_node(node, HTMLNode("strikethrough", node, [], None, tok.idx))

                    case "double_inline_code":
                        if node.name == "double_inline_code": node = end_node(node, tok.idx)
                        elif node.name == "inline_code":
                            node = end_node(node, tok.idx)
                            node = start_node(node, HTMLNode("inline_code", node, [], None, tok.idx + 1))
                        else: node = start_node(node, HTMLNode(tok.name, node, [], None, tok.idx))

                    case "inline_code":
                        if node.name == "inline_code": node = end_node(node, tok.idx)
                        elif node.name != "double_inline_code": node = start_node(node, HTMLNode(tok.name, node, [], None, tok.idx))
    return head

def parse(text:str, head=None) -> Node: return treeify(tokenize(text), head)

def start_node(parent:Node, child:Node) -> Node:
    assert child.parent is parent
    parent.children.append(child)
    return child

def end_node(node:HTMLNode, idx, offset=True) -> HTMLNode:
    assert node.parent != None, node
    end = idx + node_format_length(node) if offset else idx
    node.end = end
    if node.name == "bolditalic": # replace bolditalic with bold and italic nodes, close both
        node.name = "bold"
        node = end_node(start_node(node, HTMLNode("italic", node, [], None, node.start+2)), idx, offset=offset)
    return node.parent

def node_format_length(node:Node) -> int:
    # NOTE: this manages shifting the end of the nodes such that each node includes its formatting. It isn't done in the tokenization because tokens like italic_toggle are ambiguous
    match node.name:
        case "heading": return node.data + 1 # + 1 for space between # and heading
        case "bold": return 2
        case "italic": return 1
        case "strikethrough": return 2
        case "double_inline_code": return 2
        case "inline_code": return 1
        case "bolditalic": return 3
        case "link": return 2 if node.data == "wiki" else 1
        case "embed": return 2 if node.data == "wiki" else 1
        case _: return 0

def walk(node, level=None, seen=None) -> Generator[Node, None, None]:
    assert id(node) not in (seen := set() if seen is None else seen) or seen.add(id(node))
    yield node if level is None else (node, level)
    for n in getattr(node, "children", []): yield from walk(n, None if level is None else level+1, seen)

def show(node): [print(f"{' '*SPACES*l}{n}") for n,l in walk(node, level=0)]

def serialize(head:Node, text:str) -> str:
    """node tree to html"""
    ret = """<!DOCTYPE html><html><head><link rel="stylesheet" href="test.css"/><title>Spiritstream</title></head>"""
    prevNode, prevLevel = None, -1
    for node, level in walk(head, level=0):
        assert node.name != "bolditalic", "bolditalic nodes should be replace by bold and italic nodes"
        if node.name == "text":
            if node.parent.name in ["link", "embed"] and ((node.parent.data == "wiki" and node is not node.parent.children[0]) or \
            (node.parent.data == "md switched" and node is not node.parent.children[0])): continue
            ret += text[node.start:node.end].replace("\n", "<br>")
        else:
            while prevLevel >= level:
                ret += htmltag(prevNode, text, open=False)
                assert prevNode.parent is not None
                prevNode, prevLevel = prevNode.parent, prevLevel - 1
            ret += htmltag(node, text)
            prevNode = node
            prevLevel = level
    while prevNode is not head:
        ret += htmltag(prevNode, text, open=False)
        prevNode = prevNode.parent
    return ret + htmltag(head, text, open=False) + "</html>"

def htmltag(node:Node, text:str, open=True) -> str:
    match node.name:
        case "body": return f"<{'' if open else '/'}body>"
        case "heading": return f"<h{min(node.data, 6)}>" if open else f"</h{min(node.data, 6)}>"
        case "blockquote": return f"<{'' if open else '/'}{node.name}>"
        case "paragraph": return f"<{'' if open else '/'}p>"
        case "horizontal_rule" if open: return "<hr>" # no closing tag
        case "codeblock": return "<div class=\"codeblock\"><code><pre>" + "\n".join(node.data[1]) if open else "</pre></code></div>"
        case "empty_line" if open: return "<br>" # no closing tag
        case "bold": return f"<{'' if open else '/'}b>"
        case "italic": return f"<{'' if open else '/'}i>"
        case "strikethrough": return f"<{'' if open else '/'}s>"
        case "link": return f"<a href=\"{href(node, text)}\">" if open else "</a>"
        case "embed" if open: return f"<img src=\"{href(node, text)}\"/>" # no closing tag
        case "double_inline_code": return "<code>" if open else "</code>"
        case "inline_code": return "<code>" if open else "</code>"
        case "list": return f"<{'' if open else '/'}{'ol' if node.data[1] == 'ordered' else 'ul'}>"
        case "listitem": return f"<{'' if open else '/'}li" + ((" value=\"" + str(node.data) + "\"") if node.data is not None and open else "") + ">"
        case _: return ""

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
class CSSNode(Node):
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
                        if node.name is CSS.STYLESHEET: node = start_node(node, CSSNode(CSS.RULE,  node, [], (value,)))
                        elif node.name is CSS.RULE:
                            status = getattr(node, "status", None)
                            if status is Status.PSEUDOCLASS:
                                node.data += (":" + value,)
                                node.status = None
                            elif status is Status.OPENED: node = start_node(node, CSSNode(CSS.DECLARATION,  node, [], (value,)))
                            elif status is Status.COMMA:
                                node.data += (value,)
                                node.status = None
                            else: raise NotImplementedError(node.name, status)
                        elif node.name is CSS.DECLARATION: node = start_node(node, CSSNode(CSS.IDENTITY,  node, [], (value,)))
                        else: raise NotImplementedError(node.name)
                    case "at":
                        assert node.name is CSS.STYLESHEET
                        node = start_node(node, CSSNode(CSS.RULE,  node, [], value))
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
                        node.data += (value,)
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
                        if node.name is CSS.STYLESHEET and value == "*": node = start_node(node, CSSNode(CSS.RULE,  node, [], (value,)))
                        else: raise NotImplementedError
    while node.name is not CSS.STYLESHEET:
        node.status = Status.CLOSED
        node = node.parent
    node.status = Status.CLOSED # close head
    assert node is head
    return head
