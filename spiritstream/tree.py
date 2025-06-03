from dataclasses import dataclass
from typing import List, Optional, Union, Generator, Any, Iterable
from spiritstream.config import SPACES
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
    name:str
    parent:Optional["Node"] = None
    children:Optional[List["Node"]] = None
    text:Optional[str]=None # TODO: remove. text are children nodes
    start:int = None
    end:int = None
    data:Any = None
    
    def __repr__(self):
        return  f"\033[32mNode\033[0m({self.name}, \033[94mstart=\033[0m{self.start}, \033[94mend=\033[0m{self.end}, \033[94mdata=\033[0m{self.data})"
        # return f"Node({self.name}, start={self.start}, end={self.end}, data={self.data})"

def tokenize(text:str) -> Generator[Token, None, None]:
    text_start = None # because the non-greedy pattern matches single characters, accumulates "text" matches and stores start and end.
    for m in PATTERN.finditer(text):
        for name, value in m.groupdict().items():
            if value != None:
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
    node = head = Node("body", None, []) if head is None else head # node I am currently adding to aka. most recent unclosed node
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
            case "heading": node = start_node(head, Node(tok.name, head, [], None, tok.idx, None, tok.data))
            case "empty_line": node = start_node(head, Node(tok.name, head, [], None, tok.idx, None, tok.data))
            case "paragraph_line" if node.name != "paragraph": node = start_node(head, Node("paragraph", head, [], None, tok.idx, None, tok.data))
            case "horizontal_rule": node = start_node(head, Node(tok.name, head, [], None, tok.idx, None, tok.data))
            case "codeblock": node = start_node(head, Node(tok.name, head, [], None, tok.idx, None, tok.data))
            case "indented_line" if tok.data[0] >= 4:
                if node.name != "codeblock": node = start_node(head, Node("codeblock", head, [], None, tok.idx, None, (None, [" "*(tok.data[0]-4) + tok.data[1]])))
                else: node.data[1].append(" "*(tok.data[0]-4) + tok.data[1])
            case "blockquote_line":
                if node.name != "blockquote": node = start_node(head, Node("blockquote", head, [], None, tok.idx, None, 1))
                while node.data < tok.data: node = start_node(node, Node("blockquote", node, [], None, tok.idx, None, node.data+1))
                while node.name == "blockquote" and node.data > tok.data: node = end_node(node, tok.idx)
            case "listitem":
                newindent = tok.data[0] // SPACES
                newlisttype = tok.data[1]
                digit = tok.data[2]
                listnode = node if node.name == "list" else node.parent if node.name == "listitem" and node.parent and node.parent.name == "list" else None
                if listnode is None: node = listnode = start_node(head, Node("list", head, [], None, tok.idx, None, (0, "")))
                while (previndent:=listnode.data[0]) < newindent:
                    if node.name != "listitem": node = start_node(listnode, Node("listitem", listnode, [], None, tok.idx))
                    node = listnode = start_node(node, Node("list", node, [], None, tok.idx, None, (previndent+1, "")))
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
                        node = listnode = start_node(node, Node("list", node, [], None, tok.idx, None, (newindent, newlisttype)))
                assert node is listnode and node.data[1] == tok.data[1]
                node = start_node(node, Node(tok.name, node, [], None, tok.idx, None, digit))
            
            case "wikilink_embed_start" if not inlink: node = start_node(node, Node("embed", node, [], None, tok.idx, None, "wiki"))
            case "wikilink_start" if not inlink: node = start_node(node, Node("link", node, [], None, tok.idx, None, "wiki"))
            case "wikilink_end" if inlink:
                if node.data == "wiki": node = end_node(node, tok.idx)
                else: node.data = "md switch" # this token can end markdownlinks too. data is used when parsing next token to see if conditions for a full switch are satisfied
            case "markdownlink_embed_start" if not inlink: node = start_node(node, Node("embed", node, [], None, tok.idx, None, "md"))
            case "markdownlink_start" if not inlink: node = start_node(node, Node("link", node, [], None, tok.idx, None, "md"))
            case "markdownlink_switch" if inlink: 
                if node.data == "wiki": # can switch wikilinks too in cases like [[link](link) the first [ is ignored, so start is shifted
                    node.start += 1 if node.name == "link" else 2 # if "embed", there is a "!" that is being ignored too
                    node.name = "link"
                node.data = "md switched" 
        
            case "opening_parenthesis" if inlink and node.data == "md switch": node.data = "md switched"
            case "closing_parenthesis" if inlink and node.data == "md switched": node = end_node(node, tok.idx)

            case "text" if not (node.name in ["empty_line", "codeblock"] and tok.data - tok.idx == 1): # text will match newline at end of empty_line and codeblock which is useless
                if node.children and (n:=node.children[-1]).name == "text" and n.end == tok.idx: n.end = tok.data # merge with previous text if contiguous
                else: node.children.append(Node("text", node, [], None, tok.idx, tok.data))

            case _ if not inlink or node.data == "md": # inline formatting only allowed outside of links except in the label part of a markdown link                
                match tok.name:
                    case "italic_start":
                        if node.name != "bolditalic": node = start_node(node, Node("italic", node, [], None, tok.idx))
                    case "italic_toggle"|"italic_end":
                        if node.name == "italic": node = end_node(node, tok.idx)
                        elif node.name == "bolditalic": # replace bolditalic with opening bold and opening italic. Start of opening italic is offset accordingly. Italic ends here
                            node.name = "bold"
                            node = end_node(start_node(node, Node("italic", node, [], None, node.start+2)), tok.idx)
                        elif tok.name == "italic_toggle": node = start_node(node, Node("italic", node, [], None, tok.idx))

                    case "bold_start":
                        if node.name != "bolditalic": node = start_node(node, Node("bold", node, [], None, tok.idx))
                    case "bold_toggle"|"bold_end":
                        if node.name == "bold": node = end_node(node, tok.idx)
                        elif node.name == "bolditalic": # replace bolditalic with opening italic and opening bold. Start of opening bold is offset accordingly. Bold ends here
                            node.name = "italic"
                            node = end_node(start_node(node, Node("bold", node, [], None, node.start+1)), tok.idx)
                        elif tok.name == "bold_toggle": node = start_node(node, Node("bold", node, [], None, tok.idx))
                    
                    case "bolditalic_start" if node.name not in ["italic", "bold"]: node = start_node(node, Node("bolditalic", node, [], None, tok.idx))
                    case "bolditalic_toggle"|"bolditalic_end":
                        if node.name == "bolditalic": node = end_node(node, tok.idx)
                        elif node.name == "italic":
                            node = end_node(node, tok.idx)
                            if node.name == "bold": end_node(node, tok.idx+1)
                            else: node = start_node(node, Node("bold", node, [], None, tok.idx+1))
                        elif node.name == "bold":
                            node = end_node(node, tok.idx)
                            if node.name == "italic": node = end_node(node, tok.idx+2)
                            else: node = start_node(node, Node("italic", node, [], None, tok.idx+2))
                        elif tok.name == "bolditalic_toggle": node = start_node(node, Node("bolditalic", node, [], None, tok.idx))

                    case "strikethrough_start": node = start_node(node, Node("strikethrough", node, [], None, tok.idx))
                    case "strikethrough_toggle"|"strikethrough_end":
                        if node.name == "strikethrough": node = end_node(node, tok.idx)
                        elif tok.name == "strikethrough_toggle": node = start_node(node, Node("strikethrough", node, [], None, tok.idx))

                    case "double_inline_code":
                        if node.name == "double_inline_code": node = end_node(node, tok.idx)
                        elif node.name == "inline_code":
                            node = end_node(node, tok.idx)
                            node = start_node(node, Node("inline_code", node, [], None, tok.idx + 1))
                        else: node = start_node(node, Node(tok.name, node, [], None, tok.idx))

                    case "inline_code":
                        if node.name == "inline_code": node = end_node(node, tok.idx)
                        elif node.name != "double_inline_code": node = start_node(node, Node(tok.name, node, [], None, tok.idx))
    return head

def parse(text:str, head=None) -> Node: return treeify(tokenize(text), head)

def start_node(parent:Node, child:Node) -> Node:
    assert child.parent is parent
    parent.children.append(child)
    return child

def end_node(node, idx, offset=True) -> Node:
    assert node.parent != None, node
    end = idx + node_format_length(node) if offset else idx
    node.end = end
    if node.name == "bolditalic": # replace bolditalic with bold and italic nodes, close both
        node.name = "bold"
        node = end_node(start_node(node, Node("italic", node, [], None, node.start+2)), idx, offset=offset)
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