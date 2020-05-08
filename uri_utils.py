# Utility functions to deal with ConceptNet uris
# taken shamelessly from:
# https://github.com/commonsense/conceptnet5/blob/master/conceptnet5/uri.py


def join_uri(*pieces):
    """
    `join_uri` builds a URI from constituent pieces that should be joined with
    slashes (/).
    Leading and trailing on the pieces are acceptable, but will be ignored. The
    resulting URI will always begin with a slash and have its pieces separated
    by a single slash.
    The pieces do not have `preprocess_and_tokenize_text` applied to them; to
    make sure your URIs are in normal form, run `preprocess_and_tokenize_text`
    on each piece that represents arbitrary text.
    >>> join_uri('/c', 'en', 'cat')
    '/c/en/cat'
    >>> join_uri('c', 'en', ' spaces ')
    '/c/en/ spaces '
    >>> join_uri('/r/', 'AtLocation/')
    '/r/AtLocation'
    >>> join_uri('/test')
    '/test'
    >>> join_uri('test')
    '/test'
    >>> join_uri('/test', '/more/')
    '/test/more'
    """
    joined = "/" + ("/".join([piece.strip("/") for piece in pieces]))
    return joined


def split_uri(uri):
    """
    Get the slash-delimited pieces of a URI.
    >>> split_uri('/c/en/cat/n/animal')
    ['c', 'en', 'cat', 'n', 'animal']
    >>> split_uri('/')
    []
    """
    if not uri.startswith("/"):
        return [uri]
    uri2 = uri.lstrip("/")
    if not uri2:
        return []
    return uri2.split("/")


def parse_compound_uri(uri):
    """
    Given a compound URI, extract its operator and its list of arguments.
    >>> parse_compound_uri('/nothing/[/]')
    ('/nothing', [])
    >>> parse_compound_uri('/a/[/r/CapableOf/,/c/en/cat/,/c/en/sleep/]')
    ('/a', ['/r/CapableOf', '/c/en/cat', '/c/en/sleep'])
    >>> parse_compound_uri('/or/[/and/[/s/one/,/s/two/]/,/and/[/s/three/,/s/four/]/]')
    ('/or', ['/and/[/s/one/,/s/two/]', '/and/[/s/three/,/s/four/]'])
    """
    pieces = split_uri(uri)
    if pieces[-1] != "]":
        raise ValueError("Compound URIs must end with /]")
    if "[" not in pieces:
        raise ValueError(
            "Compound URIs must contain /[/ at the beginning of the argument list"
        )
    list_start = pieces.index("[")
    op = join_uri(*pieces[:list_start])

    chunks = []
    current = []
    depth = 0

    # Split on commas, but not if they're within additional pairs of brackets.
    for piece in pieces[(list_start + 1) : -1]:
        if piece == "," and depth == 0:
            chunks.append("/" + ("/".join(current)).strip("/"))
            current = []
        else:
            current.append(piece)
            if piece == "[":
                depth += 1
            elif piece == "]":
                depth -= 1

    assert depth == 0, "Unmatched brackets in %r" % uri
    if current:
        chunks.append("/" + ("/".join(current)).strip("/"))
    return op, chunks
