"""A tiny block-YAML emitter, because the runner ships with no YAML library.

The SDK's zero-dependency rule (see README.md)
 is not a preference — a bank installs this package inside its own
environment, and every dependency is something their software-composition
review has to clear. PyYAML would be one more of those for the sake of
writing a config file we generate ourselves.

The trick that keeps this honest: **YAML 1.2 is a superset of JSON**, so
every scalar is emitted as `json.dumps` produces it. Escaping, unicode and
quoting are therefore the stdlib's problem, not this module's, and there is
no hand-rolled quoting logic to get wrong. Only the block structure —
indentation, `key:`, `- ` — is written here.

Scope is exactly what `PromptfooAdapter.prepare` needs: dicts, lists, and
JSON scalars. Anything else raises rather than being coerced, because a
config file that silently lost a value would produce a red-team run that
tested something other than what the campaign asked for.
"""
from __future__ import annotations

import json
from typing import Any

_SCALARS = (str, int, float, bool, type(None))


def dump(value: Any) -> str:
    """Serialize `value` as block YAML. Returns a string ending in a newline."""
    if isinstance(value, dict):
        body = _block(value, 0)
    elif isinstance(value, list):
        body = _block(value, 0)
    else:
        raise TypeError("the top level of a YAML document must be a mapping or a sequence")
    return body if body.endswith("\n") else body + "\n"


def _scalar(value: Any) -> str:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int, float)):
        # json.dumps gives `null`, `true`, `123`, `1.5`, `"text"` — all valid
        # YAML 1.2 scalars with correct escaping.
        #
        # `ensure_ascii=False` is load-bearing, not a preference. The default
        # escapes a non-BMP character as a UTF-16 SURROGATE PAIR
        # (`"😀"`). JSON parsers recombine those; YAML does not — a
        # `\uXXXX` in YAML is one code point, so the value read back is two
        # lone surrogates. An emoji in a bundle's attack payload or policy
        # text would reach promptfoo mangled, which is precisely the
        # "silently lost a value" failure this module's docstring exists to
        # rule out. Writing the character itself is correct in both formats.
        return json.dumps(value, ensure_ascii=False)
    raise TypeError(f"cannot emit {type(value).__name__} in YAML: {value!r}")


def _block(value: Any, indent: int) -> str:
    pad = "  " * indent
    if isinstance(value, dict):
        if not value:
            return f"{pad}{{}}\n"
        out = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"YAML mapping keys must be strings, got {type(key).__name__}")
            if isinstance(item, (dict, list)) and item:
                out.append(f"{pad}{_scalar(key)}:\n")
                out.append(_block(item, indent + 1))
            else:
                out.append(f"{pad}{_scalar(key)}: {_inline(item)}\n")
        return "".join(out)
    if isinstance(value, list):
        if not value:
            return f"{pad}[]\n"
        out = []
        for item in value:
            if isinstance(item, (dict, list)) and item:
                nested = _block(item, indent + 1)
                # Hang the first line of the nested block off the "- ".
                first, _, rest = nested.partition("\n")
                out.append(f"{pad}- {first.lstrip()}\n")
                if rest:
                    out.append(rest if rest.endswith("\n") else rest + "\n")
            else:
                out.append(f"{pad}- {_inline(item)}\n")
        return "".join(out)
    return f"{pad}{_scalar(value)}\n"


def _inline(value: Any) -> str:
    if isinstance(value, dict) and not value:
        return "{}"
    if isinstance(value, list) and not value:
        return "[]"
    if isinstance(value, _SCALARS):
        return _scalar(value)
    raise TypeError(f"cannot emit {type(value).__name__} inline")
