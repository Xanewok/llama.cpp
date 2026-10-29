"""Narrow, versioned adapters for pinned embedded chat templates."""

import hashlib


BARTOWSKI_PREFIX = "bartowski-leading-instructions-v1"
BARTOWSKI_TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"

LEADING_INSTRUCTIONS = r"""{%- set sysns = namespace(count=0, text='') %}
{%- for message in messages %}
    {%- if sysns.count == loop.index0 and (message.role == 'system' or message.role == 'developer') %}
        {%- set sys_content = render_content(message.content, false, true)|trim %}
        {%- if sys_content %}
            {%- set sysns.text = sysns.text + ('\n' if sysns.text else '') + sys_content %}
        {%- endif %}
        {%- set sysns.count = sysns.count + 1 %}
    {%- endif %}
{%- endfor %}
{%- if sysns.count %}
    {%- set messages = [{'role': 'system', 'content': sysns.text}] + messages[sysns.count:] %}
{%- endif %}
"""


def bartowski_template(source):
    if hashlib.sha256(source.encode()).hexdigest() != BARTOWSKI_TEMPLATE_SHA256:
        raise ValueError("Bartowski embedded template differs from the pinned source")
    # Only the leading instruction block is merged; later roles must still fail in place.
    anchor = "{%- set reasoning_instructions = '' %}"
    return source.replace(anchor, LEADING_INSTRUCTIONS + anchor, 1)


SWIFT_DROP_TRACES_PREFIX = "swift-drop-historical-thinking-v1"
SWIFT_TEMPLATE_SHA256 = "4af3dcf0e98bf87adf684a4824683ab15fffd64e93ed1170061ef86a15827ba8"

PRESERVE_THINKING_GUARD = (
    "{%- if preserve_thinking is undefined or preserve_thinking is true"
    " or loop.index0 > ns.last_query_index %}"
)
DROP_THINKING_GUARD = (
    "{%- if (preserve_thinking is defined and preserve_thinking)"
    " or loop.index0 > ns.last_query_index %}"
)


def swift_drop_traces_template(source):
    """Flip the Swift template's `preserve_thinking` default from true to false."""
    if hashlib.sha256(source.encode()).hexdigest() != SWIFT_TEMPLATE_SHA256:
        raise ValueError("Swift embedded template differs from the pinned source")
    # Invert the default rather than delete the branch: assistant turns after the last user
    # message are the in-flight tool-call chain and must keep their traces, or the model
    # loses the reasoning behind a call it is still acting on.
    if source.count(PRESERVE_THINKING_GUARD) != 1:
        raise ValueError("preserve_thinking guard is not unique in the pinned source")
    return source.replace(PRESERVE_THINKING_GUARD, DROP_THINKING_GUARD, 1)
