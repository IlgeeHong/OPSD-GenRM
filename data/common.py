"""Shared utilities for OPSD-GenRM data preprocessing."""

from jinja2 import Template


MULTI_TURN_TEMPLATE = Template(
    "{% for message in messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if message['role'] == 'assistant' %}"
    "<assistant>\n{{ message['content'] | trim }}\n</assistant>\n\n"
    "{% else %}"
    "<user>\n{{ message['content'] | trim }}\n</user>\n\n"
    "{% endif %}"
    "{% endfor %}"
)


def format_context(context_messages, include_format=False):
    """Format context messages. If include_format is True, apply the chat-style
    template.  Otherwise return raw content."""
    if include_format:
        return MULTI_TURN_TEMPLATE.render(messages=context_messages).rstrip()
    return context_messages[0]["content"]


def format_response(response, include_format=False):
    """Wrap a response string in <assistant> tags if include_format is True."""
    if include_format:
        return f"<assistant>\n{response.strip()}\n</assistant>"
    return response
