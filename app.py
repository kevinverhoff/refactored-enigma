import json

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent import graph


def _extract_text(content) -> str:
    """Normalize AIMessage content: Gemini may return a list of content blocks
    instead of a plain string when extended thinking metadata is present."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


st.set_page_config(page_title="DLGF Memo Assistant", layout="wide")
st.title("DLGF Memo Assistant")
st.caption("Ask questions about Indiana DLGF memos and guidance documents (2022-2026)")

with st.sidebar:
    st.header("Optional Filters")
    st.markdown("Set defaults here, or mention filters naturally in your question.")
    year_filter = st.number_input("Year (on or after)", min_value=2022, max_value=2026,
                                  value=None, step=1)
    doc_type_filter = st.selectbox("Document type", ["", "MEMO", "TEMPLATE", "FORM", "ATTACHMENT"])
    author_filter = st.text_input("Author")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(
        "**Source code**\n\n"
        "[![GitHub](https://img.shields.io/badge/GitHub-refactored--enigma-181717?logo=github)](https://github.com/kevinverhoff/refactored-enigma)"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(_extract_text(msg.content))

if prompt := st.chat_input("Ask about DLGF memos..."):
    filter_parts = []
    if year_filter:
        filter_parts.append(f"year >= {int(year_filter)}")
    if doc_type_filter:
        filter_parts.append(f"doc_type = {doc_type_filter}")
    if author_filter:
        filter_parts.append(f"author = {author_filter}")

    full_prompt = f"[Filters active: {', '.join(filter_parts)}] {prompt}" if filter_parts else prompt

    user_msg = HumanMessage(content=full_prompt)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        step_lines = []
        steps_placeholder = st.empty()
        answer_placeholder = st.empty()
        ai_content = ""

        for update in graph.stream({"messages": st.session_state.messages}, stream_mode="updates"):
            for node_name, node_output in update.items():
                msgs = node_output.get("messages", [])
                for msg in msgs:
                    # Tool call decisions from the agent node
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            args_str = json.dumps(tc["args"], ensure_ascii=False)
                            step_lines.append(f"**Tool call:** `{tc['name']}({args_str})`")
                        steps_placeholder.markdown(
                            "**Agent reasoning:**\n\n" + "\n\n".join(step_lines)
                        )
                    # Tool results from the tools node
                    elif hasattr(msg, "content") and node_name == "tools":
                        preview = _extract_text(msg.content)[:120].replace("\n", " ")
                        step_lines.append(f"**Tool result:** {preview}...")
                        steps_placeholder.markdown(
                            "**Agent reasoning:**\n\n" + "\n\n".join(step_lines)
                        )
                    # Final answer from the agent node (no tool calls)
                    elif hasattr(msg, "content") and node_name == "agent" and not getattr(msg, "tool_calls", None):
                        ai_content = _extract_text(msg.content)
                        answer_placeholder.markdown(ai_content)

        # Collapse reasoning into an expander once done
        if step_lines:
            steps_placeholder.empty()
            with st.expander("Agent reasoning", expanded=False):
                st.markdown("\n\n".join(step_lines))

    st.session_state.messages.append(AIMessage(content=ai_content))