import streamlit as st

from services.api_client import ApiClient

HISTORY_LIMIT = 5


def render_chat(api_client: ApiClient, collection: str | None = None) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if collection:
        pill = (
            f'<div class="status-pill"><span class="dot"></span>'
            f"Conectado a <b>{collection}</b></div>"
        )
    else:
        pill = (
            '<div class="status-pill idle"><span class="dot"></span>'
            "Conversa livre</div>"
        )
    st.markdown(pill, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Digite sua mensagem...")
    if not user_input:
        return

    history = st.session_state.messages[-HISTORY_LIMIT:]

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            answer = api_client.send_message(
                user_input, collection=collection, history=history
            )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
