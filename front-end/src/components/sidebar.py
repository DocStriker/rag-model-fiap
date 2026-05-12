import streamlit as st

from services.api_client import ApiClient

NONE_OPTION = "Nenhuma"


def _refresh_collections(api_client: ApiClient) -> None:
    try:
        st.session_state.collections = api_client.list_collections()
    except Exception as e:
        st.session_state.collections = []
        st.error(f"Erro ao listar coleções: {e}")


def render_sidebar(api_client: ApiClient) -> str | None:
    with st.sidebar:
        st.subheader("Coleções")

        if "collections" not in st.session_state:
            _refresh_collections(api_client)

        if st.button("↻ Atualizar", use_container_width=True):
            _refresh_collections(api_client)

        options = [NONE_OPTION] + st.session_state.collections
        selected = st.selectbox(
            "Selecionada",
            options=options,
            key="selected_collection",
        )
        selected = None if selected == NONE_OPTION else selected

        st.divider()

        if not selected:
            st.subheader("Criar nova")
            with st.form("create_collection_form", clear_on_submit=True):
                new_name = st.text_input("Nome", placeholder="ex.: contratos-2024")
                submit = st.form_submit_button(
                    "Criar coleção", use_container_width=True
                )
                if submit:
                    name = new_name.strip()
                    if not name:
                        st.warning("Informe um nome.")
                    else:
                        try:
                            api_client.create_collection(name)
                            _refresh_collections(api_client)
                            st.success(f"Coleção '{name}' criada.")
                            st.rerun()
                        except ValueError as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(f"Erro ao criar coleção: {e}")
        else:
            st.subheader("Documentos")
            uploaded = st.file_uploader(
                "Adicionar arquivo",
                type=["txt", "md", "pdf"],
                key=f"uploader_{selected}",
                label_visibility="collapsed",
            )
            if uploaded is not None and st.button(
                "Indexar documento", use_container_width=True
            ):
                with st.spinner("Indexando..."):
                    try:
                        result = api_client.upload_document(
                            selected, uploaded.name, uploaded.getvalue()
                        )
                        st.success(
                            f"Indexado: {result['chunks']} chunks em '{selected}'."
                        )
                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Erro no upload: {e}")

        return selected
