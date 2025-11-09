"""
品質問題対応RAGシステム - Streamlit Webアプリ
"""
import streamlit as st
import os
from pathlib import Path
from rag.query_handler import RAGQueryHandler
from vectorstore.build_vectorstore import build_vectorstore_from_folder


# ページ設定
st.set_page_config(
    page_title="品質問題対応RAGシステム",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_system():
    """システムの初期化"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.handler = None
        st.session_state.chat_history = []
        st.session_state.index_built = False
    
    # インデックスの存在確認
    index_path = "faiss_index"
    if os.path.exists(index_path) and not st.session_state.initialized:
        try:
            st.session_state.handler = RAGQueryHandler(index_path)
            st.session_state.initialized = True
            st.session_state.index_built = True
        except Exception as e:
            st.error(f"インデックスの読み込みに失敗しました: {str(e)}")


def build_index(data_folder: str):
    """ベクトルインデックスを構築"""
    try:
        with st.spinner("インデックスを構築中... しばらくお待ちください"):
            build_vectorstore_from_folder(data_folder, "faiss_index")
            st.session_state.handler = RAGQueryHandler("faiss_index")
            st.session_state.initialized = True
            st.session_state.index_built = True
            st.success("✓ インデックスの構築が完了しました！")
            st.rerun()
    except Exception as e:
        st.error(f"エラー: {str(e)}")


def sidebar():
    """サイドバーの表示"""
    with st.sidebar:
        st.title("🔍 RAGシステム")
        st.markdown("---")
        
        # システム状態
        st.subheader("システム状態")
        if st.session_state.index_built:
            st.success("✓ インデックス構築済み")
        else:
            st.warning("⚠ インデックス未構築")
        
        st.markdown("---")
        
        # データフォルダ設定
        st.subheader("設定")
        data_folder = st.text_input(
            "データフォルダパス",
            value="./data",
            help="品質データが保存されているフォルダのパス"
        )
        
        # インデックス構築ボタン
        if st.button("🔄 インデックスを構築", use_container_width=True):
            if os.path.exists(data_folder):
                build_index(data_folder)
            else:
                st.error(f"フォルダが見つかりません: {data_folder}")
        
        # インデックス再構築ボタン
        if st.session_state.index_built:
            if st.button("🔄 インデックスを再構築", use_container_width=True):
                import shutil
                if os.path.exists("faiss_index"):
                    shutil.rmtree("faiss_index")
                st.session_state.initialized = False
                st.session_state.index_built = False
                build_index(data_folder)
        
        st.markdown("---")
        
        # チャット履歴クリア
        if st.button("🗑️ チャット履歴をクリア", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        
        # 使い方
        with st.expander("📖 使い方"):
            st.markdown("""
            1. **初回セットアップ**: データフォルダのパスを確認し、「インデックスを構築」をクリック
            2. **質問する**: チャット欄に質問を入力
            3. **類似案件検索**: 検索タブでキーワード検索
            4. **データ更新時**: 新しいファイル追加後、「インデックスを再構築」をクリック
            """)
        
        # API キー設定状態
        with st.expander("⚙️ API設定"):
            if os.getenv("OPENAI_API_KEY"):
                st.success("✓ OpenAI APIキー設定済み")
            else:
                st.error("✗ OpenAI APIキーが未設定")
                st.code("export OPENAI_API_KEY='your-key'")


def chat_interface():
    """チャットインターフェース"""
    st.header("💬 質問応答")
    
    if not st.session_state.index_built:
        st.warning("⚠️ まずサイドバーから「インデックスを構築」を実行してください")
        return
    
    # チャット履歴表示
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 参照元"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**{i}. {source['file']}** ({source['type']})")
                        st.caption(source['content_preview'])
    
    # チャット入力
    if prompt := st.chat_input("質問を入力してください（例: 溶接不良の是正策を教えてください）"):
        # ユーザーメッセージを表示
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })
        
        # AIの応答を取得
        with st.chat_message("assistant"):
            with st.spinner("検索中..."):
                try:
                    result = st.session_state.handler.handle_query(prompt, return_sources=True)
                    st.markdown(result["answer"])
                    
                    if result["sources"]:
                        with st.expander("📚 参照元"):
                            for i, source in enumerate(result["sources"], 1):
                                st.markdown(f"**{i}. {source['file']}** ({source['type']})")
                                st.caption(source['content_preview'])
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    })
                except Exception as e:
                    error_msg = f"エラーが発生しました: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg
                    })


def search_interface():
    """類似案件検索インターフェース"""
    st.header("🔍 類似案件検索")
    
    if not st.session_state.index_built:
        st.warning("⚠️ まずサイドバーから「インデックスを構築」を実行してください")
        return
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "検索キーワード",
            placeholder="例: 溶接、寸法不良、塗装ムラ"
        )
    with col2:
        k = st.number_input("取得件数", min_value=1, max_value=20, value=5)
    
    if st.button("🔍 検索", use_container_width=True):
        if search_query:
            with st.spinner("検索中..."):
                try:
                    results = st.session_state.handler.search_similar_cases(search_query, k=k)
                    
                    if results:
                        st.success(f"✓ {len(results)} 件の類似案件が見つかりました")
                        
                        for i, result in enumerate(results, 1):
                            with st.expander(f"📄 案件 {i} - {result['metadata'].get('source_file', '不明')}"):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"**ファイル**: {result['metadata'].get('source_file', '不明')}")
                                with col2:
                                    st.metric("類似度", f"{result['similarity_score']:.4f}")
                                
                                st.markdown("**内容**:")
                                st.text_area(
                                    f"content_{i}",
                                    result['content'],
                                    height=150,
                                    label_visibility="collapsed"
                                )
                    else:
                        st.info("該当する案件が見つかりませんでした")
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
        else:
            st.warning("検索キーワードを入力してください")


def document_interface():
    """関連ドキュメント表示インターフェース"""
    st.header("📚 関連ドキュメント検索")
    
    if not st.session_state.index_built:
        st.warning("⚠️ まずサイドバーから「インデックスを構築」を実行してください")
        return
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "検索キーワード",
            placeholder="例: 是正策、再発防止"
        )
    with col2:
        k = st.number_input("取得件数", min_value=1, max_value=10, value=3, key="doc_k")
    
    if st.button("📚 検索", use_container_width=True):
        if search_query:
            with st.spinner("検索中..."):
                try:
                    docs = st.session_state.handler.get_relevant_documents(search_query, k=k)
                    
                    if docs:
                        st.success(f"✓ {len(docs)} 件のドキュメントが見つかりました")
                        
                        for i, doc in enumerate(docs, 1):
                            with st.expander(f"📄 ドキュメント {i} - {doc['source_file']}"):
                                st.markdown(f"**ファイル**: {doc['source_file']}")
                                st.markdown(f"**種類**: {doc['file_type']}")
                                st.markdown("**内容**:")
                                st.text_area(
                                    f"doc_content_{i}",
                                    doc['content'],
                                    height=200,
                                    label_visibility="collapsed"
                                )
                    else:
                        st.info("該当するドキュメントが見つかりませんでした")
                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
        else:
            st.warning("検索キーワードを入力してください")


def main():
    """メイン関数"""
    # システム初期化
    initialize_system()
    
    # サイドバー表示
    sidebar()
    
    # メインコンテンツ
    st.title("🏭 品質問題対応RAGシステム")
    st.markdown("過去の品質データや是正策資料から、類似案件や関連情報を検索できます")
    
    # タブ作成
    tab1, tab2, tab3 = st.tabs(["💬 質問応答", "🔍 類似案件検索", "📚 ドキュメント検索"])
    
    with tab1:
        chat_interface()
    
    with tab2:
        search_interface()
    
    with tab3:
        document_interface()
    
    # フッター
    st.markdown("---")
    st.caption("品質問題対応RAGシステム v1.0 | Powered by LangChain & OpenAI")


if __name__ == "__main__":
    main()
