"""
RAGクエリハンドラー
ベクトルストアを使用して類似検索を行い、過去の案件や是正策を検索します
"""
from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate


class RAGQueryHandler:
    """RAG検索とクエリ処理を行うクラス"""
    
    def __init__(self, index_path: str = "faiss_index", model_name: str = "gpt-4o-mini"):
        """
        Args:
            index_path: FAISSインデックスのパス
            model_name: 使用するOpenAIモデル名
        """
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.vectorstore = None
        self._load_vectorstore()
    
    def _load_vectorstore(self):
        """ベクトルストアを読み込む"""
        try:
            self.vectorstore = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✓ ベクトルストアを読み込みました: {self.index_path}")
        except Exception as e:
            print(f"✗ ベクトルストアの読み込みに失敗しました: {str(e)}")
            print("初回実行の場合は、まずインデックスを構築してください。")
            self.vectorstore = None
    
    def search_similar_cases(self, query: str, k: int = 5) -> List[Dict]:
        """
        類似案件を検索
        
        Args:
            query: 検索クエリ
            k: 取得する類似ドキュメント数
            
        Returns:
            List[Dict]: 類似案件のリスト（コンテンツとメタデータを含む）
        """
        if self.vectorstore is None:
            return []
        
        # 類似ドキュメントをスコア付きで検索
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in docs_with_scores:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score)
            })
        
        return results
    
    def handle_query(self, query: str, return_sources: bool = True) -> Dict:
        """
        クエリを処理して回答を生成
        
        Args:
            query: ユーザーからの質問
            return_sources: ソース情報を含めるかどうか
            
        Returns:
            Dict: 回答とソース情報を含む辞書
        """
        if self.vectorstore is None:
            return {
                "answer": "エラー: ベクトルストアが読み込まれていません。先にインデックスを構築してください。",
                "sources": []
            }
        
        # カスタムプロンプトテンプレート
        prompt_template = """あなたは品質管理と是正策の専門家です。
以下の過去の案件情報を参考に、質問に対して詳細かつ具体的な回答を提供してください。

過去の案件情報:
{context}

質問: {question}

回答する際は以下の点に注意してください:
1. 過去の類似案件から学んだ教訓を活用する
2. 具体的な是正策や対応方法を提示する
3. 再発防止のための予防策も提案する
4. 情報源が明確な場合はファイル名を参照する

回答:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # RetrievalQAチェーンを構築
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=return_sources,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        # クエリを実行
        result = qa_chain.invoke({"query": query})
        
        # 結果を整形
        response = {
            "answer": result["result"],
            "sources": []
        }
        
        if return_sources and "source_documents" in result:
            for doc in result["source_documents"]:
                source_info = {
                    "file": doc.metadata.get("source_file", "不明"),
                    "type": doc.metadata.get("file_type", "不明"),
                    "content_preview": doc.page_content[:200] + "..."
                }
                response["sources"].append(source_info)
        
        return response
    
    def get_relevant_documents(self, query: str, k: int = 3) -> List[Dict]:
        """
        クエリに関連するドキュメントを取得（LLM処理なし）
        
        Args:
            query: 検索クエリ
            k: 取得するドキュメント数
            
        Returns:
            List[Dict]: 関連ドキュメントのリスト
        """
        if self.vectorstore is None:
            return []
        
        docs = self.vectorstore.similarity_search(query, k=k)
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "source_file": doc.metadata.get("source_file", "不明"),
                "file_type": doc.metadata.get("file_type", "不明"),
                "source_path": doc.metadata.get("source_path", "")
            })
        
        return results
    
    def search_by_keywords(self, keywords: List[str], k: int = 5) -> List[Dict]:
        """
        複数のキーワードで検索
        
        Args:
            keywords: 検索キーワードのリスト
            k: 各キーワードで取得するドキュメント数
            
        Returns:
            List[Dict]: 検索結果のリスト（重複排除済み）
        """
        if self.vectorstore is None:
            return []
        
        all_results = []
        seen_contents = set()
        
        for keyword in keywords:
            docs = self.vectorstore.similarity_search(keyword, k=k)
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    all_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "keyword": keyword
                    })
        
        return all_results


def handle_query(query: str, index_path: str = "faiss_index") -> str:
    """
    クエリを処理する便利関数
    
    Args:
        query: ユーザーからの質問
        index_path: インデックスのパス
        
    Returns:
        str: 回答テキスト
    """
    handler = RAGQueryHandler(index_path)
    result = handler.handle_query(query)
    return result["answer"]
