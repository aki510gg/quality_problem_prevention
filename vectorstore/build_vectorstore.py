"""
ベクトルストア構築モジュール
ドキュメントをベクトル化してFAISSインデックスを作成し、類似検索を可能にします
"""
import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class VectorStoreBuilder:
    """ベクトルストアの構築と管理を行うクラス"""
    
    def __init__(self, index_path: str = "faiss_index"):
        """
        Args:
            index_path: FAISSインデックスを保存するパス
        """
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def build_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        ドキュメントからベクトルストアを構築
        
        Args:
            documents: ベクトル化するドキュメントのリスト
            
        Returns:
            FAISS: 構築されたベクトルストア
        """
        if not documents:
            raise ValueError("ドキュメントが空です。ベクトルストアを構築できません。")
        
        print(f"ドキュメントを分割中... (元のドキュメント数: {len(documents)})")
        # ドキュメントをチャンクに分割
        split_docs = self.text_splitter.split_documents(documents)
        print(f"分割完了: {len(split_docs)} チャンクを作成")
        
        print("ベクトル化とインデックス構築中...")
        # ベクトルストアを作成
        vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        print("ベクトルストア構築完了")
        
        return vectorstore
    
    def save_vectorstore(self, vectorstore: FAISS):
        """
        ベクトルストアをディスクに保存
        
        Args:
            vectorstore: 保存するベクトルストア
        """
        vectorstore.save_local(self.index_path)
        print(f"ベクトルストアを保存しました: {self.index_path}")
    
    def load_vectorstore(self) -> Optional[FAISS]:
        """
        保存されたベクトルストアを読み込む
        
        Returns:
            FAISS: 読み込んだベクトルストア、存在しない場合はNone
        """
        if not os.path.exists(self.index_path):
            print(f"警告: インデックスが見つかりません: {self.index_path}")
            return None
        
        try:
            vectorstore = FAISS.load_local(
                self.index_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"ベクトルストアを読み込みました: {self.index_path}")
            return vectorstore
        except Exception as e:
            print(f"ベクトルストアの読み込みに失敗しました: {str(e)}")
            return None
    
    def update_vectorstore(self, new_documents: List[Document]):
        """
        既存のベクトルストアに新しいドキュメントを追加
        
        Args:
            new_documents: 追加するドキュメントのリスト
        """
        vectorstore = self.load_vectorstore()
        
        if vectorstore is None:
            print("既存のインデックスがないため、新規作成します")
            vectorstore = self.build_vectorstore(new_documents)
        else:
            print(f"新しいドキュメントを追加中... ({len(new_documents)} 件)")
            split_docs = self.text_splitter.split_documents(new_documents)
            vectorstore.add_documents(split_docs)
            print(f"{len(split_docs)} チャンクを追加しました")
        
        self.save_vectorstore(vectorstore)


def build_vectorstore_from_folder(data_folder: str, index_path: str = "faiss_index") -> FAISS:
    """
    フォルダからドキュメントを読み込んでベクトルストアを構築する便利関数
    
    Args:
        data_folder: データが保存されているフォルダパス
        index_path: インデックスの保存先パス
        
    Returns:
        FAISS: 構築されたベクトルストア
    """
    from loaders.load_documents import load_quality_documents
    
    print(f"データフォルダからドキュメントを読み込み中: {data_folder}")
    documents = load_quality_documents(data_folder)
    
    if not documents:
        raise ValueError(f"フォルダ内にドキュメントが見つかりませんでした: {data_folder}")
    
    builder = VectorStoreBuilder(index_path)
    vectorstore = builder.build_vectorstore(documents)
    builder.save_vectorstore(vectorstore)
    
    return vectorstore
