"""
ドキュメントローダー
特定のフォルダから様々な形式のファイルを読み込み、RAG用のドキュメントに変換します
"""
import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
    CSVLoader
)


class DocumentLoader:
    """複数のファイル形式に対応したドキュメントローダー"""
    
    def __init__(self, data_folder: str):
        """
        Args:
            data_folder: 品質データや是正策が保存されているフォルダパス
        """
        self.data_folder = data_folder
        self.supported_extensions = {
            '.txt': TextLoader,
            '.pdf': PyPDFLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader,
            '.docx': Docx2txtLoader,
            '.doc': Docx2txtLoader,
            '.csv': CSVLoader
        }
    
    def load_all_documents(self) -> List[Document]:
        """
        指定されたフォルダ内のすべてのサポートされているファイルを読み込む
        
        Returns:
            List[Document]: 読み込んだすべてのドキュメントのリスト
        """
        all_documents = []
        
        if not os.path.exists(self.data_folder):
            print(f"警告: フォルダが見つかりません: {self.data_folder}")
            return all_documents
        
        # フォルダ内のすべてのファイルを走査
        for root, dirs, files in os.walk(self.data_folder):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                # サポートされている拡張子のファイルのみ処理
                if file_ext in self.supported_extensions:
                    try:
                        documents = self._load_single_file(file_path, file_ext)
                        # メタデータに追加情報を付与
                        for doc in documents:
                            doc.metadata['source_file'] = file
                            doc.metadata['source_path'] = file_path
                            doc.metadata['file_type'] = file_ext
                        all_documents.extend(documents)
                        print(f"✓ 読み込み成功: {file} ({len(documents)} ドキュメント)")
                    except Exception as e:
                        print(f"✗ 読み込み失敗: {file} - エラー: {str(e)}")
        
        print(f"\n合計 {len(all_documents)} 件のドキュメントを読み込みました")
        return all_documents
    
    def _load_single_file(self, file_path: str, file_ext: str) -> List[Document]:
        """
        単一のファイルを読み込む
        
        Args:
            file_path: ファイルパス
            file_ext: ファイル拡張子
            
        Returns:
            List[Document]: 読み込んだドキュメントのリスト
        """
        loader_class = self.supported_extensions[file_ext]
        loader = loader_class(file_path)
        return loader.load()
    
    def load_documents_by_type(self, file_type: str) -> List[Document]:
        """
        特定のファイルタイプのドキュメントのみを読み込む
        
        Args:
            file_type: ファイル拡張子 (例: '.pdf', '.xlsx')
            
        Returns:
            List[Document]: 読み込んだドキュメントのリスト
        """
        documents = []
        
        if not os.path.exists(self.data_folder):
            print(f"警告: フォルダが見つかりません: {self.data_folder}")
            return documents
        
        for root, dirs, files in os.walk(self.data_folder):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext == file_type.lower():
                    try:
                        docs = self._load_single_file(file_path, file_ext)
                        for doc in docs:
                            doc.metadata['source_file'] = file
                            doc.metadata['source_path'] = file_path
                            doc.metadata['file_type'] = file_ext
                        documents.extend(docs)
                        print(f"✓ 読み込み成功: {file}")
                    except Exception as e:
                        print(f"✗ 読み込み失敗: {file} - エラー: {str(e)}")
        
        return documents


def load_quality_documents(data_folder: str) -> List[Document]:
    """
    品質データフォルダからすべてのドキュメントを読み込む便利関数
    
    Args:
        data_folder: データフォルダのパス
        
    Returns:
        List[Document]: 読み込んだドキュメントのリスト
    """
    loader = DocumentLoader(data_folder)
    return loader.load_all_documents()
