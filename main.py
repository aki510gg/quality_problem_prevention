"""
品質問題対応RAGチャットボット
特定フォルダ内の品質データや是正策資料を検索し、過去の案件や類似案件を呼び出します
"""
import os
import sys
from dotenv import load_dotenv
from rag.query_handler import RAGQueryHandler
from vectorstore.build_vectorstore import build_vectorstore_from_folder

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数を取得して使う
api_key = os.getenv("API_KEY")



def initialize_system(data_folder: str, force_rebuild: bool = False):
    """
    システムを初期化し、必要に応じてベクトルストアを構築
    
    Args:
        data_folder: データが保存されているフォルダパス
        force_rebuild: Trueの場合、既存のインデックスを削除して再構築
    """
    index_path = "faiss_index"
    
    # インデックスが存在するかチェック
    index_exists = os.path.exists(index_path)
    
    if force_rebuild and index_exists:
        print("既存のインデックスを削除して再構築します...")
        import shutil
        shutil.rmtree(index_path)
        index_exists = False
    
    if not index_exists:
        print("\n" + "="*60)
        print("初回セットアップ: ベクトルストアを構築しています...")
        print("="*60)
        
        if not os.path.exists(data_folder):
            print(f"エラー: データフォルダが見つかりません: {data_folder}")
            print(f"フォルダを作成するか、正しいパスを指定してください。")
            sys.exit(1)
        
        try:
            build_vectorstore_from_folder(data_folder, index_path)
            print("\n✓ セットアップ完了！")
            print("="*60 + "\n")
        except Exception as e:
            print(f"\nエラー: ベクトルストアの構築に失敗しました: {str(e)}")
            sys.exit(1)
    else:
        print("\n✓ 既存のインデックスを使用します")


def display_menu():
    """メニューを表示"""
    print("\n" + "="*60)
    print("品質問題対応RAGシステム")
    print("="*60)
    print("1. 質問する（通常検索）")
    print("2. 類似案件を検索")
    print("3. 関連ドキュメントを表示")
    print("4. インデックスを再構築")
    print("5. 終了")
    print("="*60)


def search_similar_cases(handler: RAGQueryHandler):
    """類似案件検索モード"""
    print("\n--- 類似案件検索 ---")
    query = input("検索キーワードを入力してください: ")
    
    if not query:
        return
    
    results = handler.search_similar_cases(query, k=5)
    
    if not results:
        print("該当する案件が見つかりませんでした。")
        return
    
    print(f"\n{len(results)} 件の類似案件が見つかりました:\n")
    
    for i, result in enumerate(results, 1):
        print(f"【案件 {i}】")
        print(f"ファイル: {result['metadata'].get('source_file', '不明')}")
        print(f"類似度スコア: {result['similarity_score']:.4f}")
        print(f"内容: {result['content'][:200]}...")
        print("-" * 60)


def show_relevant_documents(handler: RAGQueryHandler):
    """関連ドキュメント表示モード"""
    print("\n--- 関連ドキュメント検索 ---")
    query = input("検索キーワードを入力してください: ")
    
    if not query:
        return
    
    docs = handler.get_relevant_documents(query, k=3)
    
    if not docs:
        print("該当するドキュメントが見つかりませんでした。")
        return
    
    print(f"\n{len(docs)} 件の関連ドキュメントが見つかりました:\n")
    
    for i, doc in enumerate(docs, 1):
        print(f"【ドキュメント {i}】")
        print(f"ファイル: {doc['source_file']}")
        print(f"種類: {doc['file_type']}")
        print(f"内容:\n{doc['content']}\n")
        print("-" * 60)


def interactive_mode(data_folder: str):
    """インタラクティブモード"""
    handler = RAGQueryHandler()
    
    while True:
        display_menu()
        choice = input("\n選択してください (1-5): ").strip()
        
        if choice == "1":
            # 通常の質問
            print("\n--- 質問モード ---")
            query = input("質問を入力してください: ")
            
            if not query:
                continue
            
            print("\n検索中...")
            result = handler.handle_query(query)
            
            print("\n" + "="*60)
            print("【回答】")
            print("="*60)
            print(result["answer"])
            
            if result["sources"]:
                print("\n" + "="*60)
                print("【参照元】")
                print("="*60)
                for i, source in enumerate(result["sources"], 1):
                    print(f"{i}. {source['file']} ({source['type']})")
                    print(f"   内容: {source['content_preview']}")
            
        elif choice == "2":
            # 類似案件検索
            search_similar_cases(handler)
            
        elif choice == "3":
            # 関連ドキュメント表示
            show_relevant_documents(handler)
            
        elif choice == "4":
            # インデックス再構築
            print("\nインデックスを再構築しますか？")
            confirm = input("続行するには 'yes' と入力: ").strip().lower()
            
            if confirm == "yes":
                initialize_system(data_folder, force_rebuild=True)
                handler = RAGQueryHandler()  # ハンドラーを再初期化
            
        elif choice == "5":
            # 終了
            print("\nシステムを終了します。")
            break
            
        else:
            print("\n無効な選択です。1-5の数字を入力してください。")


def main():
    """メイン関数"""
    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        print("\n" + "="*70)
        print("エラー: OPENAI_API_KEY が設定されていません")
        print("="*70)
        print("\n以下のいずれかの方法で設定してください:\n")
        print("【方法1】ターミナルで環境変数を設定:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("\n【方法2】.envファイルを作成:")
        print("  1. cp .env.template .env")
        print("  2. .envファイルを編集してAPIキーを設定")
        print("="*70 + "\n")
        sys.exit(1)
    
    # データフォルダのパスを設定
    # 環境変数から取得するか、デフォルト値を使用
    data_folder = os.getenv("QUALITY_DATA_FOLDER", "./data")
    
    print("品質問題対応RAGチャットボットへようこそ！")
    print(f"データフォルダ: {data_folder}")
    
    # システムを初期化
    initialize_system(data_folder)
    
    # インタラクティブモードを開始
    interactive_mode(data_folder)


if __name__ == "__main__":
    main()
