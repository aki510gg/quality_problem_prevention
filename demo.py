"""
RAGシステムのクイックデモ
データフォルダからインデックスを構築し、サンプルクエリを実行します
"""
import os
from vectorstore.build_vectorstore import build_vectorstore_from_folder
from rag.query_handler import RAGQueryHandler


def demo():
    """デモンストレーションを実行"""
    print("="*70)
    print("品質問題対応RAGシステム - クイックデモ")
    print("="*70)
    
    data_folder = "./data"
    index_path = "faiss_index"
    
    # インデックスが存在しない場合は構築
    if not os.path.exists(index_path):
        print("\n[ステップ1] ベクトルインデックスを構築しています...")
        print("-"*70)
        try:
            build_vectorstore_from_folder(data_folder, index_path)
            print("\n✓ インデックスの構築が完了しました！")
        except Exception as e:
            print(f"\n✗ エラー: {str(e)}")
            return
    else:
        print("\n[ステップ1] 既存のインデックスを使用します")
        print("✓ インデックスが見つかりました")
    
    # RAGハンドラーを初期化
    print("\n[ステップ2] RAGシステムを初期化しています...")
    print("-"*70)
    handler = RAGQueryHandler(index_path)
    print("✓ RAGシステムの初期化が完了しました")
    
    # サンプルクエリを実行
    print("\n[ステップ3] サンプルクエリを実行します")
    print("-"*70)
    
    sample_queries = [
        "溶接不良の是正策を教えてください",
        "寸法不良を防ぐにはどうすればいいですか",
        "過去の品質問題で最も多い原因は何ですか"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n【クエリ {i}】: {query}")
        print("-"*70)
        
        try:
            result = handler.handle_query(query, return_sources=True)
            
            print("\n【回答】")
            print(result["answer"])
            
            if result["sources"]:
                print("\n【参照元】")
                for j, source in enumerate(result["sources"][:3], 1):  # 最初の3つのみ表示
                    print(f"  {j}. {source['file']} ({source['type']})")
        
        except Exception as e:
            print(f"エラー: {str(e)}")
        
        print("\n" + "="*70)
    
    # 類似検索のデモ
    print("\n[ステップ4] 類似案件検索のデモ")
    print("-"*70)
    
    search_query = "溶接"
    print(f"\n検索キーワード: {search_query}")
    print("-"*70)
    
    try:
        similar_cases = handler.search_similar_cases(search_query, k=3)
        
        if similar_cases:
            print(f"\n{len(similar_cases)} 件の類似案件が見つかりました:\n")
            
            for i, case in enumerate(similar_cases, 1):
                print(f"【案件 {i}】")
                print(f"ファイル: {case['metadata'].get('source_file', '不明')}")
                print(f"類似度: {case['similarity_score']:.4f}")
                print(f"内容: {case['content'][:150]}...")
                print("-"*70)
        else:
            print("類似案件が見つかりませんでした")
    
    except Exception as e:
        print(f"エラー: {str(e)}")
    
    print("\n" + "="*70)
    print("デモ完了！")
    print("本格的に使用する場合は、`python main.py` を実行してください")
    print("="*70)


if __name__ == "__main__":
    # OpenAI APIキーの確認
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: OPENAI_API_KEY が設定されていません")
        print("以下のコマンドで設定してください:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print()
    
    demo()
