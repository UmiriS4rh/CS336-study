from cs336_basics.train_bpe import train_bpe

# 1. 拿你刚刚训练好的分词器（用 fixtures 里的小样本，秒出结果）
print("⏳ 正在加载你手搓的分词器...")
vocab, merges = train_bpe(
    input_path="tests/fixtures/tinystories_sample_5M.txt",
    vocab_size=1000,
    special_tokens=["<|endoftext|>"]
)

# 2. 准备一段测试文本
text = "The little girl was very happy, but the big dog was sad."

# 3. 模拟分词过程（展示你的分词器是怎么切词的）
print("\n" + "="*50)
print("🎯 原始文本:")
print(text)
print("="*50)

# 简单的贪心匹配演示（展示词表里能匹配到的最长词）
tokens = []
i = 0
while i < len(text):
    matched = False
    # 从长到短尝试匹配
    for length in range(min(len(text) - i, 20), 0, -1):
        chunk = text[i:i+length].encode('utf-8')
        if chunk in vocab.values():
            tokens.append(chunk)
            i += length
            matched = True
            break
    if not matched:
        # 如果没匹配到，就按单字节切
        tokens.append(text[i].encode('utf-8'))
        i += 1

# 4. 可视化输出！
print("\n🔪 你的分词器切分结果:")
print(" | ".join([t.decode('utf-8', errors='replace') for t in tokens]))

print("\n🔢 对应的词表 ID 序列:")
# 构建反向映射
id_map = {v: k for k, v in vocab.items()}
ids = [id_map.get(t, -1) for t in tokens]
print(ids)

print("\n📊 词表大小:", len(vocab))
print("🔗 学习到的合并规则数:", len(merges))
print("🏆 最新学到的几个词:", [m[0]+m[1] for m in merges[-5:]])