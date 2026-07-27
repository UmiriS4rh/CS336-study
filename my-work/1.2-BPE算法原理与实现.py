'''1.1
为何需要tokenizer?
通过Unicode和UTF-8编码,我们可以将文本转换为计算机可以处理的字节序列。
然而，在自然语言处理(NLP)任务中，直接使用字节序列进行处理并不高效，也不符合人类语言的结构。
因此,我们需要一个tokenizer来将文本分割成更小的单元(tokens),这些单元可以是单词、子词或字符。
transferormer的时间复杂度是O(n^2),所以tokenizer的作用是将文本分割成更小的单元,以便模型能够更高效地处理和理解文本。

'''

'''1.2
预分词:希望压缩后的token具有正常的语义,而不是单纯的字节序列,所以我们需要预分词。

'''
import os
from collections import defaultdict, Counter
import regex as re

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. 初始化词表所有可能的字节
    vocab = {i: bytes([i]) for i in range(256)}
    num_merges = vocab_size - 256 - len(special_tokens)
    
    # 2. 读取语料并根据特殊 token 切分
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if special_tokens:
        special_regex = '|'.join(re.escape(t) for t in special_tokens)
        parts = re.split(f'({special_regex})', text)
        train_segments = [p for p in parts if p not in special_tokens]
    else:
        train_segments = [text]

    # 3. GPT-2 预分词正则 (修复了正则表达式)
    gpt2_pat = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?+[^\s\p{L}\p{N}]++[\r\n]*+|\s*[\r\n]|\s+(?!\S)|\s+""")

    raw_counter = Counter()
    for segment in train_segments:
        words = gpt2_pat.findall(segment)
        for word in words:
            raw_counter[tuple(bytes([b]) for b in word.encode('utf-8'))] += 1

    word_list = [list(word_tuple) for word_tuple in raw_counter.keys()]
    counts_list = list(raw_counter.values())

    # 4. 初始化统计和倒排索引
    stats = defaultdict(int)
    indices = defaultdict(set)

    for idx, word in enumerate(word_list):
        freq = counts_list[idx]
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            stats[pair] += freq
            indices[pair].add(idx)

    merges = []
    for _ in range(num_merges):
        if not stats:
            break

        # 找到频率最高的字节对 (修复了全角冒号)
        best_pair = max(stats.items(), key=lambda x: (x[1], x[0]))[0]

        if stats[best_pair] <= 0:
            break

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]  # bytes 可以直接相加
        relevant_indices = list(indices[best_pair])

        for idx in relevant_indices:
            # 【关键修复】合并前，先将当前单词从旧字节对的索引中移除
            indices[best_pair].discard(idx)

            word = word_list[idx]
            freq = counts_list[idx]
            i = 0
            while i < len(word) - 1:
                if word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    # 减去旧的相邻对频率
                    if i > 0:
                        prev_pair = (word[i - 1], word[i])
                        stats[prev_pair] -= freq
                        if stats[prev_pair] == 0:
                            del stats[prev_pair]
                    if i < len(word) - 2:
                        next_pair = (word[i + 1], word[i + 2])
                        stats[next_pair] -= freq
                        if stats[next_pair] == 0:
                            del stats[next_pair]

                    # 执行合并
                    word[i] = new_token
                    del word[i + 1]

                    # 加上新的相邻对频率
                    if i > 0:
                        new_prev = (word[i - 1], word[i])
                        stats[new_prev] += freq
                        indices[new_prev].add(idx)
                    if i < len(word) - 1:
                        new_next = (word[i], word[i + 1])
                        stats[new_next] += freq
                        indices[new_next].add(idx)
                else:
                    i += 1

        if best_pair in stats:
            del stats[best_pair]
        if best_pair in indices:
            del indices[best_pair]

    # 5. 【关键修复】先添加特殊 token，再按顺序添加合并的 token
    for s_tok in special_tokens:
        s_bytes = s_tok.encode('utf-8')
        vocab[len(vocab)] = s_bytes
        
    for pair in merges:
        new_id = len(vocab)
        vocab[new_id] = pair[0] + pair[1]

    return vocab, merges

if __name__ == "__main__":
    # 1. 准备测试数据
    text_test = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest"""

    # 2. 写入临时文件
    with open("test_bpe.txt", 'w', encoding='utf-8') as f:
        f.write(text_test)
    
    # 3. 设置参数并开始训练
    special_tokens = ["<|endoftext|>"]
    target_vocab_size = 265  # 256基础字节 + 1个特殊token + 8次合并
    
    print(f"正在训练 BPE... (目标词表大小: {target_vocab_size})\n")
    
    # 调用你写的核心函数
    final_vocab, merge_list = train_bpe("test_bpe.txt", target_vocab_size, special_tokens)
    
    # 4. 打印结果验证
    print("-" * 50)
    print("合并过程记录：")
    for i, (b1, b2) in enumerate(merge_list):
        # 将 bytes 转为字符串方便阅读
        s1 = b1.decode('utf-8', errors='replace')
        s2 = b2.decode('utf-8', errors='replace')
        print(f"第 {i+1} 次合并: '{s1}' + '{s2}' -> '{s1+s2}'")
        
    print("-" * 50)
    print(f"最终词表大小: {len(final_vocab)}")
    print("新增的 Token (ID >= 256):")
    for k, v in sorted(final_vocab.items()):
        if k >= 256:
            print(f"  ID {k}: {v}")