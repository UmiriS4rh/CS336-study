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
from collections import defaultdict,Counter
import regex as re
import json


def train_bpe(
    input_path: str | os.PathLike,  # 输入语料文件的路径
    vocab_size: int,                # 词汇表大小
    special_tokens: list[str],      # 需要保留的特殊token列表
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    '''
    该函数核心流程：
    1.初始化词表所有可能的字节
    2.读取输入预料,并根据特殊token切分,确保特殊token不统计
    3.使用gpt-2的预分词正则将语料库切分为单词,并统计每个单词的频率
    4.迭代进行合并操作,直到词表大小达到指定的vocab_size
    5.使用倒排索引优化合并过程的频率更新
    6.将合并的token添加到词表中,并返回最终的词表和合并操作列表
    '''
    vocab={i:bytes([i]) for i in range(256)} # 初始化词表所有可能的字节
    num_merges=vocab_size-256-len(special_tokens) # 计算需要进行的合并次数
    with open(input_path,'r',encoding='utf-8') as f:
        text=f.read() # 读取输入语料

    if special_tokens:
        special_regex='|'.join((re.escape(t) for t in special_tokens)) # 构建特殊token的正则表达式
        parts=re.split(f'({special_regex})',text) # 根据特殊token切分文本
        train_segments=[p for p in parts if p not in special_tokens] # 保留非特殊token的部分
    else:
        train_segments=[text] # 如果没有特殊token,直接使用整个文本

    gpt2_pat=re.compile(r"""'(?:[^\s'.,!?;:()]+|[.,!?;:()])""") # gpt-2的预分词正则

    raw_counter=Counter() # 统计每个单词的频率
    for segment in train_segments:
        words=gpt2_pat.findall(segment) # 使用正则切分文本
        for word in words:
            raw_counter[tuple(bytes([b]) for b in word.encode('utf-8'))] += 1

    word_list=[]
    counts_list=[]
    for word_tuple,freq in raw_counter.items():
        word_list.append(list(word_tuple))
        counts_list.append(freq)

    stats=defaultdict(int) # 统计每个字节对的频率
    indices=defaultdict(set) # 倒排索引,记录每个字节对出现的单词索引

    for idx,word in enumerate(word_list):
        freq=counts_list[idx]
        for i in range(len(word)-1):
            pair=(word[i],word[i+1])
            stats[pair]+=freq
            indices[pair].add(idx)

    merges=[]
    for _ in range(num_merges):
      if not stats:
          break


      best_pair=max(stats.items(),key=lambda x: (x[1],x[0]) )[0] # 找到出现频率最高的字节对      

      if stats[best_pair]<=0:
          break   

      merges.append(best_pair) # 将该字节对添加到合并操作列表

      new_token=bytes(best_pair[0]+best_pair[1]) # 创建新的token 

      relevant_indices=list(indices[best_pair]) # 获取包含该字节对的单词索引    

      for idx in relevant_indices:
          word=word_list[idx]
          freq=counts_list[idx]

          i=0
          while i < len(word)-1:
              if word[i]==best_pair[0] and word[i+1]==best_pair[1]:
                  if i>0:
                      prev_pair=(word[i-1],word[i])
                      stats[prev_pair]-=freq
                      if stats[prev_pair]==0:
                          del stats[prev_pair]                        