#!usr/bin/python3

import math
import random
import time

import torch
import torch.nn as nn

max_length = 16
#src = torch.LongTensor()
#tgt = torch.LongTensor()

# 定义一个生成随时数据的工具函数
def generate_random_batch(batch_size, max_length=16):
    src = []
    for i in range(batch_size):
        # 随机生成句子长度
        random_len = random.randint(1, max_length - 2)
        # 随机生成句子词汇，并在开头和结尾增加<bos>和<eos>
        random_nums = [0] + [random.randint(3, 9) for _ in range(random_len)] + [1]
        # 如果句子长度不足max_length，进行填充
        random_nums = random_nums + [2] * (max_length - random_len - 2)
        src.append(random_nums)
    src = torch.LongTensor(src)
    # tgt不要最后一个token
    tgt = src[:, :-1]
    # tgt_y不要第一个的token
    tgt_y = src[:, 1:]
    # 计算tgt_y，即要预测的有效token的数量
    n_tokens = (tgt_y != 2).sum()

    # 这里的n_tokens指的是我们要预测的tgt_y中有多少有效的token，后面计算loss要用
    return src, tgt, tgt_y, n_tokens


# 定义PositionEncoding类
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

# 定义CopyTask模型
class CopyTaskModel(nn.Module):

    def __init__(self, d_model=128):
        super(CopyTaskModel, self).__init__()

        # 定义词向量，词典数为10
        self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=128)
        # 定义Transformer
        self.transformer = nn.Transformer(d_model=128, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512, batch_first=True) # 修改参数

        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout=0)

        # 定义最后的线性层，这里并没有用Softmax，因为没必要。
        # 因为后面的CrossEntropyLoss中自带了
        self.predictor = nn.Linear(128, 10)

    def forward(self, src, tgt):
        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1])
        src_key_padding_mask = CopyTaskModel.get_key_padding_mask(src)
        tgt_key_padding_mask = CopyTaskModel.get_key_padding_mask(tgt)

        # 对src和tgt进行编码
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # 将准备好的数据送给transformer
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)

        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """
        return out

    # 定义一个辅助函数来生成src_key_padding_mask和tgt_key_padding_mask # 注意pytorch版本
    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == 2] = -torch.inf
        return key_padding_mask


model = CopyTaskModel()

# 定义损失函数和优化器，因为是多分类问题，所以用CrossEntropyLoss
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


# 开始训练
total_loss = 0

for step in range(800):
    # 生成数据
    src, tgt, tgt_y, n_tokens = generate_random_batch(batch_size=2, max_length=max_length)

    # 清空梯度
    optimizer.zero_grad()
    # 进行transformer的计算
    out = model(src, tgt)
    # 将结果送给最后的线性层进行预测
    out = model.predictor(out)
    """
    计算损失。由于训练时我们的是对所有的输出都进行预测，所以需要对out进行reshape一下。
            我们的out的Shape为(batch_size, 词数, 词典大小)，view之后变为：
            (batch_size*词数, 词典大小)。
            而在这些预测结果中，我们只需要对非<pad>部分进行，所以需要进行正则化。也就是
            除以n_tokens。
    """
    loss = criteria(out.contiguous().view(-1, out.size(-1)), tgt_y.contiguous().view(-1))
    # 计算梯度
    loss.backward()
    # 更新参数
    optimizer.step()

    total_loss += loss

    # 每40次打印一下loss
    if step != 0 and step % 40 == 0:
        print("Step {}, total_loss: {}".format(step, total_loss))
        total_loss = 0

# 测试模型
model = model.eval()
# 随便定义一个src
src = torch.LongTensor([[0, 4, 3, 4, 6, 8, 9, 9, 8, 1, 2, 2]])
# tgt从<bos>开始，看看能不能重新输出src中的值
tgt = torch.LongTensor([[0]])

# 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
for i in range(max_length):
    # 进行transformer计算
    out = model(src, tgt)
    # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
    predict = model.predictor(out[:, -1])
    # 找出最大值的index
    y = torch.argmax(predict, dim=1)
    # 和之前的预测结果拼接到一起
    tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)

    # 如果为<eos>，说明预测结束，跳出循环
    if y == 1:
        break
print(tgt)