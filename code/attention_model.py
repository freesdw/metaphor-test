import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch






class AttentionModel(nn.Module):
    # encoder_input_dim: The encoder input dimension, outside ex_dim
    # decoder_input_dim: The decoder input dimension
    # output_dim: The output dimension
    # dropout1: dropout on input(encoder output) to linear layer
    # dropout2: dropout on input(decoder output) to linear layer
    def __init__(self, para_encoder_input_dim, query_dim, output_dim):

        super(AttentionModel, self).__init__()

        # self.dropout_on_para_input = nn.Dropout(dropout1)

        self.output_dim = output_dim

        self.para_weight = nn.Linear(para_encoder_input_dim, output_dim, bias=False)

        self.query_weight = nn.Linear(query_dim, output_dim, bias=True)

        self.attention_vec = torch.nn.Parameter(torch.randn(output_dim), requires_grad=True)


    def forward(self, para_encode_state, query, enc_padding_mask):

        # expand the dim into [batch_size, atten_len, 1, atten_size]
        para_encode_state_size = para_encode_state.size()
        para_encode_state = para_encode_state.unsqueeze(2)

        batch_size = para_encode_state_size[0]
        attn_size = para_encode_state_size[-1]
        para_linear = self.para_weight(para_encode_state)
        

        # expand the dim into [batch_size, 1, 1, attn_size]
        query_linear = self.query_weight(query)
        query_linear = query_linear.unsqueeze(1)
        query_linear = query_linear.unsqueeze(1)

        e = torch.sum(self.attention_vec * torch.tanh(para_linear + query_linear), [2,3])
        

        # take softmax. shape (batch_size, attn_length)
        attn_dist = F.softmax(e)

        # apply mask
        attn_dist = attn_dist * enc_padding_mask.float()

        masked_sums = torch.sum(attn_dist, 1)  # shape (batch_size)
        masked_sums = masked_sums.unsqueeze(1)

        assert attn_dist.size()[0] == masked_sums.size()[0]

        normalized_attn_dist = []
        for idx in range(masked_sums.size()[0]):
            if masked_sums[idx] != 0:
                normalized_attn_dist.append((attn_dist[idx] / masked_sums[idx]).detach().numpy().tolist())
            else:
                normalized_attn_dist.append((attn_dist[idx] * masked_sums[idx]).detach().numpy().tolist())
        normalized_attn_dist = torch.Tensor(normalized_attn_dist)

        return normalized_attn_dist  # re-normalize







