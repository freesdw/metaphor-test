import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import torch
from attention_model import AttentionModel

class RNNSequenceModel(nn.Module):
    # num_classes: The number of classes in the classification problem.
    # embedding_dim: The input dimension
    # hidden_size: The size of the RNN hidden state.
    # num_layers: Number of layers to use in RNN
    # bidir: boolean of wether to use bidirectional or not in RNN
    # dropout1: dropout on input to RNN
    # dropout2: dropout in RNN
    # dropout3: dropout on hidden state of RNN to linear layer
    def __init__(self, num_classes, embedding_dim, hidden_size, num_layers, bidir=True,
                 dropout1=0.2, dropout2=0.2, dropout3=0.2):

        super(RNNSequenceModel, self).__init__()

        self.rnn = nn.LSTM(input_size=embedding_dim , hidden_size=hidden_size,
                           num_layers=num_layers, dropout=dropout2, batch_first=True, bidirectional=bidir)

        direc = 2 if bidir else 1

        self.output_to_label = nn.Linear(hidden_size * direc + 300, num_classes)

        self.dropout_on_input_to_LSTM = nn.Dropout(dropout1)
        self.dropout_on_input_to_linear_layer = nn.Dropout(dropout3)

        self.embedding_linear = nn.Linear(300, hidden_size * direc)
        self.tanh = nn.Tanh()

    def forward(self, inputs, lengths):

        embedded_input = self.dropout_on_input_to_LSTM(inputs)

        output, _ = self.rnn(embedded_input)

        embedding_proj = embedded_input[:,:,:300]

        output_cat = torch.cat([output, embedding_proj], -1)

        input_encoding = self.dropout_on_input_to_linear_layer(output_cat)

        unnormalized_output = self.output_to_label(input_encoding)

        output_distribution = F.log_softmax(unnormalized_output, dim=-1)

        return output_distribution


class ExpModel(nn.Module):
    # num_classes: The number of classes in the classification problem.
    # embedding_dim: The input dimension
    # hidden_size: The size of the RNN hidden state.
    # num_layers: Number of layers to use in RNN
    # bidir: boolean of wether to use bidirectional or not in RNN
    # dropout1: dropout on input to RNN
    # dropout2: dropout in RNN
    # dropout3: dropout on hidden state of RNN to linear layer
    def __init__(self, embedding_dim, hidden_size, num_layers, bidir=True,
                 dropout1=0.2):

        super(ExpModel, self).__init__()

        self.rnn = nn.LSTM(input_size=embedding_dim , hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True, bidirectional=bidir)

        self.dropout_on_input_to_LSTM = nn.Dropout(dropout1)


    def forward(self, inputs):

        embedded_input = self.dropout_on_input_to_LSTM(inputs)

        output, _ = self.rnn(embedded_input)

        return output


class ParaEncoder(nn.Module):
    # num_classes: The number of classes in the classification problem.
    # embedding_dim: The input dimension
    # hidden_size: The size of the RNN hidden state.
    # num_layers: Number of layers to use in RNN
    # bidir: boolean of wether to use bidirectional or not in RNN
    # dropout1: dropout on input to RNN
    # dropout2: dropout in RNN
    # dropout3: dropout on hidden state of RNN to linear layer
    def __init__(self, embedding_dim, hidden_size, num_layers, attn_model, bidir=True,
                 dropout1=0.2):

        super(ParaEncoder, self).__init__()

        self.rnn = nn.LSTM(input_size=embedding_dim , hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True, bidirectional=bidir)

        self.dropout_on_input_to_LSTM = nn.Dropout(dropout1)

        self.attn_model = attn_model


    def forward(self, inputs, target_query_state, paraset_len_mask):

        embedded_input = self.dropout_on_input_to_LSTM(inputs)
        batch_size = embedded_input.size()[0]
        para_array = []
        for i in range(embedded_input.size()[1]):
            temp = embedded_input[:, i, :, :]
            len_arr = torch.LongTensor(paraset_len_mask[:, i])
            output, (a, h) = self.rnn(temp)
            temp_arr = []
            len_arr_mask = []
            for batch in range(batch_size):
                temp_arr.append(output[batch, len_arr[batch]-1, :])
                t_mask = torch.ones(output.size()[1], dtype=torch.int64)
                t_mask[len_arr[batch]:] = 0
                len_arr_mask.append(t_mask)
            # temp_arr = torch.stack(temp_arr)
            len_arr_mask = torch.stack(len_arr_mask)

            # attn_model = AttentionModel(para_encoder_input_dim=512, query_dim=512, output_dim=512)
            attn_weight = self.attn_model(output, target_query_state, len_arr_mask)
            attn_weight = attn_weight.unsqueeze(-1)
            context_vector = attn_weight * output
            context_vector = torch.sum(context_vector, [1])

            para_array.append(context_vector)
        final_output = torch.stack(para_array, dim=1) #[batch, paraset_size, para_vec_dim]

        return final_output

class LinearFC(nn.Module):
    # num_classes: The number of classes in the classification problem.
    # embedding_dim: The input dimension
    # hidden_size: The size of the RNN hidden state.
    # num_layers: Number of layers to use in RNN
    # bidir: boolean of wether to use bidirectional or not in RNN
    # dropout1: dropout on input to RNN
    # dropout2: dropout in RNN
    # dropout3: dropout on hidden state of RNN to linear layer
    def __init__(self, num_classes, encoded_embedding_dim, context_dim, dropout1=0.2):

        super(LinearFC, self).__init__()

        self.output_to_label = nn.Linear(encoded_embedding_dim + context_dim, num_classes)

        self.dropout_on_input_to_FC = nn.Dropout(dropout1)


    def forward(self, encoded_state, context_vector):
        inputs = torch.cat((encoded_state, context_vector),dim=1)
        embedded_input = self.dropout_on_input_to_FC(inputs)
        output = self.output_to_label(embedded_input)
        normalized_output = torch.log_softmax(output, dim=-1)

        return normalized_output


class ModelSet(nn.Module):
    # num_classes: The number of classes in the classification problem.
    # embedding_dim: The input dimension
    # hidden_size: The size of the RNN hidden state.
    # num_layers: Number of layers to use in RNN
    # bidir: boolean of wether to use bidirectional or not in RNN
    # dropout1: dropout on input to RNN
    # dropout2: dropout in RNN
    # dropout3: dropout on hidden state of RNN to linear layer
    def __init__(self, Exp_model, Attn_model, para_encoder, linearfc, Query_model):

        super(ModelSet, self).__init__()

        self.Exp_model = Exp_model
        self.attn_model = Attn_model
        self.para_encoder = para_encoder
        self.linearfc = linearfc
        self.Query_model = Query_model

    def encode_text(self, inputs):

        return self.Exp_model(inputs)

    def encode_text_q(self, inputs):

        return self.Query_model(inputs)

    def encode_paraphrase(self, inputs):

        return self.para_encoder(inputs)

    def compute_attn(self, inputs):

        return self.attn_model(inputs)

    def fct_transmation(self, inputs):

        return self.linearfc(inputs)

    def forward(self, example_pos, example_text, example_lengths, pos_idx, paraset, paraset_attn, paraset_len_mask, para_idx):

        encoder_state = self.Exp_model(example_text)

        target_encoder_state = [encoder_state[batch][pos_idx[batch]] for batch in range(paraset_attn.size()[0])]
        target_encoder_state = torch.stack(target_encoder_state)

        query_state = self.Query_model(example_text)
        target_query_state = [query_state[batch][pos_idx[batch]] for batch in range(paraset_attn.size()[0])]
        target_query_state = torch.stack(target_query_state)

        para_encoded = self.para_encoder(paraset, target_query_state, paraset_len_mask)

        context_vector = torch.tensor([para_encoded[batch][para_idx[batch]].detach().numpy().tolist() for batch in range(para_idx.size()[0])])

        # attn_weight = self.attn_model(para_encoded, target_encoder_state, paraset_attn)
        # attn_weight = self.attn_model(para_encoded, target_query_state, paraset_attn)

        # attn_weight = attn_weight.unsqueeze(-1)

        # context_vector = attn_weight * para_encoded

        # context_vector = torch.sum(context_vector, [1])

        predicted = self.linearfc(target_encoder_state, context_vector)

        return predicted





