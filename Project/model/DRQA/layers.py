import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FusionBiLSTM(nn.Module):
    def __init__(self, hidden_dim, dropout_ratio):
        super(FusionBiLSTM, self).__init__()
        
        # The BiLSTM used after obtaining question to context attention
        # and context to question attention.
        self.fusion_bilstm = nn.LSTM(2304, hidden_dim, 1, batch_first=True,
                                     bidirectional=True, dropout=dropout_ratio)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, seq, mask):
       
        # Handling Variable Length sequences.
        lengths = mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort]) # Lengths of the sequences
        idx_sort = Variable(idx_sort)   # Indexes in the original tensor corresponding to sorted sequence.
        idx_unsort = Variable(idx_unsort)
        seq_ = torch.index_select(seq, 0, idx_sort)  # Sort x by length

        lens_sorted = lengths # Lengths of all the sequences
        lens_argsort_argsort = idx_unsort

        packed = nn.utils.rnn.pack_padded_sequence(seq_, lens_sorted, batch_first=True)
        # Pass through BiLSTM model inorder to obtain output.
        output, _ = self.fusion_bilstm(packed)
        # Apply padding to fix variable length sequences.
        e, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        e = e.contiguous()
        e = torch.index_select(e, 0, lens_argsort_argsort) 
        return e

class CoattentionModel(nn.Module):
    def __init__(self, hidden_dim, dropout_ratio):
        super(CoattentionModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        # LSTM used after performing CoAttention
        self.fusion_bilstm = FusionBiLSTM(hidden_dim, dropout_ratio)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, Q, D, d_mask):
       
        # Perform transformation of q as specified in the paper
        q = torch.tanh(self.q_proj(Q)).view(Q.size()) 
        d_trans = torch.transpose(D, 1, 2) 

        # The affinity matrix L
        L = torch.bmm(q, d_trans) 

        # Normalize over the question
        a_q_ = F.softmax(L, dim=1) 
        
        # Take the transpose
        a_q = torch.transpose(a_q_, 1, 2) 
        
        # Computing the vector b as specified in the paper.
        b_q = torch.bmm(d_trans, a_q) 

        q_t = torch.transpose(q, 1, 2) 
        a_d = F.softmax(L, dim=2) 
        
        # Obtain s_ia_i as specified in the paper
        s_ia_i = torch.bmm(torch.cat((q_t, b_q), 1), a_d) 
        s_ia_it = torch.transpose(s_ia_i, 1, 2) 

        #Fusion BiLSTM
        bilstm_input = torch.cat((s_ia_it, D), 2)
        
        # Representation of the paragraph taking into account of questions 
        U = self.fusion_bilstm(bilstm_input, d_mask)
        return U


class LinearSeqAttn(nn.Module):
    # Self Attention over a sequence
    # softmax(Wx_i) for x_i in a sequence X

    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        # Flatten the vector.
        x_flat = x.contiguous().view(-1, x.size(-1))
        # Perform Wx and reshape it back to batch_size*seq_length
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        # Fill scores corresponding to non existent words as -infinity
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        # Perform softmax operation over the sequence.
        alpha = F.softmax(scores, dim=1)
        # Return the attention weights which will be used to calculate
        # the weighted average.
        return alpha

#Stacked BRNN used for encoding the words in a paragraph/questiomn
class StackedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, nlayers,
                 dropout_rate=0, output_dropout=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        
        super().__init__()
        self.padding = padding                  # Handle padded sequence
        self.nlayers = nlayers                  # Number of layers in the Stacked RNN.
        self.concat_layers = concat_layers      # Whether the output is concatentation of outputs from all layers.
        self.output_dropout = output_dropout    # Dropout on the output
        self.dropout_rate = dropout_rate        # Rate at which dropout needs to be performed between Stacked Layers.
        self.rnns = nn.ModuleList()             # Used to store the RNN layers 
        
        # If the layer used is an intermediate layer then input size should be 2*hidden_size
        # else it will be input_size
        for i in range(self.nlayers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

    def forward(self, x, x_mask):
    
        #If there is no padding as good as not considering the mask at all.
        if x_mask.data.sum() == 0:
            return self.forward_unpadded(x, x_mask)
        
        if self.padding or not self.training:
            return self.forward_padded(x, x_mask)
        
        # Default do not consider padding in the inputs.
        return self.forward_unpadded(x, x_mask)


    # Do not give consideration to padded sequences.
    def forward_unpadded(self, x, x_mask):
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # outputs : Stores the RNN Hidden states from each layer
        outputs = [x]
        for i in range(self.nlayers):
            
            # rnn_input : Use the output from the previous layer
            # as input to the current layer.
            rnn_input = outputs[-1] 

            # Apply dropout to hidden input 
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            
            # Pass the input through the RNN,
            rnn_output = self.rnns[i](rnn_input)[0]
            # Append the output so that it can be used in the next layer.
            outputs.append(rnn_output)

        # If we need to concatenate layers concatente.
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Convert to batch_size * seq_length * dim
        output = output.transpose(0, 1)

        # Perform dropout on the output layer.
        if self.output_dropout and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

    def forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        
        # Compute sorted sequence lengths
        # Get the the length of each paragraph in the batch
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        # Sort the paragraphs by their length in descending order
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        #Find the indexes in the x corresponding to the sorted sequence.
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.nlayers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Pad up to original batch sequence length
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0),
                                  x_mask.size(1) - output.size(1),
                                  output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)

        # Dropout on output layer
        if self.output_dropout and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

# Computes the Aligned Question Embedding
class SeqAttnMatch(nn.Module):
    def __init__(self, input_size):
        
        super().__init__()
        self.linear = nn.Linear(input_size, input_size)
     
    def forward(self, x, y, y_mask):
    
        # Perform the operation E on the paragraph vectors.
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute the scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        # Accounting for the mask bits
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax operation
        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)
        return matched_seq


class BilinearSeqAttn(nn.Module):
    #   A bilinear attention layer over a sequence p w.r.t q:
    #   o_i = softmax(p_i^TWy) for x_i in X.
    def __init__(self, x_size, y_size):
        super().__init__()
        self.linear = nn.Linear(y_size, x_size)
     

    def forward(self, x, y, x_mask):
   
        # x = batch * seq_len * p_dim
        # y = batch * q_dim
        # x_mask = batch * seq_len
     
        # Compute Wy
        Wy = self.linear(y) if self.linear is not None else y
        # Compute xWy  OUTPUT :  batch*seq_len
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        # Fill Non existent words with -inf so that they get low scores while normalizing
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        
        if self.training:
            # In training we output log-softmax for NLL
            alpha = F.log_softmax(xWy, dim=1)
        else:
            # Otherwise 0-1 probabilities
            alpha = F.softmax(xWy, dim=1)
        return alpha


class ScaledDotProductAttention(nn.Module):
    # Perform Scaled Dot Product Attention
    # as specified in the paper
    # Attention is all you need.
    # softmax(QK^T/temperature) * V

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # Perform dropout
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # Compute QK^T/temperature
        attn = torch.matmul(q / self.temperature, k.transpose(0, 1))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        # Compute softmax(QK^T/temperature)
        attn = self.dropout(F.softmax(attn, dim=-1))
        # Perform softmax(QK^T/temperature)*V
        output = torch.matmul(attn, v)
        return output, attn


def weighted_avg(x, weights, d = False):
    
    # Take the weighted average of embeddings 
    # given weights given to each embedding 
    # in the parameter weights
    if d:
        return weights.unsqueeze(1).bmm(x)
    return weights.unsqueeze(1).bmm(x).squeeze(1)
