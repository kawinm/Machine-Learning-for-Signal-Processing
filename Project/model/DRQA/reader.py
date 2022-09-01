import torch
import torch.nn as nn
from . import layers



class DocReader(nn.Module):
    def __init__(self, options, padding_idx=0, embedding=None):
        super().__init__()
        # The options passed by the user.
        self.options = options

        # Word embeddings
        if options['pretrained_words']:
            assert embedding is not None
            # Load the embeddings
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=False)
            
            if options['fix_embeddings']:
                assert options['tune_partial'] == 0
                self.embedding.weight.requires_grad = False
            elif options['tune_partial'] > 0:
                # Tuning of the word embeddings.
                assert options['tune_partial'] + 2 < embedding.size(0)
                offset = self.options['tune_partial'] + 2

                # Everytime gradient is computed the method will be called.
                def embedding_hook(grad, offset=offset):
                    grad[offset:] = 0
                    return grad
                
                # Call the above method when gradient computation takes place.
                self.embedding.weight.register_hook(embedding_hook)

        else:  
            # If not using pretrained embeddings use embedding from 
            # pytorch.
            self.embedding = nn.Embedding(options['vocab_size'],
                                          options['embedding_dim'],
                                          padding_idx=padding_idx)
        
        if options['use_qemb']:
            # Aligned Question Embedding specified in the paper.
            self.qemb_match = layers.SeqAttnMatch(options['embedding_dim'])

        
        # The input size to the RNN
        # Depends on the amount of features to be addded.
       
        doc_input_size = options['embedding_dim'] + options['num_features']
        if options['use_qemb']:
            # Aligned Question Embedding
            doc_input_size += options['embedding_dim']
        if options['pos']:  
            # Parts of Speech
            doc_input_size += options['pos_size']
        if options['ner']: 
            # Named entity Recognition
            doc_input_size += options['ner_size']

        # Document Encoder.
        self.doc_rnn = layers.StackedRNN(
            input_size=doc_input_size,
            hidden_size=options['hidden_size'],
            nlayers=options['doc_layers'],
            dropout_rate=options['dropout_rnn'],
            output_dropout=options['dropout_rnn_output'],
            concat_layers=options['concat_rnn_layers'],
            rnn_type=nn.LSTM,
            padding=options['rnn_padding'],
        )

        # Question Encoder.
        self.question_rnn = layers.StackedRNN(
            input_size=options['embedding_dim'],
            hidden_size=options['hidden_size'],
            nlayers=options['question_layers'],
            dropout_rate=options['dropout_rnn'],
            output_dropout=options['dropout_rnn_output'],
            concat_layers=options['concat_rnn_layers'],
            rnn_type=nn.LSTM,
            padding=options['rnn_padding'],
        )

        # Output of the RNN Encoder size
        doc_hidden_size = 2 * options['hidden_size']
        question_hidden_size = 2 * options['hidden_size']
        
        # If we are concatenating the layers the hidden dimensions size 
        # need to be multiplied by a factor of the number of layers.

        if options['concat_rnn_layers']:
            doc_hidden_size *= options['doc_layers']
            question_hidden_size *= options['question_layers']
        
        # Linear Sequential Attention Layer (Self Attention)
        self.self_attn = layers.LinearSeqAttn(question_hidden_size)
        self.self_attn_doc = layers.LinearSeqAttn(doc_hidden_size)

        if options['model'] == 'triple_self_coat':
            self.trilinear = layers.ScaledDotProductAttention(10)
        if options['model'] != 'base':
            # The CoAttention Network
            self.coat = layers.CoattentionModel(question_hidden_size, self.options['dropout_emb']) 

        # Bilinear attention for span start/end
        if options['model'] == 'coat':
            
            # If we are using COAT then hidden dimension size will be 2* Normal Size
            # Neural Network for predicting the starting and ending span of the answers.
            self.start_attn = layers.BilinearSeqAttn(
                2*doc_hidden_size,
                question_hidden_size,
            )
            self.end_attn = layers.BilinearSeqAttn(
                2*doc_hidden_size,
                question_hidden_size,
            )
       
        else:
            # If we are using a model other than COAT
            # only hidden_dim would suffice as the length
            self.start_attn = layers.BilinearSeqAttn(
                doc_hidden_size,
                question_hidden_size,
            )
            self.end_attn = layers.BilinearSeqAttn(
                doc_hidden_size,
                question_hidden_size,
            )

    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask):
     
        # x1 = document word [batch * seq_length]
        # x1_f = exact match between some question word and the current paragraph word; [batch * seq_length * nfeat]
        #x1_pos =  Parts of Speech Tag of Document Word;       [batch * seq_length]
        #x1_ner = Named Entity Recognition Tags;               [batch * seq_length]
        #x1_mask = Paragraph Padding mask;                     [batch * seq_length]
        #x2 = Batch od questions;                              [batch * seq_length]
        #x2_mask =Question Padding Mask;                       [batch * seq_length]
        

        x1_emb = self.embedding(x1) # Get the GLOVE Embeddings  of the paragraph words
        x2_emb = self.embedding(x2) # Get the GLOVE Embeddings  of the question words


        if self.options['dropout_emb'] > 0:
            # Whether we are performing a dropout on the embeddings
            x1_emb = nn.functional.dropout(x1_emb, p=self.options['dropout_emb'],
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.options['dropout_emb'],
                                           training=self.training)

        # Append Exact Match Information to the Glove Embeddings
        drnn_input_list = [x1_emb, x1_f]
        if self.options['use_qemb']:
            # If we are using aligned question embedding
            # compute it and append to the word embedding.
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input_list.append(x2_weighted_emb)
        
        if self.options['pos']:
            # Index corresponding to the Parts of Speech Tag of the word.
            drnn_input_list.append(x1_pos)
        if self.options['ner']:
            # Index corresponding to the Named Entity Recognition Tag of the word.
            drnn_input_list.append(x1_ner)

        # Append all the info to get the embeddings of the paragraph words
        drnn_input = torch.cat(drnn_input_list, 2)
        
        # Encode document with RNN
        doc_hiddens = self.doc_rnn(drnn_input, x1_mask)

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(x2_emb, x2_mask)

        # Perform Self Attention over the question words to get attention weights
        q_merge_weights = self.self_attn(question_hiddens, x2_mask)

        # Compute the question embedding as a weighted combination of the
        # attention weights and the question embedding.
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        if self.options['model'] == 'coat':
            # Use the Encodings of the Question and the document for 
            # the CoAttention Model.
            U = self.coat(question_hiddens, doc_hiddens, x1_mask)
            
            # Predict start and end positions
            start_scores = self.start_attn(U, question_hidden, x1_mask)
            end_scores = self.end_attn(U, question_hidden, x1_mask)
            return start_scores, end_scores

        elif self.options['model'] == 'coat_self':
           
           # Perform Self Attention on the question words to
            doc_self_attention = self.self_attn_doc(doc_hiddens, x1_mask)        
            q_merge_weights = q_merge_weights.unsqueeze(2)
            doc_self_attention = doc_self_attention.unsqueeze(2)

            # Find Hidden Units of Coattention
            U = self.coat(q_merge_weights.expand(-1,-1, question_hiddens.shape[2]), doc_self_attention.expand(-1,-1, question_hiddens.shape[2]), x1_mask)
            U = torch.mean(U, dim =2)
            U = U.unsqueeze(2) * doc_hiddens

            # Predict start and end positions
            start_scores = self.start_attn(U, question_hidden, x1_mask)
            end_scores = self.end_attn(U, question_hidden, x1_mask)
            return start_scores, end_scores

        elif self.options['model'] == 'triple_self_coat':
            
            # TriLinear Self Attention (Key, Values)
            q_merge_weights_k = self.self_attn(question_hiddens, x2_mask)
            q_merge_weights_v = self.self_attn(question_hiddens, x2_mask)
            q_tri, q_tri_a = self.trilinear(q_merge_weights, q_merge_weights_k, q_merge_weights_v)

            # Compute TriLinear Attention weights.
            q_tri = layers.weighted_avg(question_hiddens, q_tri)

            doc_self_attention = self.self_attn_doc(doc_hiddens, x1_mask)

            # The merging weights for the question.
            q_merge_weights = q_merge_weights.unsqueeze(2)
            doc_self_attention = doc_self_attention.unsqueeze(2)
            U = self.coat(q_tri.expand(-1,-1, question_hiddens.shape[2]), doc_self_attention.expand(-1,-1, question_hiddens.shape[2]), x1_mask, None)
            U = torch.mean(U, dim =2)
            U = U.unsqueeze(2) * doc_hiddens
            
            # Predict start and end positions
            start_scores = self.start_attn(U, q_tri, x1_mask)
            end_scores = self.end_attn(U, q_tri, x1_mask)
            return start_scores, end_scores
        else:
            
            # Predict start and end positions (Default Base Model)
            start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
            end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
            return start_scores, end_scores


        
        
