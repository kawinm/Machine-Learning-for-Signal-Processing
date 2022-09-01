import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import torch
from torch.autograd import Variable
from .reader import DocReader


import argparse

# Helper class for the Combined loss  while predicting the starting 
# and ending postiion
class AverageMeter(object):
    def __init__(self, beta=0.99):
        self.beta = beta
        self.moment = 0
        self.value = 0
        self.t = 0
    
    def state_dict(self):
        return vars(self)
    
    def load(self, state_dict):
        for k, v in state_dict.items():
            self.__setattr__(k, v)
   
    def update(self, val):
        self.t += 1
        self.moment = self.beta * self.moment + (1 - self.beta) * val
        self.value = self.moment / (1 - self.beta ** self.t)

# Convert String to Bool. Used during preprocessing 
# training  and while inference
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



class DocReaderModel(object):
    def __init__(self, options, embedding=None, state_dict=None):
        
        # The options passed by the user.
        self.options = options
        
        self.updates = state_dict['updates'] if state_dict else 0
       
        # Whether to use CPU/GPU
        self.device = torch.cuda.current_device() if options['cuda'] else torch.device('cpu')

        # The training loss
        self.train_loss = AverageMeter()

        # The loss value if resuming from a paused state
        if state_dict:
            self.train_loss.load(state_dict['loss'])

        # Building network.
        self.network = DocReader(options, embedding=embedding)
        
        # If loading parameters from a pretrained model
        if state_dict:
            # Get the parameters of the network
            new_state = set(self.network.state_dict().keys())
            # Find the parameters from the saved model
            for k in list(state_dict['network'].keys()):
                # If the saved paramter field not present ignore.
                if k not in new_state:
                    del state_dict['network'][k]
            # Load the network parameters
            self.network.load_state_dict(state_dict['network'])
        
        # Transfer the model to GPU.
        self.network.to(self.device)

        # Get the optimizer.
        self.opt_state_dict = state_dict['optimizer'] if state_dict else None
        self.build_optimizer()

    def build_optimizer(self):
        
        # Make all the parameters of the network to have gradients.
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        # I the optimizer is SGD
        if self.options['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(parameters, self.options['learning_rate'],
                                       momentum=self.options['momentum'],
                                       weight_decay=self.options['weight_decay'])
        
        # If the optimizer is AdaMax
        elif self.options['optimizer'] == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.options['weight_decay'])
        
        # If the optimizer is Adam
        elif self.options['optimizer'] == 'adam':
            self.optimizer = optim.Adam(parameters,
                                          weight_decay=self.options['weight_decay'])
        else:
            raise RuntimeError('Unsupported optimizer: %s' % self.options['optimizer'])
        
        # Load the state dict from the optimizer.
        if self.opt_state_dict:
            self.optimizer.load_state_dict(self.opt_state_dict)

    def update(self, ex):

        # Set the model parameters to training mode
        self.network.train()

        # Obtain ( paragraph,paragraph_features,
        # paragraph_word_POS,
        # paragraph word NER ,paragraph Mask
        # question words, questions_mask)

        inputs = [e.to(self.device) for e in ex[:7]]
        
        target_s = ex[7].to(self.device) # The starting index of the answer in the words of the paragraph
        target_e = ex[8].to(self.device) # The ending index of the answer  in the words of the paragraph.

        # Get the scores as a probability distribution over the words
        # in the paragraph.
        score_s, score_e = self.network(*inputs)

        # Compute the loss
        loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
        
        # Update the loss value
        self.train_loss.update(loss.item())

        # Clear gradients and run backward
        self.optimizer.zero_grad()
        # Perform backpropogation of losses
        loss.backward()

        # Gradient clipping  to avoid exploding gradients(Large updates to weights while training)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                      self.options['grad_clipping'])

        # Update parameters
        self.optimizer.step()
        # Update the number of updates done so far
        self.updates += 1

    def predict(self, ex):

        # Set the model to evalution mode
        self.network.eval()

        # Transfer the inputs to GPU.
        if self.options['cuda']:
            inputs = [Variable(e.cuda()) for e in ex[:7]] 
        else:
            inputs = [Variable(e) for e in ex[:7]]

        # Get the scores 
        with torch.no_grad():
            score_s, score_e = self.network(*inputs)

        # Transfer the tensors to CPU after inference has
        # been performed.
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()

        # Get argmax text spans
        text = ex[-2]  
        spans = ex[-1]
        predictions = []
        max_len = self.options['max_len'] or score_s.size(1) # The maximum length of the sequences
        for i in range(score_s.size(0)):
            # Compute p_start(i) * p_end(j) to find the span that maximizes probability.
            scores = torch.ger(score_s[i], score_e[i])
            # Ending index needs to be more than the starting index.
            scores.triu_().tril_(max_len - 1)
            # Convert the scores to  numpy array.
            scores = scores.numpy() 
            # Get the starting index and the ending index of the answer.
            s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape) 
            s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1] 
           
            predictions.append(text[i][s_offset:e_offset]) # Append the answer to the predictions.

        # Return the answer spans
        return predictions

    
    def save(self, filename, epoch, scores):
        
        # Save the parametes of the model.
        # Get the EM,F1 scores.
        em, f1, best_eval = scores

        # The network parameters.
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates,
                'loss': self.train_loss.state_dict()
            },
            'config': self.options,
            'epoch': epoch,
            'em': em,
            'f1': f1,
            'best_eval': best_eval,
            'random_state': random.getstate(),
            'torch_state': torch.random.get_rng_state(), # Random Number Generator
            'torch_cuda_state': torch.cuda.get_rng_state() # Random Number Generator CUDA
        }
        # Save the Best Model.
        torch.save(params, filename)
     
