"""Basic definitions for the flows module."""

import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


from nflows.distributions.base import Distribution
from nflows.utils import torchutils
from nflows import transforms, distributions, flows

from inspect import signature


class RandomPermutationLayer(transforms.Permutation):
    """ Permutes elements with random, but fixed permutation. Keeps pixel inside layer. """
    def __init__(self, features, dim=1):
        """ features: list of dimensionalities to be permuted"""
        assert isinstance(features, list), ("Input must be a list of integers!")
        permutations = []
        for index, features_entry in enumerate(features):
            current_perm = np.random.permutation(features_entry)
            if index == 0:
                permutations.extend(current_perm)
            else:
                permutations.extend(current_perm + np.cumsum(features)[index-1])
        super().__init__(torch.tensor(permutations), dim)

class InversionLayer(transforms.Permutation):
    """ Inverts the order of the elements in each layer.  Keeps pixel inside layer. """
    def __init__(self, features, dim=1):
        """ features: list of dimensionalities to be inverted"""
        assert isinstance(features, list), ("Input must be a list of integers!")
        permutations = []
        for index, features_entry in enumerate(features):
            current_perm = np.arange(features_entry)[::-1]
            if index == 0:
                permutations.extend(current_perm)
            else:
                permutations.extend(current_perm + np.cumsum(features)[index-1])
        super().__init__(torch.tensor(permutations), dim)



class NflowsUniform(Distribution):
    """A multivariate uniform"""

    def __init__(self, shape,low,high):
        super().__init__()
        self._shape = torch.Size(shape)
        self._low=low
        self._high=high
#        self.register_buffer("_log_z",
#                             torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi),
#                                          dtype=torch.float64),
#                             persistent=False)

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        neg_energy = 1/(self._high-self._low)**(self._shape[0])*torch.ones(inputs.shape[0],device=inputs.device)
        return neg_energy

    def _sample(self, num_samples, context):
        if context is None:
            return torch.randn(num_samples, *self._shape)
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = self._low+(self._high-self._low)*torch.rand(context_size * num_samples, *self._shape,
                                  device=context.device)
            return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
#        if context is None:
#            return self._log_z.new_zeros(self._shape)
#        else:
            # The value of the context is ignored, only its size is taken into account.
        return context.new_zeros(context.shape[0], *self._shape)

class Flow_with_forward(Distribution):
    """Base class for all flow objects."""

    def __init__(self, transform, distribution, embedding_net=None):
        """Constructor.
        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
            embedding_net: A `nn.Module` which has trainable parameters to encode the
                context (condition). It is trained jointly with the flow.
        """
        super().__init__()
        self._transform = transform
        self._distribution = distribution
        distribution_signature = signature(self._distribution.log_prob)
        distribution_arguments =  distribution_signature.parameters.keys()
        self._context_used_in_base = 'context' in distribution_arguments
        if embedding_net is not None:
            assert isinstance(embedding_net, torch.nn.Module), (
                "embedding_net is not a nn.Module. "
                "If you want to use hard-coded summary features, "
                "please simply pass the encoded features and pass "
                "embedding_net=None"
            )
            self._embedding_net = embedding_net
        else:
            self._embedding_net = torch.nn.Identity()

    def _log_prob(self, inputs, context):
        embedded_context = self._embedding_net(context)
        noise, logabsdet = self._transform(inputs, context=embedded_context)
        if self._context_used_in_base:
            log_prob = self._distribution.log_prob(noise, context=embedded_context)
        else:
            log_prob = self._distribution.log_prob(noise)
        return log_prob + logabsdet

    def _sample(self, num_samples, context):
        embedded_context = self._embedding_net(context)
        if self._context_used_in_base:
            noise = self._distribution.sample(num_samples, context=embedded_context)
        else:
            repeat_noise = self._distribution.sample(num_samples*embedded_context.shape[0])
            noise = torch.reshape(
                    repeat_noise,
                    (embedded_context.shape[0], -1, repeat_noise.shape[1])
                    )

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, _ = self._transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])

        return samples

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the flow, together with their log probabilities.
        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        embedded_context = self._embedding_net(context)
        if self._context_used_in_base:
            noise, log_prob = self._distribution.sample_and_log_prob(
                num_samples, context=embedded_context
            )
        else:
            noise, log_prob = self._distribution.sample_and_log_prob(
                num_samples
            )

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, logabsdet = self._transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None):
        """Transforms given data into noise. Useful for goodness-of-fit checking.
        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.
        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        noise, _ = self._transform(inputs, context=self._embedding_net(context))
        return noise
    
    def forward(self,inputs, context):
        return self._log_prob(inputs,context)




class SinglePulsarEmbedding_LSTM(nn.Module):
    
    def __init__(self,noutput=20):
        """ Constructor
        """

        super(SinglePulsarEmbedding_LSTM, self).__init__()

#        self.dense={}
#        self.output={}
#        self.ninputlist=ninputlist
#        for ii in range(len(ninputlist)):
 
        self.lstm = nn.LSTM(1, 100,batch_first=True,bidirectional=False,num_layers=2)  # Input dim is 3, output dim is 3


        self.dense_4=nn.Linear(400,200)
        self.dense_5=nn.Linear(200,100)
        self.output=nn.Linear(100,noutput)
        
        init.kaiming_normal(self.output.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0



    def forward(self, x):
#        print(x.shape)
        inputs = x.view(-1,x.shape[1], 1)
#        print(inputs.shape)
#        hidden = (torch.randn(1, inputs.shape[1], 1), torch.randn(1, inputs.shape[1], 1))  # clean out hidden state
        out, (hidden,cell) = self.lstm(inputs)
#        print(hidden.shape)
#        print(hidden.shape,cell.shape)
#        xuse2=self.dense_4.forward(torch.hstack((hidden.view(-1,100),cell.view(-1,100))))
        xuse2=self.dense_4.forward(torch.hstack((hidden[0].view(-1,100),hidden[1].view(-1,100),\
                                                 cell[0].view(-1,100),cell[1].view(-1,100))))
#        xuse2=self.dense_4.forward(hidden.view(-1,100))
        xuse2 = F.relu(xuse2)
        xuse2=self.dense_5.forward(xuse2)
        xuse2 = F.relu(xuse2)
        return self.output(xuse2)

    
class SinglePulsarEmbedding_BiLSTM(nn.Module):
    
    def __init__(self,noutput=20):
        """ Constructor
        """

        super(SinglePulsarEmbedding_BiLSTM, self).__init__()

#        self.dense={}
#        self.output={}
#        self.ninputlist=ninputlist
#        for ii in range(len(ninputlist)):
 
        self.lstm = nn.LSTM(1, 100,batch_first=True,bidirectional=False)  # Input dim is 3, output dim is 3


        self.dense_4=nn.Linear(400,200)
        self.dense_5=nn.Linear(200,100)
        self.output=nn.Linear(100,noutput)
        
        init.kaiming_normal(self.output.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0



    def forward(self, x):
#        print(x.shape)
        inputs = x.view(-1,x.shape[1], 1)
#        print(inputs.shape)
#        hidden = (torch.randn(1, inputs.shape[1], 1), torch.randn(1, inputs.shape[1], 1))  # clean out hidden state
        out, (hidden,cell) = self.lstm(inputs)
#        print(hidden.shape)
#        print(hidden.shape,cell.shape)
        xuse2=self.dense_4.forward(torch.hstack((hidden[0].view(-1,100),hidden[1].view(-1,100),\
                                                 cell[0].view(-1,100),cell[1].view(-1,100))))
#        xuse2=self.dense_4.forward(torch.hstack((hidden.view(-1,100),cell.view(-1,100))))
#        xuse2=self.dense_4.forward(hidden.view(-1,100))
        xuse2 = F.relu(xuse2)
        xuse2=self.dense_5.forward(xuse2)
        xuse2 = F.relu(xuse2)
        return self.output(xuse2)

    

    
class EmbeddingLayer(nn.Module):
    """
    ResNext optimized for the Cifar dataset, as specified in
    https://arxiv.org/pdf/1611.05431.pdf
    """

    def __init__(self,ninputlist,noutput=20,nsingle=20):
        """ Constructor
        """

        super(EmbeddingLayer, self).__init__()

#        self.dense={}
#        self.output={}
        self.ninputlist=ninputlist
#        for ii in range(len(ninputlist)):
 
        self.embedding0=SinglePulsarEmbedding_LSTM(noutput=nsingle)
        self.embedding1=SinglePulsarEmbedding_LSTM(noutput=nsingle)
        self.embedding2=SinglePulsarEmbedding_LSTM(noutput=nsingle)
        self.embedding3=SinglePulsarEmbedding_LSTM(noutput=nsingle)
        self.embedding4=SinglePulsarEmbedding_LSTM(noutput=nsingle)
        self.embedding5=SinglePulsarEmbedding_LSTM(noutput=nsingle)
        self.embedding6=SinglePulsarEmbedding_LSTM(noutput=nsingle)
        self.embedding7=SinglePulsarEmbedding_LSTM(noutput=nsingle)
        self.embedding8=SinglePulsarEmbedding_LSTM(noutput=nsingle)
        self.embedding9=SinglePulsarEmbedding_LSTM(noutput=nsingle)


        self.dense_4=nn.Linear(nsingle*len(ninputlist),100)
        self.dense_5=nn.Linear(100,100)
        self.output=nn.Linear(100,noutput)
        
        init.kaiming_normal(self.output.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0



    def forward(self, x):
        xsplit=torch.split(x,self.ninputlist,dim=-1)
        xsplit0 = self.embedding0.forward(xsplit[0])
        xsplit1 = self.embedding1.forward(xsplit[1])
        xsplit2 = self.embedding2.forward(xsplit[2])
        xsplit3 = self.embedding3.forward(xsplit[3])
        xsplit4 = self.embedding4.forward(xsplit[4])
        xsplit5 = self.embedding4.forward(xsplit[5])
        xsplit6 = self.embedding4.forward(xsplit[6])
        xsplit7 = self.embedding4.forward(xsplit[7])
        xsplit8 = self.embedding4.forward(xsplit[8])
        xsplit9 = self.embedding4.forward(xsplit[9])
        xuse2=torch.hstack((xsplit0,xsplit1,xsplit2,xsplit3,xsplit4,xsplit5,xsplit6,xsplit7,xsplit8,xsplit9))

        xuse2=self.dense_4.forward(xuse2)
        xuse2 = F.relu(xuse2)
        xuse2=self.dense_5.forward(xuse2)
        xuse2 = F.relu(xuse2)
        return self.output(xuse2)




def define_model(nhidden=1,hidden_size=200,nblocks=8,nbins=8,embedding=None,dropout=0.05,nembedding=20,nfeatures=2):
    #hidden_size=128
    #nbins=4
    init_id=True

    if embedding==None:
        ncontext=babydataset.shape[-1]
    else:
        ncontext=nembedding
    print(ncontext)    
    
    flow_params_RQS = {'num_blocks':nhidden, # num of hidden layers per block
                       'use_residual_blocks':False,
                       'use_batch_norm':False,
                       'dropout_probability':dropout,
                       'activation':getattr(F, 'relu'),
                       'random_mask':False,
                       'num_bins':nbins,
                       'tails':'linear',
                       'tail_bound':1,
                       'min_bin_width': 1e-6,
                       'min_bin_height': 1e-6,
                       'min_derivative': 1e-6}
    flow_blocks = []
    for i in range(nblocks):
        flow_blocks.append(
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                **flow_params_RQS,
                features=nfeatures,
                context_features=ncontext,
                hidden_features=hidden_size
            ))
        if init_id:
            torch.nn.init.zeros_(flow_blocks[-1].autoregressive_net.final_layer.weight)
            torch.nn.init.constant_(flow_blocks[-1].autoregressive_net.final_layer.bias,
                                    np.log(np.exp(1 - 1e-6) - 1))

        if i%2 == 0:
            flow_blocks.append(transforms.ReversePermutation(nfeatures))
        else:
            flow_blocks.append(transforms.RandomPermutation(nfeatures))

    del flow_blocks[-1]
    flow_transform = transforms.CompositeTransform(flow_blocks)
    #if args.cond_base:
    #    flow_base_distribution = distributions.ConditionalDiagonalNormal(
    #        shape=[args.dim_sum],
    #        context_encoder=BaseContext(
    #            cond_label_size, args.dim_sum))
    #else:
    flow_base_distribution = NflowsUniform(shape=[nfeatures],low=-1,high=1)
#    flow_base_distribution = distributions.StandardNormal(shape=[2])
    if embedding==None:
        flow = Flow_with_forward(transform=flow_transform, distribution=flow_base_distribution)
    else:
        flow = Flow_with_forward(transform=flow_transform, distribution=flow_base_distribution,embedding_net=embedding)

    model = flow
    print(model)
    
    return model

