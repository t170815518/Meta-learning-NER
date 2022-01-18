import torch.nn as nn

import torch
from torch.nn import Parameter

import torch.nn.functional as F


def new_parameter(*size):
    out = Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)
    return out



#https://pytorch-nlp-tutorial-ny2018.readthedocs.io/en/latest/day2/patterns/attention.html
class ProjectionLayer(nn.Module):

    def __init__(self, attention_size):
        super(ProjectionLayer, self).__init__()

        self.attention = new_parameter(attention_size, 1) # 256 x 1




    def get_fixed_features(self, x_in): #[20,40,256]
        # after this, we have (batch, dim1) with a diff weight per each cell
        attention_score = torch.matmul(x_in, self.attention).squeeze()  #[20,40,256] [256,1] -->[20,40]
        attention_score = F.softmax(attention_score).view(x_in.size(0), x_in.size(1), 1)  #[20,40,1]
        scored_x = x_in * attention_score  #[20,40,256]  [20,40,1]   --> [20,40,256]

        # now, sum across dim 1 to get the expected feature vector
        condensed_x = torch.sum(scored_x, dim=1)  #[20,256]

        return condensed_x





class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """

    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    # def grad_reverse(x, constant):
    #     return GradReverse.apply(x, constant)

    def grad_reverse(x, constant=1.0):
        return GradReverse.apply(x, constant)




class DomainDiscriminator(nn.Module):

    def __init__(self,params):
        super(DomainDiscriminator, self).__init__()

        self.proj_attention = ProjectionLayer(params.rnn_direction_no * params.encoder_rnn_hidden)
        self.fc_layer = nn.Linear(params.rnn_direction_no * params.encoder_rnn_hidden, params.no_train_domain)


    def get_discriminator_log_output(self, input, constant):
        reverse_feature = GradReverse.grad_reverse(input, constant)

        out_att =  self.proj_attention.get_fixed_features(reverse_feature)
        domian_output = F.log_softmax(self.fc_layer(out_att), 1)

        return domian_output