#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn


# In[ ]:


class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        #assert A.size(0) == self.kernel_size
        assert A.shape[0]==self.kernel_size, print(A.shape,self.kernel_size)
        x = self.conv(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous() # return A?
    


# In[ ]:


class ST_GCNN_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, # 2d,temporal-spatial
                 stride,
                 dropout=0,
                 bias=True):
        
        super(ST_GCNN_layer,self).__init__()
        self.kernel_size = kernel_size
        
        
        self.gcn=ConvTemporalGraphical(in_channels,out_channels,kernel_size[0]) # the convolution layer
        
        self.block= [nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True),nn.Dropout(dropout,inplace=True)] # the other layers


            
        
                
        
        if stride != 1 or in_channels != out_channels: 

            self.residual=nn.Sequential(nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
            
            
        else:
            self.residual=nn.Identity()
        
        self.block=nn.Sequential(*self.block)
        
        self.relu = nn.ReLU(inplace=True)

        

    def forward(self, x, A):
        assert A.shape[0] == self.kernel_size[0], print(A.shape[0],self.kernel_size)
        res=self.residual(x)
        x=self.gcn(x,A) # do we need to return A, if its unchanged?
        output=self.block(x)
        output=output+res
        output=self.relu(output)
        return output
                                       
        
        
    


# In[ ]:


class TXC_CNN_layer(nn.Module):
    """ 
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, T_{in},C, V)` format
        - Output[0]: Output graph sequence in :math:`(N,T_{out},C, V)` format
        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :n_channels=number of channels for the coordiantes(default=3)
            in_channels=T_in
            out_channels=T_out
            +
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size, # 2d,channel-spatial
                 dropout=0,# do we need dropout and/or batchnorm?
                 bias=True):
        
        super(TXC_CNN_layer,self).__init__()
        self.kernel_size = kernel_size
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2) # padding so the channels and joints dimensions are maintained

        
        
        self.block= [nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding),nn.BatchNorm2d(out_channels)] # the other layers



            
        
        self.block=nn.Sequential(*self.block)
        

    def forward(self, x):
        
        output= self.block(x)
        return output
                                       


# In[ ]:


class Social_STGCNN_model(nn.Module):

    def __init__(self,
                 n_st_gcnn_layers,
                 n_txpcnn_layers,
                 input_channels,
                 output_channels,
                 input_time_frame,
                 output_time_frame,
                 temporal_kernel_size,
                 
                 bias=True):
        
        super(Social_STGCNN_model,self).__init__()
        
        self.n_st_gcnn_layers=n_st_gcnn_layers
        self.n_txpcnn_layers=n_txpcnn_layers
        self.st_gcnns=nn.ModuleList()
        self.txp_cnns=nn.ModuleList()
        
        
        self.st_gcnns.append(ST_GCNN_layer(input_channels,output_channels,[temporal_kernel_size,3],stride=1))
        for i in range(1,n_st_gcnn_layers):
            self.st_gcnns.append(ST_GCNN_layer(output_channels,output_channels,[temporal_kernel_size,3],stride=1))
            
        # tcn has the role of making an embedding of shape(n,C_embedding,T,V) into (n,C_original=3,T,V)
        self.tcn = nn.Sequential(
            nn.Conv2d(
                output_channels,
                input_channels,
                (3, 1),
                (1, 1),
                (1,0),
            ),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

        # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)           
        self.txp_cnns.append(TXC_CNN_layer(input_time_frame,output_time_frame,kernel_size=[3,3])) # with kernel_size[3,3] the dimensinons of C,V will be maintained       
        for i in range(1,n_txpcnn_layers):
            self.txp_cnns.append(TXC_CNN_layer(output_time_frame,output_time_frame,kernel_size=[3,3]))



            
        
                
        self.relu=nn.ReLU(inplace=True)


        

    def forward(self, x, a):
        for k in range(self.n_st_gcnn_layers):
            x = self.st_gcnns[k](x,a)
            
        x=self.tcn(x)

        x= x.permute(0,2,1,3) # prepare the input for the Time-Extrapolator-CNN
        
        x=self.txp_cnns[0](x)
        
        for i in range(1,self.n_txpcnn_layers):
            activation=self.relu(x)
            x=self.txp_cnns[i](activation)+x # residual connection
            
        return x

