import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import FairseqEncoder,FairseqDecoder,FairseqEncoderDecoderModel,register_model

@register_model('my_lstm')
class LstmModel(FairseqEncoderDecoderModel):
    @staticmethod
    def add_args(parser): # 在init.py的add_args中调用
        parser.add_argument(
            '--encoder-embed-dim',type=int,metavar='N',
            help='dimension of encoder embeddings'
        )
        parser.add_argument(
            '--encoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the encoder hidden state',
        )
        parser.add_argument(
            '--encoder-dropout', type=float, default=0.1,
            help='encoder dropout probability',
        )
        parser.add_argument(
            '--decoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the decoder embeddings',
        )
        parser.add_argument(
            '--decoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the decoder hidden state',
        )
        parser.add_argument(
            '--decoder-dropout', type=float, default=0.1,
            help='decoder dropout probability',
        )

    @classmethod
    def build_model(cls,args,task):
        encoder=LstmEncoder(
            args=args,
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_dim=args.encoder_hidden_dim,
            dropout=args.encoder_dropout
        )
        decoder=LstmDecoder(
            dictionary=task.source_dictionary,
            encoder_hidden_dim=args.encoder_hidden_dim,
            embed_dim=args.decoder_embed_dim,
            hidden_dim=args.decoder_hidden_dim,
            dropout=args.decoder_dropout
        )
        model=LstmModel(encoder,decoder)
        return model

class LstmEncoder(FairseqEncoder):
    def __init__(self,args,dictionary,embed_dim,hidden_dim,dropout=0.1):
        super().__init__(dictionary)
        self.args=args
        self.embed_tokens=nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=embed_dim,
            padding_idx=dictionary.pad()
        )
        self.dropout=nn.Dropout(dropout)
        self.lstm=nn.LSTM( # single layer, uni-dir
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=False
        )
    
    def forward(self,src_tokens,src_lengths):
        if self.args.left_pad_source:
            src_tokens=utils.convert_padding_direction(
                src_tokens,left_to_right=True,padding_idx=self.dictionary.pad()
            )
        x=self.embed_tokens(src_tokens)
        x=self.dropout(x)

        x=nn.utils.rnn.pack_padded_sequence(x,src_lengths,batch_first=True)
        _outputs,(final_hidden,_final_cell)=self.lstm(x)
        return {
            'final_hidden':final_hidden.squeeze(0)
        }

class LstmDecoder(FairseqDecoder):
    def __init__(self,dictionary,encoder_hidden_dim=128,embed_dim=128,hidden_dim=128,dropout=0.1):
        super().__init__(dictionary)
        self.embed_tokens=nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=embed_dim,
            padding_idx=dictionary.pad()
        )
        self.dropout=nn.Dropout(dropout)
        self.lstm=nn.LSTM( # single layer, uni-dir
            input_size=embed_dim+encoder_hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=False
        )
        self.output_projection=nn.Linear(hidden_dim,len(dictionary))
    
    def forward(self,prev_output_tokens,encoder_out):
        bsz,tgt_len=prev_output_tokens.size()
        final_encoder_hidden=encoder_out['final_hidden'] # (bsz,hidden)
        x=self.embed_tokens(prev_output_tokens)
        x=self.dropout(x) # (bsz,len,embed)
        x=torch.cat([x,final_encoder_hidden.unsqueeze(1).expand(bsz,tgt_len,-1)],dim=2) # unsqueeze->(bsz,1,hidden) expand->((bsz,len,hidden))
        initial_state=(
            final_encoder_hidden.unsqueeze(0),
            torch.zeros_like(final_encoder_hidden).unsqueeze(0)
        )
        output,_=self.lstm(x.transpose(0,1),initial_state) # batch_first =false
        x=output.transpose(0,1)
        x=self.output_projection(x)

        return x,None # attention is none

