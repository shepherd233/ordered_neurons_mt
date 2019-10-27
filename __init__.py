from fairseq.models import register_model_architecture
from .lstm_mt import LstmModel

@register_model_architecture('my_lstm','my_lstm')
def my_lstm(args):
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.encoder_embed_dim=getattr(args,'encoder_embed_dim',256)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 256)
