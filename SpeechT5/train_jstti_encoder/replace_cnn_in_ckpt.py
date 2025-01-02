import torch

# If the CNN fails in one training but the encoder is good, we can merge the two checkpoints.
enc_ckpt_fn = '/path/to/encoder_with_bad_cnn_but_good_enc/checkpoint_best.pt'
cnn_ckpt_fn = '/path/to/encoder_with_good_cnn_but_bad_enc/checkpoint_best.pt'
write_ckpt_fn = '/path/to/final_encoder_with_good_cnn_and_good_enc/checkpoint_best.pt'

cnn_ckpt = torch.load(cnn_ckpt_fn)

enc_ckpt = torch.load(enc_ckpt_fn)

cnn_keys = [key for key in cnn_ckpt['model'].keys() if 'policy_network' in key]

for cnn_key in cnn_keys:
    enc_ckpt['model'][cnn_key] = cnn_ckpt['model'][cnn_key]
    
torch.save(enc_ckpt, write_ckpt_fn)