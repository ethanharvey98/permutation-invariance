def permute_conv2d_out_channels(conv, perm):
    conv.weight.data = conv.weight.data[perm, :, :, :]
    if conv.bias is not None:
        conv.bias.data = conv.bias.data[perm]

def permute_conv2d_in_channels(conv, perm):
    conv.weight.data = conv.weight.data[:, perm, :, :]
    
def permute_batchnorm2d(bn, perm):
    if bn.weight is not None:
        bn.weight.data = bn.weight.data[perm]
    if bn.bias is not None:
        bn.bias.data = bn.bias.data[perm]
    bn.running_mean = bn.running_mean[perm]
    bn.running_var = bn.running_var[perm]