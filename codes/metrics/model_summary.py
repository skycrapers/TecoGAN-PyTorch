import torch
import torch.nn as nn


# define which modules to be incorporated
registered_module = [
    nn.Conv2d,
    nn.ConvTranspose2d,
    nn.Conv3d
]

# initialize
registered_hooks, model_info_lst = [], []


def calc_2d_gflops_per_batch(module, out_h, out_w):
    """ Calculate flops of conv weights (support groups_conv & dilated_conv)
    """
    gflops = 0
    if hasattr(module, 'weight'):
        # Note: in_c is already divided by groups while out_c is not
        bias = 0 if hasattr(module, 'bias') else -1
        out_c, in_c, k_h, k_w = module.weight.shape

        gflops += (2*in_c*k_h*k_w + bias)*out_c*out_h*out_w/1e9
    return gflops


def calc_3d_gflops_per_batch(module, out_d, out_h, out_w):
    """ Calculate flops of conv weights (support groups_conv & dilated_conv)
    """
    gflops = 0
    if hasattr(module, 'weight'):
        # Note: in_c is already divided by groups while out_c is not
        bias = 0 if hasattr(module, 'bias') else -1
        out_c, in_c, k_d, k_h, k_w = module.weight.shape

        gflops += (2*in_c*k_d*k_h*k_w + bias)*out_c*out_d*out_h*out_w/1e9
    return gflops


def hook_fn_forward(module, input, output):
    if isinstance(module, nn.Conv3d):
        batch_size, _, out_d, out_h, out_w = output.size()
        gflops = batch_size*calc_3d_gflops_per_batch(module, out_d, out_h, out_w)
    else:
        if isinstance(module, nn.ConvTranspose2d):
            batch_size, _, out_h, out_w = input[0].size()
        else:
            batch_size, _, out_h, out_w = output.size()
        gflops = batch_size*calc_2d_gflops_per_batch(module, out_h, out_w)

    model_info_lst.append({'gflops': gflops})


def register_hook(module):
    if isinstance(module, tuple(registered_module)):
        registered_hooks.append(module.register_forward_hook(hook_fn_forward))


def register(model, dummy_input_list):
    # reset params
    global registered_hooks, model_info_lst
    registered_hooks, model_info_lst = [], []

    # register hook
    model.apply(register_hook)

    # forward
    with torch.no_grad():
        model.eval()
        out = model(*dummy_input_list)

    # remove hooks
    for hook in registered_hooks:
        hook.remove()

    return out


def parse_model_info(model):
    tot_gflops = 0
    for module_info in model_info_lst:
        if module_info['gflops']:
            tot_gflops += module_info['gflops']

    tot_params = 0
    for param in model.parameters():
        tot_params += torch.prod(torch.tensor(param.size())).item()

    return tot_gflops, tot_params
