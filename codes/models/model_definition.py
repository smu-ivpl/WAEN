import models.archs.WAEN_P_arch as WAEN_P_arch
import models.archs.WAEN_S_arch as WAEN_S_arch

# Model Definition
def define_model(opt):
    opt_net = opt['network']
    which_model = opt_net['which_model']

    ### VSR ------------------------------------------------------------------------------------------------------------
    # Wavelet Attention Embedding Network (WAEN) Parallel
    if which_model == 'WAEN-P':
        net = WAEN_P_arch.WAEN_P(
            nframes=opt_net['nframes'],
            nf=opt_net['nf'],
            RBs=opt_net['RBs'])

    # Wavelet Attention Embedding Network (WAEN) Serial
    elif which_model == 'WAEN-S':
        net = WAEN_S_arch.WAEN_S(
            nframes=opt_net['nframes'],
            nf=opt_net['nf'],
            RBs=opt_net['RBs'])
    ### ----------------------------------------------------------------------------------------------------------------
    else:
        raise NotImplementedError('Model [{:s}] not recognized'.format(which_model))

    return net