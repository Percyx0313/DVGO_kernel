expname=None                # experiment name
basedir= './logs/'          # where to save ckpts and logs

#Tempplate of data options
data=dict(
    datadir=None,           # path to dataset root folder
    dataset_type=None,      # nsvf | blender
    white_bkgd=False,       # use white background (note that some dataset don't provide alpha and with blended bg color)
    rand_bkgd=False,        # use random background during training
    )


#Tempplate of training options
