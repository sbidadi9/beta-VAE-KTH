class mlp_config:
    """
    A class of config for LSTM Predictor
    """
    from configs.vae import VAE_config 


    nmode       = VAE_config.latent_dim
    num_layer   = 4
    hidden_size = 128
    next_step   = 1
    in_dim      = 1
    
    Epoch       = 100
    Batch_size  = 256
    lr          = 1e-3

    train_split = 0.8 
    val_split   = 0.2 
    num_train   = 100 #135000

    early_stop = True

    if early_stop == True:
        patience  = 50
    else:
        patience  = 0 
