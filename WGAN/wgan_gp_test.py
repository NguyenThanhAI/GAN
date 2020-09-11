from WGAN.wgan_gp import WGANGPConfig, WGANGP

config = WGANGPConfig(batch_size=16, learning_rate=1e-4, debug_mode=False)

wgan_gp = WGANGP(config=config)

wgan_gp.train()
