from .sde_lib import cmsubVPSDE, subVPSDE
import ml_collections as mlc

config_backbone = mlc.ConfigDict(
    {
        "training":
            {
                "lr": 0.0001,
                "batch_size": 16,
                "ema": 0.9999,
                "gradient_clip": 1,
            },
        "network":
            {
                "h_a": 64,
                "h_b": 16,
                "n_conv": 8,
            },
        "sde_config": {
            "sde": subVPSDE,
            "beta_min": 0.1,
            "beta_max": 20,
            "eps": 1e-5
        },
        "sampling": {
            "method": 'ode',
            "rtol": 1e-4,
            "atol": 1e-4,
            "noise_removal": False,
            "probability_flow": True,
            "training": {
                "continuous": True
            }
        }

    }
)
