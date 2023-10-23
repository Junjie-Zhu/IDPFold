from .sde_lib import cmsubVPSDE, subVPSDE
import ml_collections as mlc

config_backbone = mlc.ConfigDict(
    {
        "training":
            {
                "lr": 0.0001,
                "batch_size": 2,
                "ema": 0.9999,
                "gradient_clip": None,
            },
        "network":
            {
                "irreps_node_embedding": '128x0e+64x1e+32x2e',
                "num_layers": 6,
                "irreps_node_attr": '1x0e',
                "irreps_sh": '1x0e+1x1e+1x2e',
                "max_radius": 10.0,
                "number_of_basis": 128,
                "basis_type": 'gaussian',
                "fc_neurons": [64, 64],
                "irreps_feature": '512x0e',
                "irreps_head": '32x0e+16x1o+8x2e',
                "num_heads": 4,
                "irreps_pre_attn": None,
                "rescale_degree": False,
                "nonlinear_message": False,
                "irreps_mlp_mid": '128x0e+64x1e+32x2e',
                "alpha_drop": 0.2,
                "proj_drop": 0.0,
                "out_drop": 0.0,
                "drop_path_rate": 0.0,
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
