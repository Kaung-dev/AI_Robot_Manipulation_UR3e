from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UR3eRG2LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000  # pegboard pick-from-peg is harder than cube lift
    save_interval = 100
    experiment_name = "ur3e_rg2_pegboard_lift"
    # Empirical observation normalization — without it, the value-function
    # loss has spiked to inf mid-training (Iter 351) and crashed PPO.
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        # Lower LR + tighter KL than default — more stable on a long-horizon
        # task with non-trivial physics (object unhooking from peg).
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )
