"""NaviAI WA1 fine-tuning configs.

Each TrainConfig states its action_dim explicitly so the dimension is visible at
the config site, not buried in a DataConfig default. Modes:
    tcp_hand   = 24  (dual-arm TCP + both hand joints)
    joint_hand = 29  (MPC joints + both hand joints)
    tcp_finger = 14  (dual-arm TCP + both thumb MP continuous values)
    tcp_gripper = 7  (right-arm TCP + binarized gripper)
"""

import openpi.models.pi0_config as pi0_config
import openpi.training.weight_loaders as weight_loaders


def get_naviai_configs():
    # Import here to avoid circular imports (config.py imports this module).
    from openpi.training.config import DataConfig
    from openpi.training.config import LeRobotNaviAIDataConfig
    from openpi.training.config import LeRobotNaviAIGripperDataConfig
    from openpi.training.config import TrainConfig

    return [
        #
        # Hand-family (dual-arm, three cameras).
        #
        TrainConfig(
            name="pi0_naviai_lora_tcp",
            model=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                action_horizon=8,
            ),
            data=LeRobotNaviAIDataConfig(
                repo_id="naviai/tcp_hand_wa1_grasp_the_spoon",
                base_config=DataConfig(prompt_from_task=True),
                default_prompt="grasp the spoon",
                action_dim=24,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
            freeze_filter=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
            batch_size=32,
        ),
        TrainConfig(
            name="pi05_naviai_lora_tcp",
            model=pi0_config.Pi0Config(
                pi05=True,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                action_horizon=8,
            ),
            data=LeRobotNaviAIDataConfig(
                repo_id="naviai/tcp_hand_wa1_grasp_the_spoon",
                base_config=DataConfig(prompt_from_task=True),
                default_prompt="grasp the spoon",
                action_dim=24,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            freeze_filter=pi0_config.Pi0Config(
                pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
            batch_size=32,
            num_train_steps=30_000,
        ),
        TrainConfig(
            name="pi0_naviai_lora_joint",
            model=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                action_horizon=8,
            ),
            data=LeRobotNaviAIDataConfig(
                repo_id="naviai/joint_hand_wa1_grasp_the_spoon",
                base_config=DataConfig(prompt_from_task=True),
                default_prompt="grasp the spoon",
                action_dim=29,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
            freeze_filter=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
            batch_size=32,
        ),
        TrainConfig(
            name="pi05_naviai_lora_joint",
            model=pi0_config.Pi0Config(
                pi05=True,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                action_horizon=8,
            ),
            data=LeRobotNaviAIDataConfig(
                repo_id="naviai/joint_hand_wa1_grasp_the_spoon",
                base_config=DataConfig(prompt_from_task=True),
                default_prompt="grasp the spoon",
                action_dim=29,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            freeze_filter=pi0_config.Pi0Config(
                pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
            batch_size=32,
            num_train_steps=30_000,
        ),
        TrainConfig(
            name="pi0_naviai_lora_finger",
            model=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                action_horizon=8,
            ),
            data=LeRobotNaviAIDataConfig(
                repo_id="naviai/tcp_finger_wa1_grasp_the_spoon",
                base_config=DataConfig(prompt_from_task=True),
                default_prompt="grasp the spoon",
                action_dim=14,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
            freeze_filter=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
            batch_size=32,
        ),
        #
        # Gripper-family (right-arm only, left wrist zeroed and masked).
        #
        TrainConfig(
            name="pi0_naviai_gripper_lora_tcp",
            model=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                action_horizon=8,
            ),
            data=LeRobotNaviAIGripperDataConfig(
                repo_id="naviai/tcp_gripper_wa1_grasp_the_spoon",
                base_config=DataConfig(prompt_from_task=True),
                default_prompt="grasp the spoon",
                action_dim=7,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
            freeze_filter=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
            batch_size=32,
        ),
        TrainConfig(
            name="pi05_naviai_gripper_lora_tcp",
            model=pi0_config.Pi0Config(
                pi05=True,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                action_horizon=8,
            ),
            data=LeRobotNaviAIGripperDataConfig(
                repo_id="naviai/tcp_gripper_wa1_grasp_the_spoon",
                base_config=DataConfig(prompt_from_task=True),
                default_prompt="grasp the spoon",
                action_dim=7,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
            freeze_filter=pi0_config.Pi0Config(
                pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
            batch_size=32,
            num_train_steps=30_000,
        ),
    ]
