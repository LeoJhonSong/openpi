"""NaviAI fine-tuning configs.

Config 名只承载"配方": <algo>_<space>_<tuning>, 任务/机器人/数据集均无关.
space 取值与导出器 export_lerobot.py 的 mode 一致. 同一 config 可服务多个数据集 ——
repo_id 不在此硬编码, 由训练时 CLI 覆盖 (--data.repo-id=naviai/<dataset_id>), 见 vla/openpi/justfile.
决策见 docs/adr/0012-unified-data-model-naming.md.

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
            name="pi0_tcp_hand_lora",
            model=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                action_horizon=8,
            ),
            data=LeRobotNaviAIDataConfig(
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
            name="pi05_tcp_hand_lora",
            model=pi0_config.Pi0Config(
                pi05=True,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                action_horizon=8,
            ),
            data=LeRobotNaviAIDataConfig(
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
            name="pi0_joint_hand_lora",
            model=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                action_horizon=8,
            ),
            data=LeRobotNaviAIDataConfig(
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
            name="pi05_joint_hand_lora",
            model=pi0_config.Pi0Config(
                pi05=True,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                action_horizon=8,
            ),
            data=LeRobotNaviAIDataConfig(
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
            name="pi0_tcp_finger_lora",
            model=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                action_horizon=8,
            ),
            data=LeRobotNaviAIDataConfig(
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
        # Full fine-tuning variant of the finger config: no LoRA variants, no freeze
        # filter (all params trained), and EMA left at the TrainConfig default (0.99).
        # Sized for 2x H800 80G via FSDP (fsdp_devices=2, batch_size divisible by 2).
        TrainConfig(
            name="pi0_tcp_finger_full",
            model=pi0_config.Pi0Config(
                action_horizon=8,
            ),
            data=LeRobotNaviAIDataConfig(
                base_config=DataConfig(prompt_from_task=True),
                default_prompt="grasp the spoon",
                action_dim=14,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
            batch_size=32,
            fsdp_devices=2,
        ),
        #
        # Gripper-family (right-arm only, left wrist zeroed and masked).
        #
        TrainConfig(
            name="pi0_tcp_gripper_lora",
            model=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                action_horizon=8,
            ),
            data=LeRobotNaviAIGripperDataConfig(
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
            name="pi05_tcp_gripper_lora",
            model=pi0_config.Pi0Config(
                pi05=True,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                action_horizon=8,
            ),
            data=LeRobotNaviAIGripperDataConfig(
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
