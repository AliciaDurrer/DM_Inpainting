from dataset import BRATSDataset, BRATSDDPMDataset, BRATSDDPMTestDataset
from torch.utils.data import WeightedRandomSampler


def get_dataset(cfg):
    if cfg.dataset.name == "BRATS":
        train_dataset = BRATSDataset(root_dir=cfg.dataset.root_dir, train=True)
        val_dataset = BRATSDataset(root_dir=cfg.dataset.root_dir, train=False)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == "BRATS_DDPM":
        train_dataset = BRATSDDPMDataset(root_dir=cfg.dataset.root_dir, train=True)
        val_dataset = BRATSDDPMDataset(root_dir=cfg.dataset.root_dir, train=False)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == "BRATS_DDPM_TEST":
        test_dataset = BRATSDDPMTestDataset(root_dir=cfg.dataset.root_dir, train=False)
        return test_dataset

    raise ValueError(f"{cfg.dataset.name} Dataset is not available")
