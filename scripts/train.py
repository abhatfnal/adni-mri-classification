
from adni_mri_classification.training.trainer import Trainer
from adni_mri_classification.training.scheduler import Scheduler
from adni_mri_classification.training.checkpoint import CheckpointManager
from adni_mri_classification.data.splits import Splitter
from adni_mri_classification.data.datasets import ADNIDataset, TransformDataset
from adni_mri_classification.data.augmentation import Augmentation

from torch.utils.data import Subset, DataLoader


def main(epochs):
    
    # Full dataset
    ds = ADNIDataset(...)
    
    # Create splitter 
    splitter = Splitter(...)
    
    # Test dataset
    test_idx = splitter.test_split()
    test_ds = Subset(ds, test_idx)
    test_loader = DataLoader(test_ds, ...)
    
    # Create augmentation
    augmentation = Augmentation(...)

    for fold, (train_idx, val_idx) in enumerate(splitter.cv_split()):
        
        # Train and validation datasets
        train_ds = TransformDataset(Subset(ds, train_idx), transform=...)
        val_ds = Subset(ds, val_idx)
        
        # Assert there is no subject level leakage
        assert set(ds.groups[train_idx]).isdisjoint(set(ds.groups[val_idx]))
        assert set(ds.groups[train_idx]).isdisjoint(set(ds.groups[test_idx]))
        
        # initialize model, checkpoint manager, task
        task = Task(...)
        scheduler  = Scheduler(...)
        trainer = Trainer(...)
        checkpoint_manager = CheckpointManager(...)
        
        # loop
        for epoch in range(epochs):

            # train one epoch, update checkpoint, log metrics		
            trainer.train_epoch(epoch)
            
            # update checkpoint manager: stores historical metrics, saves best model state, implements patience
            if checkpoint_manager.update(epoch, trainer.current_metrics()):
                checkpoint_manager.save_state(model.state_dict())
            
            # stop if needed						
            if checkpoint_manager.early_stop():
                break
            
        # Save training+val metrics to file (e.g. losses.csv)		
        checkpoint_manager.save_metrics("/path/to/file")
        
        # Get best model checkpoint					
        best_model_state = checkpoint_manager.best_state()
        
        # Load best state and evaluate on test set
        model.load(best_model_state)
        
        model.eval()
        test_metric = TestMetric()
        
        for batch in test_dataloader.iterator:
            X, y = batch
            test_metric.update(task.test_batch())
        
        # save/log test metrics
        ...
