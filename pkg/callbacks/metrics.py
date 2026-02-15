from .callback import Callback
from pkg.utils.plot import plot_metrics

import os

class Metrics(Callback):
    """
    Collects and aggregates (averages) scalar metrics from training and validation batch steps. 
    """
    def __init__(self, priority=0):

        self.current_train_metrics = {}
        self.history_train_metrics = {}

        self.current_val_metrics = {}
        self.history_val_metrics = {}

        self.batch_count = 0

        # Set lowest priority by default
        self.priority = priority

    def on_fit_start(self, context):
        
        # Initialize path 
        self.log_csv = os.path.join(context["dir"], "metrics.csv")
        self.first_epoch = True
    
    def on_train_epoch_start(self, context):
        self.batch_count = 0
        
    def on_train_batch_end(self, context, out, batch, batch_idx):

        self.batch_count += 1
        for k,v in out.items():
            
            if not k in self.current_train_metrics:
                self.current_train_metrics[k] = v
            else:
                self.current_train_metrics[k] += v
      
    def on_train_epoch_end(self, context):

        for k, v in self.current_train_metrics.items():

            avg = v
            if self.batch_count > 0:
                # Take average over batches
                avg /= self.batch_count

            # Save them to history 
            if not k in self.history_train_metrics:
                self.history_train_metrics[k] = [avg]
            else:
                self.history_train_metrics[k].append(avg)

            # Expose to context using prefix
            name = "train/"+k
            context["metrics"][name] = avg

    def on_val_epoch_start(self, context):
        self.batch_count = 0

    def on_val_batch_end(self, context, out, batch, batch_idx):
        self.batch_count += 1
        for k,v in out.items():
            
            if not k in self.current_val_metrics:
                self.current_val_metrics[k] = v
            else:
                self.current_val_metrics[k] += v

    def on_val_epoch_end(self, context):

        for k, v in self.current_val_metrics.items():

            avg = v
            if self.batch_count > 0:
                # Take average over batches
                avg /= self.batch_count

            # Save them to history 
            if not k in self.history_val_metrics:
                self.history_val_metrics[k] = [avg]
            else:
                self.history_val_metrics[k].append(avg)

            # Expose to context using prefix
            name = "val/"+k
            context["metrics"][name] = avg
        
        # Dump to csv 
        with open(self.log_csv, "a") as f:

            if self.first_epoch:
                self.first_epoch = False

                # Write metrics names
                f.write(",".join([k for k,v in sorted(context["metrics"].items())]))

            # Write metric values
            f.write(",".join([v for k,v in sorted(context["metrics"].items())]))

        # Plot all metrics
        all_metrics = { ("train/"+k):v for k,v in self.history_train_metrics }
        for k,v in self.history_val_metrics:
            all_metrics["val/"+k] = v 
        plot_metrics( os.path.join(context["dir"], "metric_plots.png"), all_metrics)

        # Clear metrics
        self.current_val_metrics = {}
        self.current_train_metrics = {}
