import torch
import wandb
import numpy as np
import seaborn as sns
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from knockknock import discord_sender

from utils.lstm import LSTM
from utils.optimize_policy import get_optim_policies
from utils.accuracy import accuracy
from utils.logger import setup_logger
from utils.dataset import trajectory_dataset
from utils.average_meter import average_meter


webhook_url = "https://discordapp.com/api/webhooks/1319603151556448358/vkUwcCsV2lF2yP6Bf6K8Lgwjnr1oRRhJZtXYx6FAR-5h2PWYrlLZ2rvcNMfSO1Poo1id"

class Trainer:
    def __init__(self, args) -> None:
        self.args = args
        
        self.store_name = f"class{args.num_classes}_e{args.epochs}_batch{args.batch_size}_dropout{args.dropout}_{args.save_name}"
        self.train_loader, self.val_loader = self.create_dataloader()
        self.save_path = Path(f"./Confusion_Matrix/{self.store_name}")
        
        self.model = self.create_model()
        self.policies = self.model.parameters()
        # self.policies = get_optim_policies(self.model, lr=args.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.policies, lr=args.lr, weight_decay=args.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=0)

        self.is_best_loss = float('inf')
        self.is_best_acc = 0
        self.is_best = False

        if self.args.num_classes > 5:
            self.topk = (1, 5)
        else:
            self.topk = (1, self.args.num_classes)
    
    def run_init(self,):
        # Create the log and model folder
        self.check_rootfolders()

        # Set up weight and biases
        wandb.login()
        wandb.require("legacy-service")
        wandb.init(project="LSTM", name=self.store_name, config=self.args)

        # Set up the logger
        logger = Path(f"./log/{self.store_name}/log.txt")
        self.logger = setup_logger(output=str(logger), name=f'LSTM')
        self.logger.info('storing name: ' + self.store_name)

        if self.args.resume:
            self.resume()

        if self.args.tune_from:
            self.tune_from()

        # save args
        with open(f"./log/{self.store_name}/args.txt", "w") as f:
            f.write(str(self.args))

    def create_model(self):
        # model = MLP(self.args.num_classes, sample_length=self.args.sample_length)
        # model = TCN(38, self.args.num_classes, [40, 60, 80, 100], 3, 0.2)
        model = LSTM(self.args.num_classes, sample_length=self.args.sample_length)

        return model.cuda()
    
    def save_checkpoint(self, state, epoch):
        # Save checkpoint
        filename = '%s/%s/%d_epoch_ckpt.pth.tar' % ("checkpoint", self.store_name, epoch)
        torch.save(state, filename)

        # Save best checkpoint
        if self.is_best:
            best_filename = '%s/%s/best.pth.tar' % ("checkpoint", self.store_name)
            torch.save(state, best_filename)
            self.is_best = False
        
        # Keep only the last 5 checkpoints
        total = Path(f"./checkpoint/{self.store_name}").glob("*_ckpt.pth.tar")
        total_list = list(Path(f"./checkpoint/{self.store_name}").glob("*_ckpt.pth.tar"))
        if len(total_list) > 5:
            total = sorted(total, key=lambda x: x.stat().st_mtime)
            total[0].unlink()

    def check_rootfolders(self):
        '''
        Create log and model folder
        '''
        log = Path("log") / self.store_name
        checkpoint = Path("checkpoint") / self.store_name
        
        log.mkdir(parents=True, exist_ok=True)
        checkpoint.mkdir(parents=True, exist_ok=True)

        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def create_dataloader(self):
        train_dataset = trajectory_dataset(root=self.args.root, label_path=self.args.train_list, flip=False, sample_length=self.args.sample_length)
        # train_dataset_flip = trajectory_dataset(root=self.args.root, label_path=self.args.train_list, flip=True, sample_length=self.args.sample_length)
        # train_dataset_shift = trajectory_dataset(root=self.args.root, label_path=self.args.train_list, shift=True, sample_length=self.args.sample_length)
        # train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_flip, train_dataset_shift])
        # train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_shift])
        
        val_dataset = trajectory_dataset(root=self.args.root, label_path=self.args.val_list, sample_length=self.args.sample_length)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size, 
            shuffle=True,
            num_workers=self.args.workers, 
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.args.batch_size, 
            shuffle=False,
            num_workers=self.args.workers, 
            pin_memory=True,
        )

        return train_loader, val_loader

    def confusion_matrix(self, output, target, topk=(1,), cf_matrix=None):
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, False)
        pred = pred.t()[0] # for top1
            
        for i, j in zip(pred, target):
            cf_matrix[int(i.item()), j.item()] += 1
        
        return cf_matrix
    
    def Plot_confusion_matrix(self, matrix, acc, root, lab_x=None, lab_txt=None):
        # names = {0:"擺短/劈長", 1:"晃", 2:"撥球(挑球)", 3:"拉上旋/反拉/對拉", 4:"拉下旋",
        #          5:"放高球", 6:"攻/擋/快帶/殺球", 7:"發球"}
        # lab = [names[i] for i in range(self.args.num_classes)]
        lab = [i for i in range(self.args.num_classes)]
        plt.subplots(figsize=(10,10))
        sns.set(font_scale=1.3)
        plt.rcParams['font.family'] = 'Microsoft JhengHei'
        ax = sns.heatmap(matrix, cmap="Blues", annot=True, fmt="d", linewidths=2, linecolor='white', cbar=False, vmax=66, vmin=0) # cmap=YlGnBu, vmax=30, vmin=0

        ax.set_xlabel("lab", fontsize=15)
        ax.set_ylabel("Pred", fontsize=15, rotation=0)

        _ = ax.set_xticklabels(lab_x, rotation=0)
        _ = ax.set_yticklabels(lab_txt , rotation=360)

        ax.set_title("Top1 : %.2f"%acc, fontsize=30, y=1.0, pad=40)

        plt.rcParams['font.family'] = 'Microsoft JhengHei'

        ax.xaxis.set_label_coords(0.5, -0.3)
        ax.yaxis.set_label_coords(-0.3,0.5)
        ax.figure.savefig(f"{root}/{self.epoch}_top1.png")
        plt.close()

    def train_one_epoch(self, epoch):
        with torch.no_grad():
            losses = average_meter()
            top1 = average_meter()
            top5 = average_meter()

        self.model.train() # Set the model to train mode

        for i, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad() # set the gradient to zero
            
            # put the data and target to the GPU
            data = data.cuda()
            target = target.cuda()

            # forward pass
            output = self.model(data)

            # calculate the loss
            loss = self.criterion(output, target)

            # backward pass
            loss.backward()

            # clip the gradient if the gradient is too large
            if self.args.clip_gradient is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_gradient)

            # update the weights
            self.optimizer.step()

            # calculate the accuracy
            acc1, acc5 = accuracy(output.data, target, topk=self.topk)

            # update the average meter
            with torch.no_grad():
                losses.update(loss.item(), data.size(0))
                top1.update(acc1.item(), data.size(0))
                top5.update(acc5.item(), data.size(0))

            # print the logger
            if i % self.args.print_freq == 0:
                self.logger.info((
                    f"Epoch: [{epoch}][{i}/{len(self.train_loader)}] " \
                    f"Loss: {losses.val:.4f} ({losses.avg:.4f}) " \
                    f"Acc@1: {top1.val:.3f} ({top1.avg:.3f}) " \
                    f"Acc@5: {top5.val:.3f} ({top5.avg:.3f})" \
                ))
                # print(f"Epoch: {epoch}, Iter: {i}, Loss: {loss.item():.2f}")
        
        return losses.avg, top1.avg, top5.avg
        
    def validate_one_epoch(self, epoch):
        self.model.eval() # Set the model to evaluation mode
        cf_matrix = torch.zeros(self.args.num_classes, self.args.num_classes, dtype=torch.int32)

        with torch.no_grad():
            losses = average_meter()
            top1 = average_meter()
            top5 = average_meter()
            
            for i, (data, target) in enumerate(self.val_loader):
                # put the data and target to the GPU
                data = data.cuda()
                target = target.cuda()

                # forward pass
                output = self.model(data)
                
                # calculate the loss
                loss = self.criterion(output, target)

                # calculate the accuracy
                acc1, acc5 = accuracy(output.data, target, topk=self.topk)
                cf_matrix = self.confusion_matrix(output, target, cf_matrix=cf_matrix)

                # update the average meter
                losses.update(loss.item(), data.size(0))
                top1.update(acc1.item(), data.size(0))
                top5.update(acc5.item(), data.size(0))

                if i % self.args.print_freq == 0:
                    self.logger.info((
                        f"Test: [{i}/{len(self.val_loader)}] " \
                        f"Epoch: [{epoch}][{i}/{len(self.val_loader)}] " \
                        f"Loss: {losses.val:.4f} ({losses.avg:.4f}) " \
                        f"Acc@1: {top1.val:.3f} ({top1.avg:.3f}) " \
                        f"Acc@5: {top5.val:.3f} ({top5.avg:.3f})" \
                    ))

        self.logger.info((
            f"Testing Results: Acc@1: {top1.avg:.3f} " \
            f"Acc@5: {top5.avg:.3f} " \
            f"Loss: {losses.avg:.4f}" \
            ))
        self.logger.info(f"Confusion Matrix: \n{cf_matrix}")
        
        np.save(f"./log/{self.store_name}/cf_matrix.npy", cf_matrix)
        self.Plot_confusion_matrix(cf_matrix, top1.avg, self.save_path, lab_x=[str(i) for i in range(self.args.num_classes)], lab_txt=[str(i) for i in range(self.args.num_classes)])
        
        return top1.avg, top5.avg, losses.avg

    
    def run(self):
        @discord_sender(webhook_url=webhook_url)
        def Training_Message():
            self.run_init()
            for epoch in range(self.args.epochs):
                train_loss, train_top1, train_top5 = self.train_one_epoch(epoch)
                self.scheduler.step()
                # break
                wandb.log({'loss/train': train_loss})
                wandb.log({'acc/train_top1': train_top1})
                wandb.log({'acc/train_top2': train_top5})
                wandb.log({'lr': self.optimizer.param_groups[-1]['lr']})
                self.epoch = epoch + 1
                if (epoch + 1) % self.args.eval_freq == 0 or epoch == self.args.epochs - 1:
                    acc1, acc5, val_loss = self.validate_one_epoch(epoch)
                    wandb.log({'loss/test': val_loss})
                    wandb.log({'acc/test_top1': acc1})
                    wandb.log({'acc/test_top2': acc5})
                    if acc1 > self.is_best_acc:
                        self.is_best_acc = acc1
                        self.is_best_loss = val_loss
                        self.is_best = True
                    elif acc1 == self.is_best_acc and val_loss < self.is_best_loss:
                        self.is_best_loss = val_loss
                        self.is_best = True

                    self.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            "scheduler": self.scheduler.state_dict(),
                            'acc1': acc1,
                            'best_acc1': self.is_best_acc,
                        }, epoch + 1)
                    
                    wandb.log({'acc/test_top1_best': self.is_best_acc})
                    self.logger.info(("Best Prec@1: '{}'".format(self.is_best_acc)))

            wandb.finish()
            return f"{self.store_name}" + "\n" + f"Best Prec@1: {self.is_best_acc}"
        Training_Message()