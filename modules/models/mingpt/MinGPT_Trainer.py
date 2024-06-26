import torch, torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import wandb, random
from torchenhanced import Trainer
from .MinGPT import MinGPT
from ...dataset.TokenText import TokenTextBOS
from ... import SimpleTokenizer


class MinGPT_Trainer(Trainer):
    def __init__(self, model: MinGPT, train_dataset: TokenTextBOS, valid_dataset : TokenTextBOS,
                 detokenizer :SimpleTokenizer=None, optim: Optimizer = None, scheduler: _LRScheduler = None, 
                 state_save_loc=None, device: str = 'cpu',parallel=None, run_name: str = None, project_name: str = None,
                 run_config: dict ={}):
        super().__init__(model, optim, scheduler, state_save_loc=state_save_loc,parallel=parallel, device=device, run_name=run_name, project_name=project_name, run_config=run_config)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        # For logging :
        self.batch_loss = []

        self.detokenizer= detokenizer

        # Print number of parameters
        print(f"Number of parameters : {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")

    def get_loaders(self,batch_size,num_workers=0):
        t_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
        v_dataloader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

        self.text_table = wandb.Table(columns=["batches","text"])

        return t_dataloader, v_dataloader

    def process_batch(self, batch_data):
        # Check if correct, but should be :
        loss = self.compute_loss(batch_data) #(B,T) loss

        if(self.do_batch_log) :
            wandb.log({'lr' : self.scheduler.get_last_lr()[0]},commit=False)

        return loss.mean()

    def compute_loss(self, batch_data):
        token_in, token_truth = batch_data
        token_in = token_in.to(self.device) # (B, T)
        token_truth = token_truth.to(self.device) # (B, T)

        token_in = self.model.forward(token_in).transpose(1,2) # (B, vsize, T)

        # Check if correct, but should be :
        loss = F.cross_entropy(token_in,token_truth,reduction='none') # (B, T)

        return loss

    def process_batch_valid(self, batch_data):
        # Check if correct, but should be :
        loss = self.compute_loss(batch_data) 


        return loss.mean() # Average loss, (B, T) -> (1,)
        

    def valid_log(self):
        # Completes a sentence and logs it in the table
        data, _ = self.valid_dataset[random.randint(0,len(self.valid_dataset)-1)] # (T,)*2
        data = data[:10].to(self.device) # only keep first 10 tokens
        if(self.parallel_train):
            modello = self.model.module
        else:
            modello = self.model
            
        phrase_out = modello.generate(data[None,:],max_new_tokens=100, do_sample=True).cpu() # (1, 5+300)

        if(self.backwards):
            phrase_out= self.detokenizer.detokenize(torch.flip(phrase_out,dims=[1])) 
        else :
            phrase_out=self.detokenizer.detokenize(phrase_out)

        self.text_table.add_data(f"{self.steps_done/1000:.1f}k",phrase_out) 
        # Trick to be able to update table on the fly... Fucking wandb
        new_table = wandb.Table(
        columns=self.text_table.columns, data=self.text_table.data
        )
        wandb.log({'gen_samples': new_table},commit=False)