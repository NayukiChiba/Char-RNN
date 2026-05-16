"""
训练模块
1. 单epoch训练
2. 验证
3. 训练循环
4. 检查点存取
5. log存储

"""
from src.training.utils import init_hidden, calc_perplexity
import torch
from torch import nn
from tqdm import tqdm
from config.defaults import TrainingParams, ModelParams, DefaultParams
from src.models import create_model
from src.data import train_loader, val_loader, test_loader
optimizer = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
    "AdamW": torch.optim.AdamW,
}


class Trainer:
    def __init__(self):
        self.device = DefaultParams.DEVICE
        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
       
        self.optimizer = optimizer.get(TrainingParams.OPTIMIZER)
        self.criterion = nn.CrossEntropyLoss()

        # 训练超参数
        self.epochs = TrainingParams.EPOCHS
        self.clip = TrainingParams.CLIP_GRAD
        self.lr = TrainingParams.LEARNING_RATE
        self.lr_scheduler = TrainingParams.LR_SCHEDULER
        

        

        # 模型参数
        # 训练时会重新创建并加载词表大小
        self.model = create_model(ModelParams.RNN_TYPE, vocab_size=None)  

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_perplexity": [],
            "val_perplexity": [],
        }
        self.current_epoch = 0

    def train_one_epoch(self):
        """单epoch训练"""

        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"[Train] Epoch {self.current_epoch+1:2d}/{self.epochs}")
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)

            hidden = init_hidden(self.model, x.size(0))
            output, _ = self.model(x, hidden)
            loss = self.criterion(output.transpose(1, 2), y)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "perplexity": f"{calc_perplexity(loss.item()):.4f}"})
        
        return total_loss / len(self.train_loader)
    
