import os
import torch
import utils
from blocks import MLP, build_encoder


class EffectRegressorMLP:

    def __init__(self, opts):
        self.device = torch.device(opts["device"])
        self.encoder1 = build_encoder(opts, 1).to(self.device)
        self.encoder2 = build_encoder(opts, 2).to(self.device)
        self.decoder1 = MLP([opts["code1_dim"] + 3] + [opts["hidden_dim"]] * opts["depth"] + [3]).to(self.device)
        self.decoder2 = MLP([opts["code2_dim"] + opts["code1_dim"]*2] + [opts["hidden_dim"]] * opts["depth"] + [6]).to(self.device)
        
        # Unified optimizer for all 4 networks
        self.optimizer = torch.optim.Adam(lr=opts.get("learning_rate1", 0.0001), 
                                           params=[
                                               {"params": self.encoder1.parameters()},
                                               {"params": self.decoder1.parameters()},
                                               {"params": self.encoder2.parameters()},
                                               {"params": self.decoder2.parameters()}],
                                           amsgrad=True)

        self.criterion = torch.nn.MSELoss()
        self.iteration = 0
        self.save_path = opts["save"]

    def loss1(self, sample):
        obs = sample["observation"].to(self.device)
        effect = sample["effect"].to(self.device)
        action = sample["action"].to(self.device)

        h = self.encoder1(obs)
        h_aug = torch.cat([h, action], dim=-1)
        effect_pred = self.decoder1(h_aug)
        loss = self.criterion(effect_pred, effect)
        return loss

    def loss2(self, sample):
        obs = sample["observation"].to(self.device)
        effect = sample["effect"].to(self.device)

        # Removed with torch.no_grad():
        # Gradients from paired task now flow back into encoder1
        h1 = self.encoder1(obs.reshape(-1, 1, obs.shape[2], obs.shape[3]))
        h1 = h1.reshape(obs.shape[0], -1)
        h2 = self.encoder2(obs)
        h_aug = torch.cat([h1, h2], dim=-1)
        effect_pred = self.decoder2(h_aug)
        loss = self.criterion(effect_pred, effect)
        return loss

    def loss(self, sample_single, sample_paired):
        # Calculate individual losses and sum them up
        l1 = self.loss1(sample_single)
        l2 = self.loss2(sample_paired)
        return l1 + l2

    # Update arguments to take both loaders
    def one_pass_optimize(self, loader1, loader2):
        running_avg_loss = 0.0
        i = 0
        # ZIP THEM HERE: This creates a fresh iterator every epoch
        for sample_single, sample_paired in zip(loader1, loader2):
            self.optimizer.zero_grad()
            total_loss = self.loss(sample_single, sample_paired)
            total_loss.backward()
            self.optimizer.step()
            
            running_avg_loss += total_loss.item()
            self.iteration += 1
            i += 1
            
        return running_avg_loss / max(1, i)

    # Update arguments to take both loaders
    def train(self, epoch, loader1, loader2):
        best_loss = 1e100
        for e in range(epoch):
            # Pass both loaders to the optimizer
            epoch_loss = self.one_pass_optimize(loader1, loader2)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save(self.save_path, "_best")
            print("Epoch: %d, iter: %d, loss: %.4f" % (e+1, self.iteration, epoch_loss))
            self.save(self.save_path, "_last")

    def load(self, path, ext):
        # Load all components at once
        enc1_dict = torch.load(os.path.join(path, "encoder1"+ext+".ckpt"))
        dec1_dict = torch.load(os.path.join(path, "decoder1"+ext+".ckpt"))
        self.encoder1.load_state_dict(enc1_dict)
        self.decoder1.load_state_dict(dec1_dict)
        
        enc2_dict = torch.load(os.path.join(path, "encoder2"+ext+".ckpt"))
        dec2_dict = torch.load(os.path.join(path, "decoder2"+ext+".ckpt"))
        self.encoder2.load_state_dict(enc2_dict)
        self.decoder2.load_state_dict(dec2_dict)

    def save(self, path, ext):
        # Save all components at once using the original naming structure
        enc1_dict = self.encoder1.eval().cpu().state_dict()
        dec1_dict = self.decoder1.eval().cpu().state_dict()
        torch.save(enc1_dict, os.path.join(path, "encoder1"+ext+".ckpt"))
        torch.save(dec1_dict, os.path.join(path, "decoder1"+ext+".ckpt"))
        self.encoder1.train().to(self.device)
        self.decoder1.train().to(self.device)

        enc2_dict = self.encoder2.eval().cpu().state_dict()
        dec2_dict = self.decoder2.eval().cpu().state_dict()
        torch.save(enc2_dict, os.path.join(path, "encoder2"+ext+".ckpt"))
        torch.save(dec2_dict, os.path.join(path, "decoder2"+ext+".ckpt"))
        self.encoder2.train().to(self.device)
        self.decoder2.train().to(self.device)

    def print_model(self):
        print("="*10+"ENCODER 1"+"="*10)
        print(self.encoder1)
        print("parameter count: %d" % utils.get_parameter_count(self.encoder1))
        print("="*10+"DECODER 1"+"="*10)
        print(self.decoder1)
        print("="*10+"ENCODER 2"+"="*10)
        print(self.encoder2)
        print("parameter count: %d" % utils.get_parameter_count(self.encoder2))
        print("="*10+"DECODER 2"+"="*10)
        print(self.decoder2)
        print("="*27)