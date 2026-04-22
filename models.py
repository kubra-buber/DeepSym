import os
import torch
import utils
from blocks import MLP, build_encoder

class EMAVQLayer(torch.nn.Module):
    """
    Vector Quantization Layer using Exponential Moving Average (EMA).
    This prevents codebook collapse by tracking the moving average of assigned vectors 
    rather than relying on unstable gradient descent for the codebook.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # The Codebook
        self.embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()
        # Turn off gradients for the codebook; we update it manually via EMA
        self.embedding.weight.requires_grad = False 

        # Buffers for EMA tracking (saved automatically in state_dict)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())
        
        self.last_vq_loss = 0.0

    def forward(self, inputs):
        flat_inputs = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_inputs, self.embedding.weight.t()))

        # Find closest vector
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view_as(inputs)

        # EMA Training Updates
        if self.training:
            # 1. Track how many inputs were assigned to each cluster
            self.cluster_size.data.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )

            # Laplace smoothing (prevents cluster size from reaching absolute zero)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )

            # 2. Track the sum of the input vectors assigned to each cluster
            embed_sum = torch.matmul(encodings.t(), flat_inputs)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # 3. Update the actual codebook weights to be the average of assigned vectors
            self.embedding.weight.data.copy_(self.embed_avg / cluster_size.unsqueeze(1))

            # Loss: We only need the commitment loss now (forces encoder to stay near codebook)
            e_latent_loss = torch.nn.functional.mse_loss(quantized.detach(), inputs)
            self.last_vq_loss = self.commitment_cost * e_latent_loss
        else:
            self.last_vq_loss = torch.tensor(0.0, device=inputs.device)

        # Straight-through estimator trick
        quantized = inputs + (quantized - inputs).detach()

        return quantized

    def get_indices(self, inputs):
        flat_inputs = inputs.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_inputs, self.embedding.weight.t()))
        return torch.argmin(distances, dim=1)

class EffectRegressorMLP:
    def __init__(self, opts):
        self.device = torch.device(opts["device"])
        
        # Build original encoders
        self.encoder1 = build_encoder(opts, 1).to(self.device)
        self.encoder2 = build_encoder(opts, 2).to(self.device)
        
        # Calculate codebook capacity (2^dim ensures we have the same capacity as binary approach)
        num_emb_1 = 2 ** opts["code1_dim"]
        num_emb_2 = 2 ** opts["code2_dim"]

        # CRITICAL: Replace the STLayer (the last layer, index -1) with our new VQ layer
        self.encoder1[-1] = EMAVQLayer(num_emb_1, opts["code1_dim"]).to(self.device)
        self.encoder2[-1] = EMAVQLayer(num_emb_2, opts["code2_dim"]).to(self.device)

        self.decoder1 = MLP([opts["code1_dim"] + 3] + [opts["hidden_dim"]] * opts["depth"] + [3]).to(self.device)
        self.decoder2 = MLP([opts["code2_dim"] + opts["code1_dim"]*2] + [opts["hidden_dim"]] * opts["depth"] + [6]).to(self.device)
        
        # Optimizers (will automatically pick up the new VQ codebook parameters)
        self.optimizer1 = torch.optim.Adam(lr=opts["learning_rate1"],
                                           params=[
                                               {"params": self.encoder1.parameters()},
                                               {"params": self.decoder1.parameters()}],
                                           amsgrad=True)

        self.optimizer2 = torch.optim.Adam(lr=opts["learning_rate2"],
                                           params=[
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
        
        mse_loss = self.criterion(effect_pred, effect)
        # Add the VQ Codebook commitment loss stored in the layer
        vq_loss = self.encoder1[-1].last_vq_loss
        
        return mse_loss + vq_loss

    def loss2(self, sample):
        obs = sample["observation"].to(self.device)
        effect = sample["effect"].to(self.device)

        with torch.no_grad():
            h1 = self.encoder1(obs.reshape(-1, 1, obs.shape[2], obs.shape[3]))
        h1 = h1.reshape(obs.shape[0], -1)
        
        h2 = self.encoder2(obs)
        h_aug = torch.cat([h1, h2], dim=-1)
        effect_pred = self.decoder2(h_aug)
        
        mse_loss = self.criterion(effect_pred, effect)
        # Add the VQ Codebook commitment loss stored in the layer
        vq_loss = self.encoder2[-1].last_vq_loss
        
        return mse_loss + vq_loss

    def one_pass_optimize(self, loader, level):
        running_avg_loss = 0.0
        for i, sample in enumerate(loader):
            if level == 1:
                self.optimizer1.zero_grad()
                loss = self.loss1(sample)
                loss.backward()
                self.optimizer1.step()
            else:
                self.optimizer2.zero_grad()
                loss = self.loss2(sample)
                loss.backward()
                self.optimizer2.step()
                
            running_avg_loss += loss.item()
            self.iteration += 1
            
        return running_avg_loss / max(1, i)

    def train(self, epoch, loader, level):
        best_loss = 1e100
        for e in range(epoch):
            epoch_loss = self.one_pass_optimize(loader, level)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save(self.save_path, "_best", level)
            print("Epoch: %d, iter: %d, loss: %.4f" % (e+1, self.iteration, epoch_loss))
            self.save(self.save_path, "_last", level)

    def load(self, path, ext, level):
        if level == 1:
            encoder = self.encoder1
            decoder = self.decoder1
        else:
            encoder = self.encoder2
            decoder = self.decoder2

        encoder_dict = torch.load(os.path.join(path, "encoder"+str(level)+ext+".ckpt"))
        decoder_dict = torch.load(os.path.join(path, "decoder"+str(level)+ext+".ckpt"))
        encoder.load_state_dict(encoder_dict)
        decoder.load_state_dict(decoder_dict)

    def save(self, path, ext, level):
        if level == 1:
            encoder = self.encoder1
            decoder = self.decoder1
        else:
            encoder = self.encoder2
            decoder = self.decoder2

        encoder_dict = encoder.eval().cpu().state_dict()
        decoder_dict = decoder.eval().cpu().state_dict()
        torch.save(encoder_dict, os.path.join(path, "encoder"+str(level)+ext+".ckpt"))
        torch.save(decoder_dict, os.path.join(path, "decoder"+str(level)+ext+".ckpt"))
        encoder.train().to(self.device)
        decoder.train().to(self.device)

    def print_model(self, level):
        encoder = self.encoder1 if level == 1 else self.encoder2
        decoder = self.decoder1 if level == 1 else self.decoder2
        print("="*10+"ENCODER"+"="*10)
        print(encoder)
        print("parameter count: %d" % utils.get_parameter_count(encoder))
        print("="*27)
        print("="*10+"DECODER"+"="*10)
        print(decoder)
        print("parameter count: %d" % utils.get_parameter_count(decoder))
        print("="*27)