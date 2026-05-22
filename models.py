import os
import torch
import utils
from blocks import MLP, build_encoder

class DynamicEMAVQLayer(torch.nn.Module):
    """
    Dynamic Vector Quantization Layer.
    Starts with 1 cluster and dynamically grows the codebook when it encounters 
    inputs that exceed the `surprise_threshold`.
    """
    def __init__(self, max_embeddings, embedding_dim, surprise_threshold=1.0, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_embeddings = max_embeddings
        self.surprise_threshold = surprise_threshold
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Track how many embeddings are currently active (Saved in .ckpt)
        self.register_buffer('active_embeddings', torch.tensor(1))

        # Allocate the maximum possible codebook, but we only use [:active_embeddings]
        self.embedding = torch.nn.Embedding(self.max_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()
        self.embedding.weight.requires_grad = False 

        # Buffers for EMA tracking
        self.register_buffer('cluster_size', torch.zeros(max_embeddings))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())
        
        # Initialize the very first cluster size so it isn't zeroed out
        self.cluster_size[0] = 1.0
        
        self.last_vq_loss = 0.0

    def forward(self, inputs):
        flat_inputs = inputs.view(-1, self.embedding_dim)
        active_k = self.active_embeddings.item()

        # 1. Calculate distances against currently ACTIVE embeddings only
        active_weights = self.embedding.weight[:active_k]
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True)
                    + torch.sum(active_weights**2, dim=1)
                    - 2 * torch.matmul(flat_inputs, active_weights.t()))

        # 2. Dynamic Growth Phase (Only during training)
        if self.training and active_k < self.max_embeddings:
            # Find the minimum distance for each input in the batch
            min_dists, _ = torch.min(distances, dim=1)
            # Find the most "surprising" input in the entire batch
            max_min_dist, outlier_idx = torch.max(min_dists, dim=0)

            # If the outlier is too strange, spawn a new cluster center!
            if max_min_dist > self.surprise_threshold:
                new_idx = active_k
                
                # Initialize the new cluster exactly at the outlier's position
                self.embedding.weight.data[new_idx] = flat_inputs[outlier_idx].detach()
                self.cluster_size[new_idx] = 1.0 
                self.embed_avg.data[new_idx] = flat_inputs[outlier_idx].detach()
                
                # Increment active clusters
                self.active_embeddings += 1
                active_k = self.active_embeddings.item()
                active_weights = self.embedding.weight[:active_k]
                print(f"[VQ Layer] Surprise > {self.surprise_threshold}! Codebook grew to size: {active_k}")

                # Recompute distances with the new codebook vector included
                distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True)
                            + torch.sum(active_weights**2, dim=1)
                            - 2 * torch.matmul(flat_inputs, active_weights.t()))

        # 3. Find the closest vector for standard quantization
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], active_k, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, active_weights).view_as(inputs)

        # 4. Standard EMA Training Updates for active clusters
        if self.training:
            self.cluster_size[:active_k].data.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )

            n = self.cluster_size[:active_k].sum()
            cluster_size = (
                (self.cluster_size[:active_k] + self.epsilon)
                / (n + active_k * self.epsilon)
                * n
            )

            embed_sum = torch.matmul(encodings.t(), flat_inputs)
            self.embed_avg[:active_k].data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            self.embedding.weight.data[:active_k] = self.embed_avg[:active_k] / cluster_size.unsqueeze(1)

            e_latent_loss = torch.nn.functional.mse_loss(quantized.detach(), inputs)
            self.last_vq_loss = self.commitment_cost * e_latent_loss
        else:
            self.last_vq_loss = torch.tensor(0.0, device=inputs.device)

        quantized = inputs + (quantized - inputs).detach()

        return quantized

    def get_indices(self, inputs):
        flat_inputs = inputs.view(-1, self.embedding_dim)
        active_k = self.active_embeddings.item()
        active_weights = self.embedding.weight[:active_k]
        
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True)
                    + torch.sum(active_weights**2, dim=1)
                    - 2 * torch.matmul(flat_inputs, active_weights.t()))
        return torch.argmin(distances, dim=1)


class EffectRegressorMLP:
    def __init__(self, opts):
        self.device = torch.device(opts["device"])
        
        self.encoder1 = build_encoder(opts, 1).to(self.device)
        self.encoder2 = build_encoder(opts, 2).to(self.device)
        
        # Max capacity based on binary dimensions (Acts as a ceiling, not a requirement)
        max_emb_1 = 2 ** opts.get("code1_dim", 6) # Default safe ceiling: 64
        max_emb_2 = 2 ** opts.get("code2_dim", 6)

        # Pull surprise thresholds from opts.yaml, fallback to 1.0
        thresh_1 = opts.get("surprise_threshold_1", 1.0)
        thresh_2 = opts.get("surprise_threshold_2", 1.0)

        # Swap to the Dynamic Layer
        self.encoder1[-1] = DynamicEMAVQLayer(max_emb_1, opts["code1_dim"], surprise_threshold=thresh_1).to(self.device)
        self.encoder2[-1] = DynamicEMAVQLayer(max_emb_2, opts["code2_dim"], surprise_threshold=thresh_2).to(self.device)

        self.decoder1 = MLP([opts["code1_dim"] + 3] + [opts["hidden_dim"]] * opts["depth"] + [3]).to(self.device)
        self.decoder2 = MLP([opts["code2_dim"] + opts["code1_dim"]*2] + [opts["hidden_dim"]] * opts["depth"] + [6]).to(self.device)
        
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
        
        raw_mse = torch.nn.functional.mse_loss(effect_pred, effect, reduction='none')
        weights = torch.tensor([1.0, 1.0, 10.0], device=self.device)
        mse_loss = (raw_mse * weights).mean()

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
        
        raw_mse = torch.nn.functional.mse_loss(effect_pred, effect, reduction='none')
        weights = torch.tensor([1.0, 1.0, 5.0, 1.0, 1.0, 1.0], device=self.device)
        mse_loss = (raw_mse * weights).mean()

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