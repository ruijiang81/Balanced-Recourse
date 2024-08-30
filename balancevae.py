import os
from typing import List, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn

from carla import log
from carla.recourse_methods.autoencoder.save_load import get_home

tf.compat.v1.disable_eager_execution()


class BalanceVariationalAutoencoder(nn.Module):
    def __init__(self, data_name: str, layers: List, mutable_mask, sen_mask = None, gamma = 2):
        """
        Parameters
        ----------
        data_name:
            Name of the dataset, used for the name when saving and loading the model.
        layers:
            List of layer sizes.
        mutable_mask:
            Mask that indicates which feature columns are mutable, and which are immutable. Setting
            all columns to mutable, results in the standard case.
        """
        super(BalanceVariationalAutoencoder, self).__init__()

        if len(layers) < 2:
            raise ValueError(
                "Number of layers have to be at least 2 (input and latent space), and number of neurons bigger than 0"
            )

        self._data_name = data_name
        self._input_dim = layers[0]
        latent_dim = layers[-1]

        # The VAE components
        lst_encoder = []
        for i in range(1, len(layers) - 1):
            lst_encoder.append(nn.Linear(layers[i - 1], layers[i]))
            lst_encoder.append(nn.ReLU())
        encoder = nn.Sequential(*lst_encoder)

        self._mu_enc = nn.Sequential(encoder, nn.Linear(layers[-2], latent_dim))
        self._log_var_enc = nn.Sequential(encoder, nn.Linear(layers[-2], latent_dim))

        # the decoder does use the immutables, so need to increase layer size accordingly.
        # minus the sensitive attribute 
        layers[-1] += np.sum(~mutable_mask) - 1

        lst_decoder = []
        for i in range(len(layers) - 2, 0, -1):
            lst_decoder.append(nn.Linear(layers[i + 1], layers[i]))
            lst_decoder.append((nn.ReLU()))
        decoder = nn.Sequential(*lst_decoder)

        self.mu_dec = nn.Sequential(
            decoder,
            nn.Linear(layers[1], self._input_dim),
            nn.Sigmoid(),
        )

        hidd = 8
        self.sen_clf = nn.Sequential(
            nn.Linear(latent_dim, hidd),
            nn.Sigmoid(), 
            nn.Linear(hidd, hidd),
            nn.Sigmoid(), 
            nn.Linear(hidd, hidd),
            nn.Sigmoid(), 
            nn.Linear(hidd, 1),
            nn.Sigmoid()
            )

        hidd = 8
        self.sen_clf_recon = nn.Sequential(
            nn.Linear(layers[0] + np.sum(~mutable_mask) - 1, hidd),
            nn.Sigmoid(), 
            nn.Linear(hidd, hidd),
            nn.Sigmoid(), 
            nn.Linear(hidd, hidd),
            nn.Sigmoid(), 
            nn.Linear(hidd, 1),
            nn.Sigmoid()
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        self.mutable_mask = mutable_mask
        self.sen_mask = np.array(sen_mask, dtype = bool)
        self.gamma = gamma
    def encode(self, x):
        return self._mu_enc(x), self._log_var_enc(x)

    def decode(self, z):
        return self.mu_dec(z)

    def get_elbo(self, data):
        with torch.no_grad():
            reconstruction, mu, log_var, sens, nonsens = self(data)
            recon_loss = criterion(reconstruction, data)
            kld_loss = self.kld(mu, log_var)
            elbo_loss = recon_loss + beta * kld_loss        
        return elbo_loss

    def __reparametrization_trick(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)  # the Gaussian random noise
        return mu + std * epsilon

    def forward(self, x):

        # split up the input in a mutable and immutable part
        x = x.clone()
        x_mutable = x[:, self.mutable_mask]
        x_immutable = x[:, np.logical_and((~self.mutable_mask),(~self.sen_mask))]
        x_sens = x[:, self.sen_mask]
        # the mutable part gets encoded
        mu_z, log_var_z = self.encode(x_mutable)
        z = self.__reparametrization_trick(mu_z, log_var_z)
        # concatenate the immutable part to the latents and decode both
        z1 = z
        z = torch.cat([z, x_immutable], dim=-1)
        recon = self.decode(z)

        # add the immutable features to the reconstruction
        x[:, self.mutable_mask] = recon
        x_nonsens = x[:, ~self.sen_mask]

        return x, mu_z, log_var_z, x_sens, x_nonsens, z1

    def ae_forward(self, x):

        # split up the input in a mutable and immutable part
        x = x.clone()
        x_mutable = x[:, self.mutable_mask]
        x_immutable = x[:, np.logical_and((~self.mutable_mask),(~self.sen_mask))]
        x_sens = x[:, self.sen_mask]
        # the mutable part gets encoded
        mu_z, log_var_z = self.encode(x_mutable)
        z = mu_z
        # concatenate the immutable part to the latents and decode both
        z = torch.cat([z, x_immutable], dim=-1)
        recon = self.decode(z)

        # add the immutable features to the reconstruction
        x[:, self.mutable_mask] = recon
        x_nonsens = x[:, ~self.sen_mask]

        return x, mu_z, log_var_z, x_sens, x_nonsens


    def predict(self, data):
        return self.forward(data)

    def kld(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD


    def fit(
        self,
        xtrain: Union[pd.DataFrame, np.ndarray],
        kl_weight=0.3,
        lambda_reg=1e-6,
        epochs=5,
        lr=1e-3,
        batch_size=32,
    ):
        if isinstance(xtrain, pd.DataFrame):
            xtrain = xtrain.values

        train_loader = torch.utils.data.DataLoader(
            xtrain, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            #self.parameters(),
            list(self._mu_enc.parameters()) + list(self._log_var_enc.parameters()) + list(self.mu_dec.parameters()),
            lr=lr,
            weight_decay=lambda_reg,
        )

        sen_optimizer = torch.optim.Adam(
            self.sen_clf.parameters(),
            #self.sen_clf_recon.parameters(), 
            lr=1e-2,
            weight_decay=lambda_reg,
        )

        criterion = nn.BCELoss(reduction="sum")
        criterionmse = nn.MSELoss(reduction="sum")

        # Train the VAE with the new prior
        ELBO = np.zeros((epochs, 1))
        sensELBO = np.zeros((epochs, 1))
        log.info("Start training of Variational Autoencoder...")
        gamma = self.gamma
        log.info(f'sen regularization is {gamma}')
        for epoch in range(epochs):

            beta = epoch * kl_weight / epochs
            #beta = 1

            # Initialize the losses
            train_loss = 0
            train_loss_num = 0
            sens_loss0 = 0 
            # Train for all the batches
            for data in train_loader:
                data = data.view(data.shape[0], -1)
                data = data.float()


                reconstruction, mu, log_var, sens, nonsens, z = self(data)
                self.sen_clf.zero_grad()
                self.sen_clf_recon.zero_grad()
                sens_pred = self.sen_clf(z)
                sens_loss = criterion(sens_pred, sens)
                sen_optimizer.zero_grad()
                # Compute the loss
                sens_loss.backward()
                # Update the parameters
                sen_optimizer.step()

                self._mu_enc.zero_grad()
                self._log_var_enc.zero_grad()
                self.mu_dec.zero_grad()
                # forward pass
                for _ in range(1):
                    reconstruction, mu, log_var, sens, nonsens, z = self(data)

                    recon_loss = criterion(reconstruction, data)
                    kld_loss = self.kld(mu, log_var)
                    sens_pred = self.sen_clf(z)
                    self.sen_clf.zero_grad()
                    self.sen_clf_recon.zero_grad()
                    sens_entropy =  - sens_pred * torch.log(sens_pred+1e-10) - (1-sens_pred) * torch.log(1 - sens_pred+1e-10)
                    elbo_loss = recon_loss + beta * kld_loss
                    loss = elbo_loss - gamma * sens_entropy.sum()

                    # Update the parameters
                    optimizer.zero_grad()
                    # Compute the loss
                    loss.backward()
                    # Update the parameters
                    optimizer.step()
                # Collect the ways
                train_loss += elbo_loss.item()
                train_loss_num += 1
                sens_loss0 += sens_entropy.mean().item()


            ELBO[epoch] = train_loss / train_loss_num
            #sensELBO[epoch] = sens_loss0 / train_loss_num
            sensELBO[epoch] = sens_entropy.mean().detach().numpy()
            if epoch % 10 == 0:
                log.info(
                    "[Epoch: {}/{}] [objective: {:.3f}]".format(
                        epoch, epochs, ELBO[epoch, 0]
                    )
                )
                log.info(
                    "[Epoch: {}/{}] [sens objective: {:.3f}]".format(
                        #epoch, epochs, sensELBO[epoch, 0]
                        epoch, epochs, sens_loss.mean().item()
                    )
                )
            ELBO_train = ELBO[epoch, 0].round(2)
            log.info("[ELBO train: " + str(ELBO_train) + "]")

        self.save()
        log.info("... finished training of Variational Autoencoder.")

        self.eval()

    def fit_ae(
        self,
        xtrain: Union[pd.DataFrame, np.ndarray],
        kl_weight=0.3,
        lambda_reg=1e-6,
        epochs=5,
        lr=1e-3,
        batch_size=32,
    ):
        if isinstance(xtrain, pd.DataFrame):
            xtrain = xtrain.values

        train_loader = torch.utils.data.DataLoader(
            xtrain, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            #self.parameters(),
            list(self._mu_enc.parameters()) + list(self._log_var_enc.parameters()) + list(self.mu_dec.parameters()),
            lr=lr,
            weight_decay=lambda_reg,
        )

        sen_optimizer = torch.optim.Adam(
            #self.parameters(),
            self.sen_clf.parameters(),
            lr=1e-3,
            weight_decay=lambda_reg,
        )

        criterion = nn.BCELoss(reduction="sum")

        # Train the VAE with the new prior
        ELBO = np.zeros((epochs, 1))
        sensELBO = np.zeros((epochs, 1))
        log.info("Start training of Variational Autoencoder...")
        gamma = self.gamma
        log.info(f'sen regularization is {gamma}')
        for epoch in range(epochs):

            beta = epoch * kl_weight / epochs
            #beta = 1

            # Initialize the losses
            train_loss = 0
            train_loss_num = 0
            sens_loss0 = 0 
            # Train for all the batches
            for data in train_loader:
                data = data.view(data.shape[0], -1)
                data = data.float()


                reconstruction, mu, log_var, sens, nonsens = self(data)
                self._mu_enc.zero_grad()
                self._log_var_enc.zero_grad()
                self.mu_dec.zero_grad()
                #self.sen_clf.zero_grad()
                sens_pred = self.sen_clf(nonsens)
                sens_loss = criterion(sens_pred, sens)
                sen_optimizer.zero_grad()
                # Compute the loss
                sens_loss.backward()
                # Update the parameters
                sen_optimizer.step()

                reconstruction, mu, log_var, sens, nonsens = self.ae_forward(data)

                recon_loss = criterion(reconstruction, data)
                kld_loss = self.kld(mu, log_var)
                sens_pred = self.sen_clf(nonsens)
                self.sen_clf.zero_grad()
                #sens_loss = criterion(sens_pred, sens)
                sens_entropy =  - sens_pred * torch.log(sens_pred + 1e-6) - (1-sens_pred) * torch.log(1 - sens_pred + 1e-6)
                elbo_loss = recon_loss 
                loss = elbo_loss - gamma * sens_entropy.mean()

                optimizer.zero_grad()
                # Compute the loss
                loss.backward()
                # Update the parameters
                optimizer.step()


                # Collect the ways
                train_loss += elbo_loss.item()
                train_loss_num += 1
                sens_loss0 += sens_entropy.mean().item()

            ELBO[epoch] = train_loss / train_loss_num
            sensELBO[epoch] = sens_loss0 / train_loss_num
            if epoch % 10 == 0:
                log.info(
                    "[Epoch: {}/{}] [objective: {:.3f}]".format(
                        epoch, epochs, ELBO[epoch, 0]
                    )
                )
                log.info(
                    "[Epoch: {}/{}] [sens objective: {:.3f}]".format(
                        epoch, epochs, sensELBO[epoch, 0]
                    )
                )
            ELBO_train = ELBO[epoch, 0].round(2)
            log.info("[ELBO train: " + str(ELBO_train) + "]")

        self.save()
        log.info("... finished training of Autoencoder.")

        self.eval()

    def load(self, input_shape):
        cache_path = get_home()

        load_path = os.path.join(
            cache_path,
            "{}_{}.{}".format(self._data_name, input_shape, "pt"),
        )

        self.load_state_dict(torch.load(load_path))

        self.eval()

        return self

    def save(self):
        cache_path = get_home()

        save_path = os.path.join(
            cache_path,
            "{}_{}.{}".format(self._data_name, self._input_dim, "pt"),
        )

        torch.save(self.state_dict(), save_path)
