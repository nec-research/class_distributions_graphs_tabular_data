"""
  On Class Distributions Induced by Nearest Neighbor Graphs for Node Classification of Tabular Data

  File:     class_separator.py
  Authors:  Federico Errica (federico.errica@neclab.eu)

NEC Laboratories Europe GmbH, Copyright (c) 2023, All rights reserved.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.

       PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT

ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY

BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution
or non-profit organization or self (called "Licensee" or "You" in this
Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
Agreement).  All rights not specifically granted to you in this Agreement
are reserved for Licensor.

RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
ownership of any copy of the Software (as defined below) licensed under this
Agreement and hereby grants to Licensee a personal, non-exclusive,
non-transferable license to use the Software for noncommercial research
purposes, without the right to sublicense, pursuant to the terms and
conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
Agreement, the term "Software" means (i) the actual copy of all or any
portion of code for program routines made accessible to Licensee by Licensor
pursuant to this Agreement, inclusive of backups, updates, and/or merged
copies permitted hereunder or subsequently supplied by Licensor,  including
all or any file structures, programming instructions, user interfaces and
screen formats and sequences as well as any and all documentation and
instructions related to it, and (ii) all or any derivatives and/or
modifications created or made by You to any of the items specified in (i).

CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
proprietary to Licensor, and as such, Licensee agrees to receive all such
materials and to use the Software only in accordance with the terms of this
Agreement.  Licensee agrees to use reasonable effort to protect the Software
from unauthorized use, reproduction, distribution, or publication. All
publication materials mentioning features or use of this software must
explicitly include an acknowledgement the software was developed by NEC
Laboratories Europe GmbH.

COPYRIGHT: The Software is owned by Licensor.

PERMITTED USES:  The Software may be used for your own noncommercial
internal research purposes. You understand and agree that Licensor is not
obligated to implement any suggestions and/or feedback you might provide
regarding the Software, but to the extent Licensor does so, you are not
entitled to any compensation related thereto.

DERIVATIVES: You may create derivatives of or make modifications to the
Software, however, You agree that all and any such derivatives and
modifications will be owned by Licensor and become a part of the Software
licensed to You under this Agreement.  You may only use such derivatives and
modifications for your own noncommercial internal research purposes, and you
may not otherwise use, distribute or copy such derivatives and modifications
in violation of this Agreement.

BACKUPS:  If Licensee is an organization, it may make that number of copies
of the Software necessary for internal noncommercial use at a single site
within its organization provided that all information appearing in or on the
original labels, including the copyright and trademark notices are copied
onto the labels of the copies.

USES NOT PERMITTED:  You may not distribute, copy or use the Software except
as explicitly permitted herein. Licensee has not been granted any trademark
license as part of this Agreement.  Neither the name of NEC Laboratories
Europe GmbH nor the names of its contributors may be used to endorse or
promote products derived from this Software without specific prior written
permission.

You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
whole or in part, or provide third parties access to prior or present
versions (or any parts thereof) of the Software.

ASSIGNMENT: You may not assign this Agreement or your rights hereunder
without the prior written consent of Licensor. Any attempted assignment
without such consent shall be null and void.

TERM: The term of the license granted by this Agreement is from Licensee's
acceptance of this Agreement by downloading the Software or by using the
Software until terminated as provided below.

The Agreement automatically terminates without notice if you fail to comply
with any provision of this Agreement.  Licensee may terminate this Agreement
by ceasing using the Software.  Upon any termination of this Agreement,
Licensee will delete any and all copies of the Software. You agree that all
provisions which operate to protect the proprietary rights of Licensor shall
remain in force should breach occur and that the obligation of
confidentiality described in this Agreement is binding in perpetuity and, as
such, survives the term of the Agreement.

FEE: Provided Licensee abides completely by the terms and conditions of this
Agreement, there is no fee due to Licensor for Licensee's use of the
Software in accordance with this Agreement.

DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
RELATED MATERIALS.

SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
provided as part of this Agreement.

EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
permitted under applicable law, Licensor shall not be liable for direct,
indirect, special, incidental, or consequential damages or lost profits
related to Licensee's use of and/or inability to use the Software, even if
Licensor is advised of the possibility of such damage.

EXPORT REGULATION: Licensee agrees to comply with any and all applicable
export control laws, regulations, and/or other laws related to embargoes and
sanction programs administered by law.

SEVERABILITY: If any provision(s) of this Agreement shall be held to be
invalid, illegal, or unenforceable by a court or other tribunal of competent
jurisdiction, the validity, legality and enforceability of the remaining
provisions shall not in any way be affected or impaired thereby.

NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
or remedy under this Agreement shall be construed as a waiver of any future
or other exercise of such right or remedy by Licensor.

GOVERNING LAW: This Agreement shall be construed and enforced in accordance
with the laws of Germany without reference to conflict of laws principles.
You consent to the personal jurisdiction of the courts of this country and
waive their rights to venue outside of Germany.

ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
entire agreement between Licensee and Licensor as to the matter set forth
herein and supersedes any previous agreements, understandings, and
arrangements between the parties relating hereto.

       THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""

import torch
from typing import Tuple, List, Optional

from torch.distributions import (
    MixtureSameFamily,
    Categorical,
    Independent,
    Normal, MultivariateNormal,
)
from torch.nn.functional import softplus, normalize
from torch.nn.parameter import Parameter

from SED import SED, SED_full
from M_c import M_c
from mixture import sum_of_mixtures, mixture_times_constant


STD_MIN = 1e-0


class ClassSeparator(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_mixtures: int,
        num_features: int,
        use_full_covariance: bool,
    ):
        """
        Instantiates a model that tries to maximize the (|C|*(|C|+1))/2 - |C|
        interactions between pairs of distinct classes c,c' un this way
            \sum_{c,c'} SED(p(h|c), p(h|c')) - SED(p(x|c), p(x|c'))

        :param num_classes: the parameter |C| in the paper
        :param num_mixtures: the parameter |M| in the paper
        :param num_features: the parameter D in the paper
        :param use_full_covariance: whether to use the full covariance matrix
        """
        super().__init__()
        self.C = num_classes
        self.M = num_mixtures
        self.D = num_features

        self.use_full_covariance = use_full_covariance

        self.k = Parameter(
            torch.ones(1, dtype=torch.double), requires_grad=True
        )

        self.epsilon = Parameter(
            torch.ones(1, dtype=torch.double), requires_grad=True
        )  # 1 as epsilon

        self.sigma = Parameter(
            torch.ones(1, dtype=torch.double), requires_grad=True
        )  # 1 as epsilon

        self.prior = Parameter(
            torch.rand(self.C, dtype=torch.double), requires_grad=True
        )
        # normalize prior weights
        self.prior.data = self.prior.data / self.prior.data.sum(
            0, keepdims=True
        )

        self.weights = Parameter(
            torch.rand(self.C, self.M, dtype=torch.double)
        )
        # normalize mixture weights
        self.weights.data = self.weights.data / self.weights.data.sum(
            1, keepdims=True
        )

        self.mean = Parameter(
            torch.rand(self.C, self.M, self.D, dtype=torch.double) * 100.0
        )

        if not self.use_full_covariance:
            self.std = Parameter(
                torch.rand(self.C, self.M, self.D, dtype=torch.double) * 100.0
            )
        else:
            self.std = Parameter(
                torch.rand(self.C, self.M, self.D, self.D, dtype=torch.double)
                * 100.0
            )

    def get_parameters(self):
        epsilon = torch.abs(self.epsilon)
        prior = torch.abs(self.prior)
        prior = normalize(prior, p=1, dim=-1)
        weights = torch.abs(self.weights)
        weights = normalize(weights, p=1, dim=-1)
        mean = self.mean
        std = torch.abs(self.std) + STD_MIN
        return prior, weights, mean, std, epsilon

    def compute_SED_X(self, c, c_prime) -> torch.Tensor:
        """
        Compute the SED(H_c,H_c')

        :param c: the first class
        :param c_prime: the second class
        :return: a tensor with a single value representing SED(H_c,H_c')
        """
        _, weights, mean, std, _ = self.get_parameters()

        # SED for X
        if not self.use_full_covariance:
            sed_x = SED(
                mixture_weights_x=weights[c],
                gaussian_mean_x=mean[c],
                gaussian_std_x=std[c],
                mixture_weights_y=weights[c_prime],
                gaussian_mean_y=mean[c_prime],
                gaussian_std_y=std[c_prime],
            )
        else:
            sigma = torch.matmul(std, std.transpose(2, 3))

            sed_x = SED_full(
                mixture_weights_x=weights[c],
                gaussian_mean_x=mean[c],
                gaussian_sigma_x=sigma[c],
                mixture_weights_y=weights[c_prime],
                gaussian_mean_y=mean[c_prime],
                gaussian_sigma_y=sigma[c_prime],
            )
        return sed_x

    def compute_SED_H(self, c, c_prime) -> torch.Tensor:
        """
        Compute the SED(H_c,H_c')

        :param c: the first class
        :param c_prime: the second class
        :return: a tensor with a single value representing SED(H_c,H_c')
        """
        _, weights, mean, std, _ = self.get_parameters()

        lambda_epsilon = (
            torch.ones(
                self.C,
                self.M,
                self.D,
                dtype=torch.double,
                device=self.sigma.device,
            )
            * torch.pow(self.sigma, 2)
            / (torch.relu(self.k) + 1)
        )

        if not self.use_full_covariance:
            h_std = std + torch.sqrt(lambda_epsilon)

            # SED for H
            sed_h = SED(
                mixture_weights_x=weights[c],
                gaussian_mean_x=mean[c],
                gaussian_std_x=h_std[c],
                mixture_weights_y=weights[c_prime],
                gaussian_mean_y=mean[c_prime],
                gaussian_std_y=h_std[c_prime],
            )

        else:
            sigma = torch.matmul(std, std.transpose(2, 3))
            h_sigma = sigma + torch.diag_embed(lambda_epsilon)

            # SED for H
            sed_h = SED_full(
                mixture_weights_x=weights[c],
                gaussian_mean_x=mean[c],
                gaussian_sigma_x=h_sigma[c],
                mixture_weights_y=weights[c_prime],
                gaussian_mean_y=mean[c_prime],
                gaussian_sigma_y=h_sigma[c_prime],
            )

        return sed_h

    def compute_CCNS_LB(self,
                        remove_std_min: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the lower bound of the CCNS for two classes c,c'

        :param remove_std_min: whether or not to remove the STD_MIN
            contribution from std

        :return: a CxC matrix with the lower bound of CCNS between pairs of
        classes and a CxC matrix with M_c'(c)
        """
        prior, weights, mean, std, epsilon = self.get_parameters()

        if remove_std_min:
            std.data -= STD_MIN

        # compute normalized M_c'(c) for every c'
        m_cc = M_c(
            epsilon=epsilon,
            prior=prior,
            p_m_given_c=weights,
            gaussian_mean=mean,
            gaussian_std=std,
            normalize=True,
        )
        lb = torch.cdist(m_cc, m_cc, p=2)  # |C|x|C|
        return lb, m_cc

    def forward(
        self,
    ) -> Tuple[
        List[Tuple[torch.Tensor, int, int]], torch.Tensor, torch.Tensor
    ]:
        """
        Computes the two terms of the inequality in Th. 3.7, so that we can
        optimize wrt their difference

        :return: a list of tuples of the form (SED(hc,hc') - SED(xc,xc'), c, c')
            for every c != c'
        """
        sed_tuples = []

        if not self.use_full_covariance:
            lb_ccns, m_cc = self.compute_CCNS_LB()
        else:
            lb_ccns, m_cc = None, None

        # compute pairwise SED for x and h
        for c_prime in range(self.C):
            for c in range(c_prime + 1, self.C):
                # SED for X
                sed_x = self.compute_SED_X(c, c_prime)
                sed_h = self.compute_SED_H(c, c_prime)

                sed_tuples.append(
                    (
                        sed_h - sed_x,
                        c_prime,
                        c,
                        sed_h.detach().cpu(),
                        sed_x.detach().cpu(),
                    )
                )

        return sed_tuples, lb_ccns, m_cc

    def get_log_likelihood_original_distribution(
        self, input_tensor: torch.Tensor, class_id: int = None
    ) -> torch.Tensor:
        """
        Given a set of points, it returns the likelihood of the HNB. If a class
        id is specified, it returns the likelihood of the GMM associated with
        a specific class only (thus ignoring the learned prior distribution)
        :param input_tensor: the data wrt which compute the likelihood
        :param class_id: Optional parameter to specify one of the classes of
            the HNB
        :return: a tensor containing the log-likelihood for each input point
        """
        prior, weights, mean, std, _ = self.get_parameters()

        raise NotImplementedError("Full Covariance case to be implemented")
        # TODO IF FULL COVARIANCE WE NEED TO TAKE IT INTO ACCOUNT

        gmm_class = []
        for c in range(self.C):
            gmm_class.append(
                MixtureSameFamily(
                    Categorical(weights[c]),
                    Independent(Normal(loc=mean[c], scale=std[c]), 1),
                )
            )
            # print(gmm_class[-1].batch_shape, gmm_class[-1].event_shape)

        if class_id is not None:
            return gmm_class[class_id].log_prob(input_tensor)
        else:
            log_likelihood = torch.stack(
                [
                    prior[c].log() + gmm_class[c].log_prob(input_tensor)
                    for c in range(self.C)
                ],
                dim=1,
            )
            log_likelihood = torch.logsumexp(log_likelihood, dim=1)
            return log_likelihood

    def get_log_likelihood_embedding_distribution(
        self, input_tensor: torch.Tensor, class_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Given a set of points, it returns the likelihood of the HNB
        corresponding to the embedding space of a DGN. If a class
        id is specified, it returns the likelihood of the GMM associated with
        a specific class only (thus ignoring the learned prior distribution)
        :param input_tensor: the data wrt which compute the likelihood
        :param class_id: optional parameter to specify to which class the
            neighboring embedding distribution is supposed to belong to
        :return: a tensor containing the log-likelihood for each input point
        """

        prior, weights, mean, std, _ = self.get_parameters()

        raise NotImplementedError("Full Covariance case to be implemented")
        # TODO IF FULL COVARIANCE WE NEED TO TAKE IT INTO ACCOUNT

        lambda_epsilon = (
            torch.ones(
                self.C,
                self.M,
                self.D,
                dtype=torch.double,
                device=self.sigma.device,
            )
            * torch.pow(self.sigma, 2)
            / (softplus(self.k) + 1)
        )
        h_std = std + torch.sqrt(lambda_epsilon)

        gmm_emb_class = []
        for c in range(self.C):
            gmm_emb_class.append(
                MixtureSameFamily(
                    Categorical(weights[c]),
                    Independent(Normal(loc=mean[c], scale=h_std[c]), 1),
                )
            )
            # print(gmm_class[-1].batch_shape, gmm_class[-1].event_shape)

        if class_id is not None:
            return gmm_emb_class[class_id].log_prob(input_tensor)
        else:
            log_likelihood = torch.stack(
                [
                    prior[c].log() + gmm_emb_class[c].log_prob(input_tensor)
                    for c in range(self.C)
                ],
                dim=1,
            )
            log_likelihood = torch.logsumexp(log_likelihood, dim=1)
            return log_likelihood

    def sample_data(
            self,
            num_samples: int = 10000,
            remove_std_min: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples data points from the learned distribution.
        :param num_samples: the number of total points to sample
        :param remove_std_min: whether or not to remove the STD_MIN
            contribution from std
        :return: tuple with X and y tensors
        """
        prior, weights, mean, std, _ = self.get_parameters()

        if remove_std_min:
            std.data -= STD_MIN

        class_choice = Categorical(prior)
        y = class_choice.sample(sample_shape=(num_samples,))

        if self.use_full_covariance and self.D > 1:
            sigma = torch.matmul(std, std.transpose(2, 3))

            # print(mean.shape, sigma.shape)
            # print(mean[y].shape, sigma[y].shape, weights[y].shape)

            gmm = MixtureSameFamily(
                Categorical(weights[y]),
                Independent(MultivariateNormal(loc=mean[y],
                                               covariance_matrix=sigma[y]), 0),
            )
            # print(gmm.batch_shape, gmm.event_shape)
        else:
            gmm = MixtureSameFamily(
                Categorical(weights[y]),
                Independent(Normal(loc=mean[y], scale=std[y]), 1),
            )

        X = gmm.sample()

        return X, y
