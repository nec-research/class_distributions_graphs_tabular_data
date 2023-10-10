"""
  On Class Distributions Induced by Nearest Neighbor Graphs for Node Classification of Tabular Data

  File:     SED.py
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
from torch.distributions import *


def SED(
    mixture_weights_x: torch.Tensor,
    gaussian_mean_x: torch.Tensor,
    gaussian_std_x: torch.Tensor,
    mixture_weights_y: torch.Tensor,
    gaussian_mean_y: torch.Tensor,
    gaussian_std_y: torch.Tensor,
    log_space: bool = True,
) -> torch.Tensor:
    """
    Computes the Squared Error Distance (SED) divergence between two
    multivariate Gaussian mixture models.

    IMPORTANT: thi function assumes that the dimensions of the Gaussian are
    independent, which is the case of our main results.
    Use SED_full() if you want to work with full covariance matrices.

    This allows us to save a lot of memory (a factor D, which means a lot
    when the feature space is highly dimensional).


    :param mixture_weights_x: a tensor of shape (M) with the mixing weights
        for each class associated with the first distribution
    :param gaussian_mean_x: a tensor of shape (M, D) with the means
        for each class and mixture associated with the first distribution
    :param gaussian_std_x: a tensor of shape (M, D) with the std
        for each class and mixture associated with the first distribution
    :param mixture_weights_y: a tensor of shape (M') with the mixing weights
        for each class associated with the second distribution
    :param gaussian_mean_y: a tensor of shape (M', D) with the means
        for each class and mixture associated with the second distribution
    :param gaussian_std_y: a tensor of shape (M', D) with the std
        for each class and mixture associated with the second distribution
    :param log_space: whether to use log_space computations until the
        very end or not. IF TRUE, the weights are assumed to be already in log
        space.
    :return:
    """
    M_x, D = gaussian_mean_x.shape[0], gaussian_mean_x.shape[1]
    M_y = gaussian_mean_y.shape[0]
    assert D == gaussian_mean_y.shape[1]

    if log_space:
        alphas_alphas = mixture_weights_x.unsqueeze(
            1
        ) + mixture_weights_x.unsqueeze(
            0
        )  # (M,1) + (1,M) = (M,M)

        betas_betas = mixture_weights_y.unsqueeze(
            1
        ) + mixture_weights_y.unsqueeze(
            0
        )  # (M',1) + (1,M') = (M',M')

        alphas_betas = mixture_weights_x.unsqueeze(
            1
        ) + mixture_weights_y.unsqueeze(
            0
        )  # (M,1) + (1,M') = (M,M')

    else:
        alphas_alphas = mixture_weights_x.unsqueeze(
            1
        ) * mixture_weights_x.unsqueeze(
            0
        )  # (M,1) x (1,M) = (M,M)

        betas_betas = mixture_weights_y.unsqueeze(
            1
        ) * mixture_weights_y.unsqueeze(
            0
        )  # (M',1) x (1,M') = (M',M')

        alphas_betas = mixture_weights_x.unsqueeze(
            1
        ) * mixture_weights_y.unsqueeze(
            0
        )  # (M,1) x (1,M') = (M,M')

    # fixing C and D, repeat a vector of shape Mx1 across M columns
    args_a = gaussian_mean_x.unsqueeze(1).repeat(1, M_x, 1)  # (M,M,D)

    # fixing C and D, repeat a vector of shape 1xM across M rows
    mean_a = gaussian_mean_x.unsqueeze(0).repeat(M_x, 1, 1)  # (M,M,D)

    std_a = gaussian_std_x.unsqueeze(1) + gaussian_std_x.unsqueeze(
        0
    )  # (M,M,D)

    # fixing C and D, repeat a vector of shape Mx1 across M columns
    args_b = gaussian_mean_y.unsqueeze(1).repeat(1, M_y, 1)  # (M',M',D)

    # fixing C and D, repeat a vector of shape 1xM across M rows
    mean_b = gaussian_mean_y.unsqueeze(0).repeat(M_y, 1, 1)  # (M',M',D)

    std_b = gaussian_std_y.unsqueeze(1) + gaussian_std_y.unsqueeze(
        0
    )  # (M',M',D)

    # fixing C and D, repeat a vector of shape Mx1 across M columns
    args_c = gaussian_mean_x.unsqueeze(1).repeat(1, M_y, 1)  # (M,M',D)

    # fixing C and D, repeat a vector of shape 1xM across M rows
    mean_c = gaussian_mean_y.unsqueeze(0).repeat(M_x, 1, 1)  # (M,M',D)

    std_c = gaussian_std_x.unsqueeze(1) + gaussian_std_y.unsqueeze(
        0
    )  # (M,1,D) + ((1,M',D) = (M,M',D)

    """
    Each term in SED defines a mixture of multivariate Gaussians with
    M*M' components. Therefore we reshape
    """
    alphas_alphas_ = alphas_alphas.reshape(-1)
    betas_betas_ = betas_betas.reshape(-1)
    alphas_betas_ = alphas_betas.reshape(-1)
    args_a = args_a.reshape(-1, D)
    args_b = args_b.reshape(-1, D)
    args_c = args_c.reshape(-1, D)
    mean_a = mean_a.reshape(-1, D)
    mean_b = mean_b.reshape(-1, D)
    mean_c = mean_c.reshape(-1, D)
    std_a = std_a.reshape(-1, D)
    std_b = std_b.reshape(-1, D)
    std_c = std_c.reshape(-1, D)

    comp_a = Independent(Normal(loc=mean_a, scale=std_a), 1)

    comp_b = Independent(Normal(loc=mean_b, scale=std_b), 1)

    comp_c = Independent(Normal(loc=mean_c, scale=std_c), 1)

    if log_space:
        sed = (
            torch.logsumexp(
                alphas_alphas_ + comp_a.log_prob(args_a), dim=0
            ).exp()
            + torch.logsumexp(
                betas_betas_ + comp_b.log_prob(args_b), dim=0
            ).exp()
            - 2
            * torch.logsumexp(
                alphas_betas_ + comp_c.log_prob(args_c), dim=0
            ).exp()
        )
    else:
        sed = (
            (alphas_alphas_ * (comp_a.log_prob(args_a).exp())).sum()
            + (betas_betas_ * (comp_b.log_prob(args_b).exp())).sum()
            - 2 * (alphas_betas_ * (comp_c.log_prob(args_c).exp())).sum()
        )

    return sed


def test_SED_1():
    """
    If you permute the mixtures the result should still be 0.
    """
    M = 5
    D = 3
    gaussian_mean_x = torch.rand((M, D)).double()
    gaussian_std_x = torch.rand((M, D)).double()
    weights_x = torch.rand(M).double()

    for logspace in [False, True]:
        sed = SED(
            mixture_weights_x=weights_x.log() if logspace else weights_x,
            gaussian_mean_x=gaussian_mean_x,
            gaussian_std_x=gaussian_std_x,
            mixture_weights_y=weights_x.log() if logspace else weights_x,
            gaussian_mean_y=gaussian_mean_x,
            gaussian_std_y=gaussian_std_x,
            log_space=logspace,
        )
        assert torch.allclose(sed, torch.zeros(1).double())

        trials = 50
        for t in range(trials):
            permutation = torch.randperm(M)

            gaussian_mean_y = gaussian_mean_x[permutation]
            gaussian_std_y = gaussian_std_x[permutation]
            weights_y = weights_x[permutation]

            sed = SED(
                mixture_weights_x=weights_x.log() if logspace else weights_x,
                gaussian_mean_x=gaussian_mean_x,
                gaussian_std_x=gaussian_std_x,
                mixture_weights_y=weights_y.log() if logspace else weights_y,
                gaussian_mean_y=gaussian_mean_y,
                gaussian_std_y=gaussian_std_y,
                log_space=logspace,
            )

            assert torch.allclose(sed, torch.zeros(1).double())


def test_SED_2():
    """
    Test in the case of degenerate shapes
    :return:
    """
    M = 1
    D = 1
    gaussian_mean_x = torch.rand((M, D)).double()
    gaussian_std_x = torch.rand((M, D)).double()
    weights_x = torch.rand(M).double()

    for logspace in [False, True]:

        trials = 50
        for t in range(trials):
            permutation = torch.randperm(M)

            gaussian_mean_y = gaussian_mean_x[permutation]
            gaussian_std_y = gaussian_std_x[permutation]
            weights_y = weights_x[permutation]

            sed = SED(
                mixture_weights_x=weights_x.log() if logspace else weights_x,
                gaussian_mean_x=gaussian_mean_x,
                gaussian_std_x=gaussian_std_x,
                mixture_weights_y=weights_y.log() if logspace else weights_y,
                gaussian_mean_y=gaussian_mean_y,
                gaussian_std_y=gaussian_std_y,
                log_space=logspace,
            )

            assert torch.allclose(sed, torch.zeros(1).double())


def test_SED_3():
    """
    Test that SED > 0 when mixtures are not the same.
    In high dimensions the distance tends to be very small!
    :return:
    """
    M = 20
    M2 = 10
    D = 40
    gaussian_mean_x = torch.rand((M, D)).double() * 10
    gaussian_std_x = torch.rand((M, D)).double()
    weights_x = torch.rand(M).double()

    for logspace in [False]:

        trials = 50
        for t in range(trials):
            gaussian_mean_y = torch.rand((M2, D)).double() * 10
            gaussian_std_y = torch.rand((M2, D)).double()
            weights_y = torch.rand(M2).double()

            sed = SED(
                mixture_weights_x=weights_x.log() if logspace else weights_x,
                gaussian_mean_x=gaussian_mean_x,
                gaussian_std_x=gaussian_std_x,
                mixture_weights_y=weights_y.log() if logspace else weights_y,
                gaussian_mean_y=gaussian_mean_y,
                gaussian_std_y=gaussian_std_y,
                log_space=logspace,
            )

            # print(sed)
            assert not torch.allclose(sed, torch.zeros(1).double()), sed
            assert torch.all(sed > 0.0), sed


def SED_full(
    mixture_weights_x: torch.Tensor,
    gaussian_mean_x: torch.Tensor,
    gaussian_sigma_x: torch.Tensor,
    mixture_weights_y: torch.Tensor,
    gaussian_mean_y: torch.Tensor,
    gaussian_sigma_y: torch.Tensor,
    log_space: bool = True,
) -> torch.Tensor:
    """
    Computes the Squared Error Distance (SED) divergence between two
    multivariate Gaussian mixture models (full covariance matrices).

    :param mixture_weights_x: a tensor of shape (M) with the mixing weights
        for each class associated with the first distribution
    :param gaussian_mean_x: a tensor of shape (M, D) with the means
        for each class and mixture associated with the first distribution
    :param gaussian_sigma_x: a tensor of shape (M, D, D) with the covariance
        matrix for each class and mixture associated with the first
        distribution
    :param mixture_weights_y: a tensor of shape (M') with the mixing weights
        for each class associated with the second distribution
    :param gaussian_mean_y: a tensor of shape (M', D) with the means
        for each class and mixture associated with the second distribution
    :param gaussian_sigma_y: a tensor of shape (M', D, D) with the covariance
        matrix for each class and mixture associated with the second
        distribution
    :param log_space: whether to use log_space computations until the
        very end or not. IF TRUE, the weights are assumed to be already in log
        space.
    :return:
    """
    M_x, D = gaussian_mean_x.shape[0], gaussian_mean_x.shape[1]
    M_y = gaussian_mean_y.shape[0]
    assert D == gaussian_mean_y.shape[1]

    if log_space:
        alphas_alphas = mixture_weights_x.unsqueeze(
            1
        ) + mixture_weights_x.unsqueeze(
            0
        )  # (M,1) + (1,M) = (M,M)

        betas_betas = mixture_weights_y.unsqueeze(
            1
        ) + mixture_weights_y.unsqueeze(
            0
        )  # (M',1) + (1,M') = (M',M')

        alphas_betas = mixture_weights_x.unsqueeze(
            1
        ) + mixture_weights_y.unsqueeze(
            0
        )  # (M,1) + (1,M') = (M,M')

    else:
        alphas_alphas = mixture_weights_x.unsqueeze(
            1
        ) * mixture_weights_x.unsqueeze(
            0
        )  # (M,1) x (1,M) = (M,M)

        betas_betas = mixture_weights_y.unsqueeze(
            1
        ) * mixture_weights_y.unsqueeze(
            0
        )  # (M',1) x (1,M') = (M',M')

        alphas_betas = mixture_weights_x.unsqueeze(
            1
        ) * mixture_weights_y.unsqueeze(
            0
        )  # (M,1) x (1,M') = (M,M')

    # fixing C and D, repeat a vector of shape Mx1 across M columns
    args_a = gaussian_mean_x.unsqueeze(1).repeat(1, M_x, 1)  # (M,M,D)

    # fixing C and D, repeat a vector of shape 1xM across M rows
    mean_a = gaussian_mean_x.unsqueeze(0).repeat(M_x, 1, 1)  # (M,M,D)

    sigma_a = gaussian_sigma_x.unsqueeze(1) + gaussian_sigma_x.unsqueeze(
        0
    )  # (M,M,D,D)

    # fixing C and D, repeat a vector of shape Mx1 across M columns
    args_b = gaussian_mean_y.unsqueeze(1).repeat(1, M_y, 1)  # (M',M',D)

    # fixing C and D, repeat a vector of shape 1xM across M rows
    mean_b = gaussian_mean_y.unsqueeze(0).repeat(M_y, 1, 1)  # (M',M',D)

    sigma_b = gaussian_sigma_y.unsqueeze(1) + gaussian_sigma_y.unsqueeze(
        0
    )  # (M',M',D,D)

    # fixing C and D, repeat a vector of shape Mx1 across M columns
    args_c = gaussian_mean_x.unsqueeze(1).repeat(1, M_y, 1)  # (M,M',D)

    # fixing C and D, repeat a vector of shape 1xM across M rows
    mean_c = gaussian_mean_y.unsqueeze(0).repeat(M_x, 1, 1)  # (M,M',D)

    sigma_c = gaussian_sigma_x.unsqueeze(1) + gaussian_sigma_y.unsqueeze(
        0
    )  # (M,1,D,D) + (1,M',D,D) = (M,M',D,D)

    """
    Each term in SED defines a mixture of multivariate Gaussians with
    M*M' components. Therefore we reshape
    """
    alphas_alphas_ = alphas_alphas.reshape(-1)
    betas_betas_ = betas_betas.reshape(-1)
    alphas_betas_ = alphas_betas.reshape(-1)
    args_a = args_a.reshape(-1, D)
    args_b = args_b.reshape(-1, D)
    args_c = args_c.reshape(-1, D)
    mean_a = mean_a.reshape(-1, D)
    mean_b = mean_b.reshape(-1, D)
    mean_c = mean_c.reshape(-1, D)
    sigma_a = sigma_a.reshape(-1, D, D)
    sigma_b = sigma_b.reshape(-1, D, D)
    sigma_c = sigma_c.reshape(-1, D, D)

    comp_a = Independent(
        MultivariateNormal(loc=mean_a, covariance_matrix=sigma_a), 0
    )

    comp_b = Independent(
        MultivariateNormal(loc=mean_b, covariance_matrix=sigma_b), 0
    )

    comp_c = Independent(
        MultivariateNormal(loc=mean_c, covariance_matrix=sigma_c), 0
    )

    if log_space:
        sed = (
            torch.logsumexp(
                alphas_alphas_ + comp_a.log_prob(args_a), dim=0
            ).exp()
            + torch.logsumexp(
                betas_betas_ + comp_b.log_prob(args_b), dim=0
            ).exp()
            - 2
            * torch.logsumexp(
                alphas_betas_ + comp_c.log_prob(args_c), dim=0
            ).exp()
        )
    else:
        sed = (
            (alphas_alphas_ * (comp_a.log_prob(args_a).exp())).sum()
            + (betas_betas_ * (comp_b.log_prob(args_b).exp())).sum()
            - 2 * (alphas_betas_ * (comp_c.log_prob(args_c).exp())).sum()
        )
    return sed


def test_SED_full_1():
    """
    If you permute the mixtures the result should still be 0.
    """
    M = 10
    D = 30
    gaussian_mean_x = torch.rand((M, D)).double()
    gaussian_std_x = torch.rand((M, D, D)).double()
    gaussian_sigma_x = torch.bmm(
        gaussian_std_x, gaussian_std_x.transpose(2, 1)
    )
    weights_x = torch.rand(M).double()

    for logspace in [False, True]:
        sed = SED_full(
            mixture_weights_x=weights_x.log() if logspace else weights_x,
            gaussian_mean_x=gaussian_mean_x,
            gaussian_sigma_x=gaussian_sigma_x,
            mixture_weights_y=weights_x.log() if logspace else weights_x,
            gaussian_mean_y=gaussian_mean_x,
            gaussian_sigma_y=gaussian_sigma_x,
            log_space=logspace,
        )
        assert torch.allclose(sed, torch.zeros(1).double())

        trials = 10
        for t in range(trials):
            permutation = torch.randperm(M)

            gaussian_mean_y = gaussian_mean_x[permutation]
            gaussian_sigma_y = gaussian_sigma_x[permutation]
            weights_y = weights_x[permutation]

            sed = SED_full(
                mixture_weights_x=weights_x.log() if logspace else weights_x,
                gaussian_mean_x=gaussian_mean_x,
                gaussian_sigma_x=gaussian_sigma_x,
                mixture_weights_y=weights_y.log() if logspace else weights_y,
                gaussian_mean_y=gaussian_mean_y,
                gaussian_sigma_y=gaussian_sigma_y,
                log_space=logspace,
            )

            assert torch.allclose(sed, torch.zeros(1).double())
