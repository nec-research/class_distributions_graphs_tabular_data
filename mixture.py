"""
  On Class Distributions Induced by Nearest Neighbor Graphs for Node Classification of Tabular Data

  File:     mixture.py
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


def sum_of_mixtures(
    mixture_weights_x: torch.Tensor,
    gaussian_mean_x: torch.Tensor,
    gaussian_std_x: torch.Tensor,
    mixture_weights_y: torch.Tensor,
    gaussian_mean_y: torch.Tensor,
    gaussian_std_y: torch.Tensor,
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Computes the sum of two mixtures, one with M components and the other with
    M' components

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
    :return: a tuple (weights, mean, std) associated with a new mixture of
        M*M' components
    """
    D = gaussian_mean_x.shape[1]
    assert D == gaussian_mean_y.shape[1]

    weights = mixture_weights_x.unsqueeze(1) * mixture_weights_y.unsqueeze(
        0
    )  # (M',M)

    gaussian_mean = gaussian_mean_x.unsqueeze(1) + gaussian_mean_y.unsqueeze(
        0
    )  # (M',M, D)

    gaussian_std = torch.sqrt(
        torch.pow(gaussian_std_x.unsqueeze(1), 2)
        + torch.pow(gaussian_std_y.unsqueeze(0), 2)
    )  # (M',M, D)

    return (
        weights.reshape(-1),
        gaussian_mean.reshape(-1, D),
        gaussian_std.reshape(-1, D),
    )


def test_sum_of_mixtures():
    M, D = 2, 2

    gaussian_mean = torch.zeros((M, D)).double()
    gaussian_mean[1, :] += 2
    # means mixture 0 = [0,0]
    # mean mixture 1 = [2,2]

    gaussian_std = torch.ones((M, D)).double()
    # std = 1.

    weights = torch.tensor([0.3, 0.7]).double()

    w, m, s = sum_of_mixtures(
        weights,
        gaussian_mean,
        gaussian_std,
        weights,
        gaussian_mean,
        gaussian_std,
    )

    assert w.sum() == 1.0
    assert torch.allclose(m.sum(dim=0), torch.tensor([8.0, 8.0]).double())
    assert torch.allclose(
        s.sum(dim=0),
        torch.tensor(
            [
                torch.sqrt(torch.tensor([2.0])) * 4,
                torch.sqrt(torch.tensor([2.0])) * 4,
            ]
        ).double(),
    )


def sum_of_mixtures_full(
    mixture_weights_x: torch.Tensor,
    gaussian_mean_x: torch.Tensor,
    gaussian_sigma_x: torch.Tensor,
    mixture_weights_y: torch.Tensor,
    gaussian_mean_y: torch.Tensor,
    gaussian_sigma_y: torch.Tensor,
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Computes the sum of two mixtures, one with M components and the other with
    M' components for general mixtures.
    :param mixture_weights_x: a tensor of shape (M) with the mixing weights
        for each class associated with the first distribution
    :param gaussian_mean_x: a tensor of shape (M, D) with the means
        for each class and mixture associated with the first distribution
    :param gaussian_sigma_x: a tensor of shape (M, D, D) with the covariance
        for each class and mixture associated with the first distribution
    :param mixture_weights_y: a tensor of shape (M') with the mixing weights
        for each class associated with the second distribution
    :param gaussian_mean_y: a tensor of shape (M', D) with the means
        for each class and mixture associated with the second distribution
    :param gaussian_sigma_y: a tensor of shape (M', D, D) with the covariance
        for each class and mixture associated with the second distribution
    :return: a tuple (weights, mean, covariance) associated with a new
        mixture of M*M' components
    """
    D = gaussian_mean_x.shape[1]
    assert D == gaussian_mean_y.shape[1]

    weights = mixture_weights_x.unsqueeze(1) * mixture_weights_y.unsqueeze(
        0
    )  # (M',M)

    gaussian_mean = gaussian_mean_x.unsqueeze(1) + gaussian_mean_y.unsqueeze(
        0
    )  # (M',M, D)

    gaussian_sigma = gaussian_sigma_x.unsqueeze(
        1
    ) + gaussian_sigma_y.unsqueeze(
        0
    )  # (M',M, D, D)

    return (
        weights.reshape(-1),
        gaussian_mean.reshape(-1, D),
        gaussian_sigma.reshape(-1, D, D),
    )


def test_sum_of_mixtures_full():
    M, D = 2, 2

    gaussian_mean = torch.zeros((M, D)).double()
    gaussian_mean[1, :] += 2
    # means mixture 0 = [0,0]
    # mean mixture 1 = [2,2]

    gaussian_sigma = torch.diag_embed(torch.ones((M, D))).double()
    # Sigma = diag(1.,1.,,,)

    weights = torch.tensor([0.3, 0.7]).double()

    w, m, s = sum_of_mixtures_full(
        weights,
        gaussian_mean,
        gaussian_sigma,
        weights,
        gaussian_mean,
        gaussian_sigma,
    )

    assert w.sum() == 1.0
    assert torch.allclose(m.sum(dim=0), torch.tensor([8.0, 8.0]).double())
    assert torch.allclose(
        s.sum(dim=(0, 1)),
        torch.tensor(
            [
                torch.tensor([2.0]) * 4,
                torch.tensor([2.0]) * 4,
            ]
        ).double(),
    )


def mixture_times_constant(
    constant: float,
    mixture_weights: torch.Tensor,
    gaussian_mean: torch.Tensor,
    gaussian_std_or_sigma: torch.Tensor,
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Computes the product of a mixture with a constant. This can be used with
    a diagonal or full covariance matrix

    :param constant: the constant to be multiplied
    :param mixture_weights: a tensor of shape (M) with the mixing weights
        for each class associated with the first distribution
    :param gaussian_mean: a tensor of shape (M, D) with the means
        for each class and mixture associated with the first distribution
    :param gaussian_std_or_sigma: a tensor of shape (M, D) with the std
        for each class and mixture associated with the first distribution.
        Alternatively, one can specify a full covariance (M, D, D) matrix
    :return: a tuple (weights, mean, std/covariance) associated with a new
        mixture of M*M' components
    """
    return (
        mixture_weights,
        gaussian_mean * constant,
        gaussian_std_or_sigma * constant * constant,
    )


def test_mixture_times_constant():
    M, D = 2, 2

    for k in [3.0, 1.0 / 3.0]:

        gaussian_mean = torch.zeros((M, D)).double()
        gaussian_mean[1, :] += 2
        # means mixture 0 = [0,0]
        # mean mixture 1 = [2,2]

        gaussian_std = torch.ones((M, D)).double()
        # std = 1.

        weights = torch.tensor([0.3, 0.7]).double()

        w, m, s = mixture_times_constant(
            k, weights, gaussian_mean, gaussian_std
        )

        assert w.sum() == 1.0
        assert torch.allclose(m, gaussian_mean * k)
        assert torch.allclose(s, gaussian_std * (k * k))


def mixture_linear_transform(
    W: torch.Tensor,
    mixture_weights: torch.Tensor,
    gaussian_mean: torch.Tensor,
    gaussian_std_or_sigma: torch.Tensor,
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Computes the linear transformation of a mixture. This can be used with
    a diagonal or full covariance matrix

    :param W: linear transformation of shape (D, D)
    :param mixture_weights: a tensor of shape (M) with the mixing weights
        for each class associated with the first distribution
    :param gaussian_mean: a tensor of shape (M, D) with the means
        for each class and mixture associated with the first distribution
    :param gaussian_std_or_sigma: a tensor of shape (M, D) with the std
        for each class and mixture associated with the first distribution.
        Alternatively, one can specify a full covariance (M, D, D) matrix
    :return: a tuple (weights, mean, std/covariance) associated with a new
        mixture of M*M' components
    """
    # if the user provided a std matrix with shape (M,D), convert it to
    # diagonal covariance matrix (M,D,D) (one for each of the M mixtures).
    if len(gaussian_std_or_sigma.shape) == 2:
        gaussian_std_or_sigma = torch.diag_embed(gaussian_std_or_sigma)

    return (
        mixture_weights,
        gaussian_mean @ W,
        torch.matmul(
            W.unsqueeze(0),  # left multiplication (broadcasted)
            torch.matmul(
                gaussian_std_or_sigma, W.transpose(1, 0).unsqueeze(0)
            ),  # right multiplication
        ),
    )


def test_mixture_linear_transform():
    M, D = 2, 2

    gaussian_mean = torch.zeros((M, D)).double()
    gaussian_mean[1, :] += 2
    # means mixture 0 = [0,0]
    # mean mixture 1 = [2,2]

    gaussian_sigma = torch.diag_embed(torch.ones((M, D))).double()
    # std = 1.

    weights = torch.tensor([0.3, 0.7]).double()

    W = torch.tensor([[1.0, 1.0], [1.0, 1.0]]).double()

    w, m, s = mixture_linear_transform(
        W, weights, gaussian_mean, gaussian_sigma
    )

    assert w.sum() == 1.0
    assert torch.allclose(m, gaussian_mean.sum(1, keepdims=True).repeat(1, 2))
    assert torch.allclose(
        s,
        (gaussian_sigma.sum(2, keepdims=True).repeat(1, 1, 2))
        .sum(2, keepdims=True)
        .repeat(1, 1, 2),
    )

    gaussian_std = torch.ones((M, D)).double()

    w, m, s = mixture_linear_transform(W, weights, gaussian_mean, gaussian_std)

    assert w.sum() == 1.0
    assert torch.allclose(m, gaussian_mean.sum(1, keepdims=True).repeat(1, 2))
    assert torch.allclose(
        s,
        (gaussian_sigma.sum(2, keepdims=True).repeat(1, 1, 2))
        .sum(2, keepdims=True)
        .repeat(1, 1, 2),
    )
