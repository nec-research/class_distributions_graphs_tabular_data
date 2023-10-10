"""
  On Class Distributions Induced by Nearest Neighbor Graphs for Node Classification of Tabular Data

  File:     M_x.py
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


def M_x(
    X: torch.Tensor,
    epsilon: torch.double,
    p_c: torch.Tensor,
    p_m_given_c: torch.Tensor,
    gaussian_mean: torch.Tensor,
    gaussian_std: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Computes the class posterior mass vector around a point (Prop. 3.2).
    :param X: matrix of size (N,D) with N vectors of dimension D for which
        we need to compute M_x
    :param epsilon: the length of the hypercube centered at x
    :param p_c: the prior class distribution p(c) estimated from the dataset.
        It is a vector of shape (C,), where C is the number of classes
    :param p_m_given_c: the weight vector of the class-conditional mixture
        p(m|c). It is a tensor of shape (C,M), where M is the number of
        mixtures.
    :param gaussian_mean: a tensor of shape (C,M,D) containing the means of
        the gaussian distributions associated with the different classes,
        mixtures and features
    :param gaussian_std: a tensor of shape (C,M,D) containing the standard
        deviations of the gaussian distributions associated with the different
        classes, mixtures and features
    :param normalize: whether to compute a normalization over the classes
        for each data point
    :return: the M_x of shape (N,C) containing the class posterior mass around
        all N points in the matrix X
    """
    normal = Normal(loc=gaussian_mean.double(), scale=gaussian_std.double())

    X_ = X.double().unsqueeze(1).unsqueeze(1)  # (N,1,1,D)

    F_left = normal.cdf(X_ + epsilon / 2.0)  # (N,C,M,D)
    F_right = normal.cdf(X_ - epsilon / 2.0)  # (N,C,M,D)

    # working version
    # prod_F = torch.prod(F_left - F_right, dim=-1)  # (N,C,M)
    # w_prod_F = (p_m_given_c.unsqueeze(0)*prod_F).sum(-1)  # (N,C)
    # result = (p_c.unsqueeze(0)*w_prod_F)  # (N,C)
    # if normalize:
    #     result_sum = result.sum(1, keepdims=True)
    #     result_sum[result_sum == 0] = 1.
    #     result = result/result_sum

    # working version in log-space
    log_prod_F = torch.sum(torch.log(F_left - F_right), dim=-1)  # (N,C,M)
    log_w_prod_F = torch.logsumexp(
        p_m_given_c.log().unsqueeze(0) + log_prod_F, dim=-1
    )  # (N,C)
    log_result = p_c.log().unsqueeze(0) + log_w_prod_F  # (N,C)

    if normalize:
        log_result_sum = torch.logsumexp(log_result, 1, keepdim=True)
        # If log log_result_sum=-inf then all individual logs are -inf
        # In this case there is not much we can do for now, unless setting
        # it to zero?
        log_result = log_result - log_result_sum

    return log_result.exp()


def test_1_M_x():
    """
    This configuration considers identical normal distributions.
    When normalized, it should give the prior distribution as M_x for all
    points and epsilon, since it does not matter which class you belong to.
    """
    N, C, M, D = 10, 2, 2, 2

    gaussian_mean = torch.zeros((C, M, D)).double()
    gaussian_std = torch.ones((C, M, D)).double()
    p_c = torch.Tensor([0.3, 0.7]).double()

    trials = 10
    for t in range(trials):
        X = torch.rand(N, D).double()
        epsilon = torch.rand(1).double() * 10
        p_m_given_c_unnorm = torch.rand(C, M)
        p_m_given_c = p_m_given_c_unnorm / p_m_given_c_unnorm.sum(
            1, keepdims=True
        )

        m_x = M_x(
            X,
            epsilon,
            p_c,
            p_m_given_c,
            gaussian_mean,
            gaussian_std,
            normalize=True,
        )  # (N, C)

        # Now compare with prior
        assert torch.allclose(
            m_x, p_c.unsqueeze(0).repeat(N, 1), rtol=1e62, atol=1e-64
        )


def test_2_M_x():
    """
    This configuration considers identical normal distributions given a class.
    We keep the distributions very distant from each other and consider a
    sample close to the first class, assuming p_m_given_c is uniformly
    distributed. We compare the result with manual computation.
    """
    N, C, M, D = 2, 2, 2, 2

    gaussian_mean = torch.zeros((C, M, D)).double()
    gaussian_mean[1, :, :] += 10
    # means class 0 = [0,0]
    # mean class 1 = [10,10]

    gaussian_std = torch.ones((C, M, D)).double() * 2
    # std = 2.

    p_c = torch.Tensor([0.3, 0.7]).double()
    # prior = [0.3, 0.7]

    X = torch.ones(N, D).double()
    # X = [[1., 1.]]

    epsilon = torch.ones(1).double() / 2
    # epsilon = 0.5

    p_m_given_c = torch.ones(C, M) / M
    # p_m_given_c = [[0.5, 0.5], [0.5, 0.5]]

    m_x = M_x(
        X, epsilon, p_c, p_m_given_c, gaussian_mean, gaussian_std
    )  # (N, C)

    # results computed manually
    correct_m_x = torch.tensor(
        [
            [0.002315007518907, 0.000000000012348],
            [0.002315007518907, 0.000000000012348],
        ]
    ).double()

    # Now compare with our results
    assert torch.allclose(m_x.squeeze(0), correct_m_x, rtol=1e62, atol=1e-64)


def test_3_M_x():
    """
    This configuration tests potential shape problems when dimensions are 1.
    We keep the distributions very distant from each other and consider a
    sample close to the first class, assuming p_m_given_c is uniformly
    distributed. We compare the result with manual computation.
    """
    N, C, M, D = 1, 2, 1, 1

    gaussian_mean = torch.zeros((C, M, D)).double()
    gaussian_mean[1, :, :] += 10
    # means class 0 = [0,0]
    # mean class 1 = [10,10]

    gaussian_std = torch.ones((C, M, D)).double() * 2
    # std = 2.

    p_c = torch.Tensor([0.3, 0.7]).double()
    # prior = [0.3, 0.7]

    X = torch.ones(N, D).double()
    # X = [[1., 1.]]

    epsilon = torch.ones(1).double() / 2
    # epsilon = 0.5

    p_m_given_c = torch.ones(C, M) / M
    # p_m_given_c = [[0.5, 0.5], [0.5, 0.5]]

    m_x = M_x(
        X, epsilon, p_c, p_m_given_c, gaussian_mean, gaussian_std
    )  # (N, C)

    # results computed manually
    correct_m_x = torch.tensor([0.002315007518907, 0.000000000012348]).double()

    # Now compare with our results
    assert torch.allclose(m_x.squeeze(0), correct_m_x, rtol=1e62, atol=1e-64)


def test_4_M_x():
    """
    Like test 3 but with normalization.
    """
    N, C, M, D = 1, 2, 1, 1

    gaussian_mean = torch.zeros((C, M, D)).double()
    gaussian_mean[1, :, :] += 10
    # means class 0 = [0,0]
    # mean class 1 = [10,10]

    gaussian_std = torch.ones((C, M, D)).double() * 2
    # std = 2.

    p_c = torch.Tensor([0.3, 0.7]).double()
    # prior = [0.3, 0.7]

    X = torch.ones(N, D).double()
    # X = [[1., 1.]]

    epsilon = torch.ones(1).double() / 2
    # epsilon = 0.5

    p_m_given_c = torch.ones(C, M) / M
    # p_m_given_c = [[0.5, 0.5], [0.5, 0.5]]

    m_x = M_x(
        X,
        epsilon,
        p_c,
        p_m_given_c,
        gaussian_mean,
        gaussian_std,
        normalize=True,
    )  # (N, C)

    # results computed manually
    correct_m_x_unnorm = torch.tensor(
        [0.002315007518907, 0.000000000012348]
    ).double()
    correct_m_x_norm = correct_m_x_unnorm / correct_m_x_unnorm.sum(
        0, keepdims=True
    )

    # Now compare with our results
    assert torch.allclose(
        m_x.squeeze(0), correct_m_x_norm, rtol=1e62, atol=1e-64
    )


def test_5_M_x():
    """
    Like test 4 but with epsilon very large.
    The normalized M_x should tend to p_c
    """
    N, C, M, D = 1, 2, 1, 1

    gaussian_mean = torch.zeros((C, M, D)).double()
    gaussian_mean[1, :, :] += 10
    # means class 0 = [0,0]
    # mean class 1 = [10,10]

    gaussian_std = torch.ones((C, M, D)).double() * 2
    # std = 2.

    p_c = torch.Tensor([0.3, 0.7]).double()
    # prior = [0.3, 0.7]

    X = torch.ones(N, D).double()
    # X = [[1., 1.]]

    epsilon = torch.ones(1).double() * 100000
    # epsilon = 0.5

    p_m_given_c = torch.ones(C, M) / M
    # p_m_given_c = [[0.5, 0.5], [0.5, 0.5]]

    m_x = M_x(
        X,
        epsilon,
        p_c,
        p_m_given_c,
        gaussian_mean,
        gaussian_std,
        normalize=True,
    )  # (N, C)

    # Now compare with our results
    assert torch.allclose(m_x.squeeze(0), p_c, rtol=1e62, atol=1e-64)
