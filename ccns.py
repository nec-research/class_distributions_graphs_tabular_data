"""
  On Class Distributions Induced by Nearest Neighbor Graphs for Node Classification of Tabular Data

  File:     ccns.py
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
from torch import cdist
from torch.distributions import Categorical, MixtureSameFamily, \
    MultivariateNormal, Independent, Normal
from torch.nn.functional import one_hot
from torch_geometric.utils import degree
from torch_scatter import scatter

from M_x import M_x


def ccns(
    edge_index: torch.Tensor,
    class_labels: torch.Tensor,
    num_nodes: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the CCNS of https://arxiv.org/abs/2106.06134.
    It assumes source nodes are stored in `edge_index[0]` and destination
    nodes in `edge_index[1]`. It uses L2 distance to compute discrepancy

    :param edge_index: the connectivity of the graph in PyG format
    :param class_labels: a 1D vector of class labels for the nodes
    :param num_nodes: the number of nodes in the graph
    :return: a tensor of size CxC containing the empirical CCNS
    """
    src_index = edge_index[0]
    dst_index = edge_index[1]

    deg = degree(index=dst_index, num_nodes=num_nodes).unsqueeze(1)  # Nx1


    classes_one_hot = one_hot(class_labels)  # NxC
    num_classes = classes_one_hot.shape[1]

    # compute distribution of neighboring classes for each node
    dist = scatter(src=classes_one_hot[src_index], index=dst_index, dim=0) / deg

    ccns = torch.zeros(num_classes, num_classes)

    for c in range(num_classes):
        for c_prime in range(num_classes):
            v_c_mask = class_labels == c
            v_c_prime_mask = class_labels == c_prime

            # normalize count of classes to get a probability
            dist_c = dist[v_c_mask]  # NVC x num_classes
            dist_c_prime = dist[v_c_prime_mask]  # NVC' x num_classes

            assert torch.allclose(dist_c.sum(1), torch.tensor([1.]))
            assert torch.allclose(dist_c_prime.sum(1), torch.tensor([1.]))

            ccns[c, c_prime] += (
                cdist(
                    dist_c,
                    dist_c_prime,
                    p=2.0,
                    compute_mode="donot_use_mm_for_euclid_dist",
                ).sum() / (v_c_mask.sum() * v_c_prime_mask.sum())
            )

    return ccns



def monte_carlo_ccns(num_classes: int,
                     num_samples: int,
                     epsilon: torch.double,
                     p_c: torch.Tensor,
                     p_m_given_c: torch.Tensor,
                     gaussian_mean: torch.Tensor,
                     gaussian_std: torch.Tensor) -> torch.Tensor:
    """
    It estimates the expectation of Equation 1 using a q_c and q_c' computed
    using the normalized version of Proposition 3.3.

    :param num_classes: the value C of the number of classes
    :param num_samples: the number of samples **for each class** to use to
        approximate CCNS(c,c').
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
    :param q_x: a tensor of shape NxC holding the distributions computed by
        Proposition 3.3 for the dataset of points.

    :return: an estimate of the ccns as a tensor of size CxC
    """
    num_features = gaussian_mean.shape[-1]
    assert len(gaussian_std.shape) < 4, 'the theory of Section 3.1 assumes a' \
                                        'diagonal covariance matrix'

    def sample(class_id):

        if num_features > 1:
            sigma = torch.diag_embed(gaussian_std)

            gmm = MixtureSameFamily(
                Categorical(p_m_given_c[class_id]),
                Independent(MultivariateNormal(loc=gaussian_mean[class_id],
                                               covariance_matrix=sigma[class_id]), 0),
            )

        else:
            gmm = MixtureSameFamily(
                Categorical(p_m_given_c[class_id]),
                Independent(Normal(loc=gaussian_mean[class_id],
                                   scale=gaussian_std[class_id]), 1),
            )

        samples = gmm.sample((num_samples,))
        return samples

    ccns_monte_carlo = torch.zeros(num_classes, num_classes)

    for c in range(num_classes):
        for c_prime in range(num_classes):
            X_c = sample(c)
            q_c = M_x(X=X_c,
                      epsilon=epsilon,
                      p_c=p_c,
                      p_m_given_c=p_m_given_c,
                      gaussian_mean=gaussian_mean,
                      gaussian_std=gaussian_std,
                      normalize=True)

            X_c_prime = sample(c_prime)
            q_c_prime = M_x(X=X_c_prime,
                            epsilon=epsilon,
                            p_c=p_c,
                            p_m_given_c=p_m_given_c,
                            gaussian_mean=gaussian_mean,
                            gaussian_std=gaussian_std,
                            normalize=True)

            ccns_monte_carlo[c, c_prime] +=  \
                torch.norm(q_c - q_c_prime, p=2, dim=1).sum() / num_samples

    return ccns_monte_carlo