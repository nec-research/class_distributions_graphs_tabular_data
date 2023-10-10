"""
  On Class Distributions Induced by Nearest Neighbor Graphs for Node Classification of Tabular Data

  File:     util.py
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

from typing import List

import torch
from torch.distributions import *
import matplotlib.pyplot as plt
import seaborn as sns

from M_c import M_c
from M_x import M_x


def add_axes(ax, use_x_ticks=True, use_y_ticks=True, axis_range=None):
    """
    Adds x and y axes with an arrow to an existing plot
    :param ax: the Axis object of matplotlib
    :param use_x_ticks: whether to use x ticks
    :param use_y_ticks: whether to use y ticks
    :param axis_range: [xmin, xmax, ymin, ymax]
    :return:
    """
    ax.spines["left"].set_position("zero")
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position("zero")
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    # make arrows
    xmin, xmax, ymin, ymax = axis_range
    ax.plot(
        (xmin, xmax - 0.05),
        (0, 0),
        ls="solid",
        lw=1,
        marker=">",
        color="black",
        markevery=[-1],
    )
    ax.plot(
        (0, 0),
        (0, ymax - 0.05),
        ls="solid",
        lw=1,
        marker="^",
        color="black",
        markevery=[-1],
    )

    if axis_range is not None:
        plt.axis(axis_range)

    if not use_x_ticks:
        ax.set_xticks([], [])

    if not use_y_ticks:
        ax.set_yticks([], [])

    plt.xlabel("x", loc="right")


def plot_gaussian_mixture(
    ax,
    gaussian_mean: torch.Tensor,
    gaussian_std: torch.Tensor,
    components_weights: torch.Tensor,
    min_x: float,
    max_x: float,
    steps: int,
    color: str = "black",
    label: str = None,
):
    """
    Plot a Gaussian mixture
    :param ax: the Axis object of matplotlib
    :param gaussian_mean: tensor of shape (M,1)
    :param gaussian_std: tensor of shape (M,1)
    :param components_weights: tensor of shape (M)
    :param min_x: minimum x for which to plot
    :param max_x: maximum x for which to plot
    :param steps: number of points to plot between min_x and max_x
    :param color: color of line (matplotlib)
    :param label: label of the curve in the legend
    :return:
    """
    X = torch.linspace(start=min_x, steps=steps, end=max_x).unsqueeze(
        1
    )  # (N,1)

    mix = Categorical(components_weights)
    comp = Normal(gaussian_mean.squeeze(1), gaussian_std.squeeze(1))
    gmm = MixtureSameFamily(mix, comp)

    Y = gmm.log_prob(X).exp()

    ax.plot(X, Y, c=color, label=label)


def plot_M_x(
    ax,
    epsilon: torch.double,
    gaussian_mean: torch.Tensor,
    gaussian_std: torch.Tensor,
    p_m_given_c: torch.Tensor,
    p_c,
    normalize: bool,
    min_x: float,
    max_x: float,
    steps: int,
    colors: List[str] = ["black", "black"],
    labels: List[str] = None,
):
    """
    Plot the (normalized) M_x(c) curves
    :param ax: the Axis object of matplotlib
    :param epsilon: the length of the interval
    :param gaussian_mean: tensor of shape (C, M,1)
    :param gaussian_std: tensor of shape (C, M,1)
    :param p_m_given_c: tensor of shape (C, M)
    :param p_c: tensor of shape (C)
    :param normalize: whether to normalize M_x or not
    :param min_x: minimum x for which to plot
    :param max_x: maximum x for which to plot
    :param steps: number of points to plot between min_x and max_x
    :param colors: list of colors, one per class (matplotlib)
    :param labels: labels of the curves in the legend
    :return:
    """
    X = torch.linspace(start=min_x, steps=steps, end=max_x).unsqueeze(
        1
    )  # (N,1)

    m_x = M_x(
        X,
        epsilon,
        p_c,
        p_m_given_c,
        gaussian_mean,
        gaussian_std,
        normalize=normalize,
    )

    for c in range(m_x.shape[1]):
        ax.plot(X, m_x[:, c], c=colors[c], ls="--", label=labels[c])


def plot_M_c(
    ax,
    epsilon: torch.double,
    gaussian_mean: torch.Tensor,
    gaussian_std: torch.Tensor,
    p_m_given_c: torch.Tensor,
    p_c,
    normalize: bool,
):
    """
    Plot the (normalized) M_c'(c) matrix
    :param ax: the Axis object of matplotlib
    :param epsilon: the length of the interval
    :param gaussian_mean: tensor of shape (C, M,1)
    :param gaussian_std: tensor of shape (C, M,1)
    :param p_m_given_c: tensor of shape (C, M)
    :param p_c: tensor of shape (C)
    :param normalize: whether to normalize M_x or not
    :return:
    """
    m_c = M_c(
        epsilon,
        p_c,
        p_m_given_c,
        gaussian_mean,
        gaussian_std,
        normalize=normalize,
    )

    if normalize:
        sns.heatmap(m_c, cmap="YlGnBu", ax=ax, vmin=0.0, vmax=1.0)
    else:
        sns.heatmap(m_c, cmap="YlGnBu", ax=ax)
