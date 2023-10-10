"""
  On Class Distributions Induced by Nearest Neighbor Graphs for Node Classification of Tabular Data

  File:     model.py
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

from typing import Tuple, Optional, List, Callable

import faiss
import torch
import torch_geometric
from pydgn.model.interface import ModelInterface
from torch import relu
from torch.nn import Sequential, Linear, ReLU, ModuleList
from torch.nn.functional import normalize
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing, GIN, GCNConv
from torch_geometric.nn import knn_graph


class MLP(ModelInterface):
    def __init__(
        self,
        dim_node_features: int,
        dim_edge_features: int,
        dim_target: int,
        readout_class: Callable[..., torch.nn.Module],
        config: dict,
    ):
        super().__init__(
            dim_node_features,
            dim_edge_features,
            dim_target,
            readout_class,
            config,
        )
        self.cosine = config.get("cosine", False)

        self.num_layers = config["num_layers"]
        self.hidden_units = config["hidden_units"]

        layers = (
            [Linear(dim_node_features, self.hidden_units)]
            + [
                Linear(self.hidden_units, self.hidden_units)
                for _ in range(self.num_layers)
            ]
            + [Linear(self.hidden_units, dim_target)]
        )
        self.layers = ModuleList(layers)

    def forward(
        self, data: Batch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:

        h = relu(self.layers[0](data.x))

        for l in range(1, len(self.layers) - 1):
            h = relu(self.layers[l](h))

        o = self.layers[-1](h)

        # we need to compute the subset of [tr/val/test] output and target
        # values using the indices provided by the data loader
        if self.training:
            o = o[data.training_indices]
            h = h[data.training_indices]
            y = data.y[data.training_indices]
        else:
            o = o[data.eval_indices]
            h = h[data.eval_indices]
            y = data.y[data.eval_indices]

        return o, h, [y]


class SimpleDGNConv(MessagePassing):
    """
    Simply computes mean aggregation of neighbors
    """

    def __init__(self):
        super().__init__(aggr="mean")

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j):
        return x_j


class SimpleDGN(ModelInterface):
    def __init__(
        self,
        dim_node_features: int,
        dim_edge_features: int,
        dim_target: int,
        readout_class: Callable[..., torch.nn.Module],
        config: dict,
    ):
        super().__init__(
            dim_node_features,
            dim_edge_features,
            dim_target,
            readout_class,
            config,
        )
        self.cosine = config.get("cosine", False)


        self.num_layers = config["num_layers"]
        self.hidden_units = config["hidden_units"]
        self.k = config["k"]

        self.mean_aggregation = SimpleDGNConv()

        layers = (
            [Linear(dim_node_features, self.hidden_units)]
            + [
                Linear(self.hidden_units, self.hidden_units)
                for _ in range(self.num_layers)
            ]
            + [Linear(self.hidden_units, dim_target)]
        )
        self.layers = ModuleList(layers)

        self.knn_edge_index = None

    def forward(
        self, data: Batch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:

        if self.knn_edge_index is None:
            if not self.cosine:
                # build knn graph and store it in the model
                self.knn_edge_index = knn_graph(
                    data.x, self.k,
                    loop=True,
                    batch=None  # use dest node as well
                )
            else:
                index = faiss.IndexFlatIP(data.x.shape[1])
                x_norm = normalize(data.x, p=2)
                index.add(x_norm.detach().numpy())
                D, I = index.search(x_norm, self.k+1)  # sanity check
                self.knn_edge_index = torch.stack(
                    [   torch.tensor(I).reshape(-1),
                        torch.arange(data.x.shape[0]).unsqueeze(1).repeat(1, self.k+1).reshape(-1)], dim=0)

        # compute mean aggregation of neighbors in input space
        x = self.mean_aggregation(data.x, self.knn_edge_index)

        # same code as MLP above (the classifier)
        h = relu(self.layers[0](x))

        for l in range(1, len(self.layers) - 1):
            h = relu(self.layers[l](h))

        o = self.layers[-1](h)

        # we need to compute the subset of [tr/val/test] output and target
        # values using the indices provided by the data loader
        if self.training:
            o = o[data.training_indices]
            h = h[data.training_indices]
            y = data.y[data.training_indices]
        else:
            o = o[data.eval_indices]
            h = h[data.eval_indices]
            y = data.y[data.eval_indices]

        return o, h, [y]


class GIN(ModelInterface):
    def __init__(
        self,
        dim_node_features: int,
        dim_edge_features: int,
        dim_target: int,
        readout_class: Callable[..., torch.nn.Module],
        config: dict,
    ):
        super().__init__(
            dim_node_features,
            dim_edge_features,
            dim_target,
            readout_class,
            config,
        )

        self.cosine = config.get("cosine", False)

        self.num_layers = config["num_layers"]
        self.hidden_units = config["hidden_units"]
        self.k = config["k"]
        self.dropout = config["dropout"]
        self.aggregation = config["aggregation"]

        self.gin = torch_geometric.nn.GIN(
            dim_node_features,
            self.hidden_units,
            self.num_layers,
            dim_target,
            self.dropout,
            jk="cat",
            train_eps=True,
            eps=1.0,
        )

        # change aggregation method to hyper-parameter value
        for l in range(len(self.gin.convs)):
            self.gin.convs[l].aggr = self.aggregation

        self.knn_edge_index = None

    def forward(
        self, data: Batch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:

        if self.knn_edge_index is None:
            if not self.cosine:
                # build knn graph and store it in the model
                self.knn_edge_index = knn_graph(
                    data.x,
                    self.k,
                    cosine=self.cosine,
                    loop=False,  # GIN already uses dest node
                    batch=None,
                )
            else:
                index = faiss.IndexFlatIP(data.x.shape[1])
                x_norm = normalize(data.x, p=2)
                index.add(x_norm.detach().numpy())
                D, I = index.search(x_norm, self.k+1)  # sanity check
                self.knn_edge_index = torch.stack(
                    [   torch.tensor(I[:,1:]).reshape(-1),
                        torch.arange(data.x.shape[0]).unsqueeze(1).repeat(1, self.k).reshape(-1)], dim=0)

        o = self.gin(data.x, self.knn_edge_index)

        # we need to compute the subset of [tr/val/test] output and target
        # values using the indices provided by the data loader
        if self.training:
            o = o[data.training_indices]
            h = o  # not used
            y = data.y[data.training_indices]
        else:
            o = o[data.eval_indices]
            h = o  # not used
            y = data.y[data.eval_indices]

        return o, h, [y]


class GCN(ModelInterface):
    def __init__(
        self,
        dim_node_features: int,
        dim_edge_features: int,
        dim_target: int,
        readout_class: Callable[..., torch.nn.Module],
        config: dict,
    ):
        super().__init__(
            dim_node_features,
            dim_edge_features,
            dim_target,
            readout_class,
            config,
        )

        self.cosine = config.get("cosine", False)


        self.num_layers = config["num_layers"]
        self.hidden_units = config["hidden_units"]
        self.k = config["k"]

        # This code might not correspond to the exact implementation of the
        # original GCN paper
        # self.gcn = torch_geometric.nn.GCN(
        #     dim_node_features,
        #     self.hidden_units,
        #     self.num_layers,
        #     dim_target,
        #     self.dropout,
        #     jk="cat",
        #     train_eps=True,
        #     eps=1.0,
        # )

        layers = []
        # change aggregation method to hyper-parameter value
        for l in range(self.num_layers):
            layers.append(GCNConv(dim_node_features if l == 0 else self.hidden_units,
                                  self.hidden_units, cached=True))

        self.layers = ModuleList(layers)

        self.knn_edge_index = None


    def forward(
        self, data: Batch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[object]]]:

        if self.knn_edge_index is None:
            if not self.cosine:
                # build knn graph and store it in the model
                self.knn_edge_index = knn_graph(
                    data.x,
                    self.k,
                    cosine=self.cosine,
                    loop=False,  # GIN already uses dest node
                    batch=None,
                )
            else:
                index = faiss.IndexFlatIP(data.x.shape[1])
                x_norm = normalize(data.x, p=2)
                index.add(x_norm.detach().numpy())
                D, I = index.search(x_norm, self.k+1)  # sanity check
                self.knn_edge_index = torch.stack(
                    [   torch.tensor(I[:,1:]).reshape(-1),
                        torch.arange(data.x.shape[0]).unsqueeze(1).repeat(1, self.k).reshape(-1)], dim=0)

        for l in range(self.num_layers):
            if l == 0:
                o = torch.relu(self.layers[0](data.x, self.knn_edge_index))
            elif l == self.num_layers - 1:
                # output class responsibilities
                o = self.layers[l](o, self.knn_edge_index)
            else:
                # intermediate layer
                o = torch.relu(self.layers[l](o, self.knn_edge_index))

        # we need to compute the subset of [tr/val/test] output and target
        # values using the indices provided by the data loader
        if self.training:
            o = o[data.training_indices]
            h = o  # not used
            y = data.y[data.training_indices]
        else:
            o = o[data.eval_indices]
            h = o  # not used
            y = data.y[data.eval_indices]

        return o, h, [y]