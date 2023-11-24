"""
  On Class Distributions Induced by Nearest Neighbor Graphs for Node Classification of Tabular Data

  File:     dataset.py
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

import os.path as osp
import zipfile
from io import StringIO, BytesIO
from pathlib import Path
from typing import Union, List, Tuple, Optional, Callable

import numpy as np
import requests
from unlzw3 import unlzw

import pandas as pd
import torch
from pydgn.data.dataset import DatasetInterface
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch_geometric.data import Data


class UCIDatasetInterface(DatasetInterface):
    column_names: List[str] = None
    categorical_columns: List[str] = None
    columns_to_drop: List[str] = None
    target_column: str = None
    uci_url: str = None

    def df_loader(self):
        df = pd.read_csv(self.uci_url, header=None, names=self.column_names)
        return df

    def load_uci_dataset(self):
        df = self.df_loader()

        # drop columns we do not want to use in these experiments
        df = df.drop(columns=self.columns_to_drop)

        # extract target from dataset
        y = df[self.target_column]
        df = df.drop(columns=[self.target_column])

        # replace categorical features with one hot
        df = pd.get_dummies(
            data=df, columns=self.categorical_columns, dummy_na=True
        )

        # fill NaNs with column means in each column
        df = df.fillna(df.mean())

        # convert values of y into classes
        unique_values = y.unique()
        unique_values_sorted = np.sort(unique_values)  # ascending order

        counter = 0
        for v in unique_values_sorted:
            y = y.replace(v, counter)
            counter += 1

        return df.to_numpy().astype(float), y.to_numpy()

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        **kwargs,
    ):
        self.root = root
        self.name = name

        assert transform is None, "no preprocessing allowed"
        assert pre_transform is None, "no preprocessing allowed"
        assert pre_filter is None, "no preprocessing allowed"

        super().__init__(root, name, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ["data.py"]

    @property
    def raw_paths(self) -> List[str]:
        return []

    @property
    def processed_paths(self) -> List[str]:
        return [osp.join(self.root, self.name, "processed", "data.py")]

    def download(self):
        pass

    def process(self):
        X, y = self.load_uci_dataset()

        d = Data(
            x=torch.tensor(X, dtype=torch.float),
            edge_index=None,
            edge_attr=None,
            y=torch.tensor(y, dtype=torch.long).unsqueeze(1),
            dtype=torch.float,
        )

        torch.save(d, osp.join(self.root, self.name, "processed", "data.py"))

    def get(self, idx: int) -> Data:
        return self.data

    @property
    def dim_node_features(self) -> int:
        return self.data.x.shape[1]

    @property
    def dim_edge_features(self) -> int:
        return 0

    @property
    def dim_target(self) -> int:
        return int(self.data.y.max().item()) + 1

    def __len__(self) -> int:
        return 1  # single graph


class Abalone(UCIDatasetInterface):
    column_names = [
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Weight.whole",
        "Weight.shucked",
        "Weight.viscera",
        "Weight.shell",
        "Rings",
    ]
    columns_to_drop = []
    categorical_columns = []
    target_column = "Sex"
    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"


class Adult(UCIDatasetInterface):
    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "class",
    ]
    columns_to_drop = []
    categorical_columns = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    target_column = "class"
    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"


class ElectricalGrid(UCIDatasetInterface):
    column_names = [
        "tau1",
        "tau2",
        "tau3",
        "tau4",
        "p1",
        "p2",
        "p3",
        "p4",
        "g1",
        "g2",
        "g3",
        "g4",
        "stab",
        "stabf",
    ]
    columns_to_drop = ["stab"]
    categorical_columns = []
    target_column = "stabf"
    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00471/Data_for_UCI_named.csv"

    def df_loader(self):
        df = pd.read_csv(self.uci_url)
        return df


class Musk(UCIDatasetInterface):
    column_names = [0, 1] + [i for i in range(168)]
    columns_to_drop = [0, 1]
    categorical_columns = []
    target_column = 168
    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/musk/clean2.data.Z"

    def df_loader(self):
        u = requests.get(self.uci_url)
        uncompressed_data = unlzw(u.content).decode("utf-8")
        s = StringIO(uncompressed_data)
        df = pd.read_csv(s, sep=",", header=None)
        return df


class Waveform(UCIDatasetInterface):
    column_names = [i for i in range(21)]
    columns_to_drop = []
    categorical_columns = []
    target_column = 21
    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/waveform/waveform.data.Z"

    def df_loader(self):
        u = requests.get(self.uci_url)
        uncompressed_data = unlzw(u.content).decode("utf-8")
        s = StringIO(uncompressed_data)
        df = pd.read_csv(s, sep=",", header=None)
        return df


class Isolet(UCIDatasetInterface):
    column_names = [i for i in range(617)]
    columns_to_drop = []
    categorical_columns = []
    target_column = 617
    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet1+2+3+4.data.Z"
    uci_url_2 = "https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet5.data.Z"

    def df_loader(self):
        u1 = requests.get(self.uci_url)
        uncompressed_data1 = unlzw(u1.content).decode("utf-8")
        s1 = StringIO(uncompressed_data1)
        df1 = pd.read_csv(s1, sep=",", header=None)

        u2 = requests.get(self.uci_url_2)
        uncompressed_data2 = unlzw(u2.content).decode("utf-8")
        s2 = StringIO(uncompressed_data2)
        df2 = pd.read_csv(s2, sep=",", header=None)

        return pd.concat([df1, df2])


class OccupancyDetection(UCIDatasetInterface):
    column_names = [
        "date",
        "Temperature",
        "Humidity",
        "Light",
        "CO2",
        "HumidityRatio",
        "Occupancy",
    ]
    columns_to_drop = ["date"]
    categorical_columns = []
    target_column = "Occupancy"
    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip"

    def df_loader(self):
        u = requests.get(self.uci_url)

        with zipfile.ZipFile(BytesIO(u.content), "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)

        path1 = Path(self.raw_dir, "datatraining.txt")
        df1 = pd.read_csv(path1, sep=",")

        path2 = Path(self.raw_dir, "datatest.txt")
        df2 = pd.read_csv(path2, sep=",")

        path3 = Path(self.raw_dir, "datatest2.txt")
        df3 = pd.read_csv(path3, sep=",")

        return pd.concat([df1, df2, df3])


class DryBean(UCIDatasetInterface):
    column_names = [
        "Area",
        "Perimeter",
        "MajorAxisLength",
        "MinorAxisLength",
        "AspectRation",
        "Eccentricity",
        "ConvexArea",
        "EquivDiameter",
        "Extent",
        "Solidity",
        "roundness",
        "Compactness",
        "ShapeFactor1",
        "ShapeFactor2",
        "ShapeFactor3",
        "ShapeFactor4",
        "Class",
    ]
    columns_to_drop = []
    categorical_columns = []
    target_column = "Class"
    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip"

    def df_loader(self):
        u = requests.get(self.uci_url)

        with zipfile.ZipFile(BytesIO(u.content), "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)

        path = Path(self.raw_dir, "DryBeanDataset", "Dry_Bean_Dataset.xlsx")
        df = pd.read_excel(path)

        return df
