"""
  On Class Distributions Induced by Nearest Neighbor Graphs for Node Classification of Tabular Data

  File:     launch_class_separator_exp.py
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

import os

import torch
from torch.optim import SGD, Adam

from class_separator import ClassSeparator
from tqdm import tqdm


def main():
    seed = 42
    torch.manual_seed(seed)
    device = "cuda:0"

    for num_features in [1, 2, 5, 10, 50]:
        for num_classes in [2, 5]:
            for num_mixtures in [2, 5, 10]:
                # lambda1 controls the importance of the CCNS lower bound term
                # minimizing deltaSED might lead to different class distributions
                # being completely identical, and we want to push the model away
                # from a naive solution that possibly represents a local minimum
                # in the loss landscape.
                for lambda1 in [0.0, 1.0, 5.0, 10.0]:
                    retry = True
                    while retry:
                        retry = False
                        try:

                            model_ckpt_pth = (
                                f"C_{num_classes}_M_{num_mixtures}_"
                                f"D_{num_features}_"
                                f"lambda_{lambda1}_"
                                f"seed_{seed}.pth"
                            )
                            results_ckpt_pth = (
                                f"C_{num_classes}_M_{num_mixtures}_"
                                f"D_{num_features}_"
                                f"lambda_{lambda1}_"
                                f"seed_{seed}.pt"
                            )

                            if os.path.exists(
                                os.path.join("RESULTS", results_ckpt_pth)
                            ):
                                print(
                                    "Skipping configuration due to existing results"
                                )
                                continue

                            # k, sigma, epsilon are optimized by the method
                            model = ClassSeparator(
                                num_classes=num_classes,
                                num_mixtures=num_mixtures,
                                num_features=num_features,
                                use_full_covariance=(
                                    lambda1 == 0.0 and num_features > 1
                                ),
                            )

                            epochs = 10000
                            patience = 3000
                            best_epoch = 0

                            optimizer = Adam(
                                params=model.parameters(), lr=0.01
                            )

                            load_checkpoint = False
                            if load_checkpoint:
                                p = os.path.join(
                                    "./checkpoints", model_ckpt_pth
                                )
                                if os.path.exists(p):
                                    model.load_state_dict(
                                        torch.load(
                                            p,
                                            map_location="cpu",
                                        )
                                    )

                            model.to(device)
                            model.train()

                            print(
                                f"Start training with D={num_features}, "
                                f"|C|={num_classes}, |M|={num_mixtures}, "
                                f"lambda={lambda1}..."
                            )

                            with tqdm(total=epochs) as tepoch:
                                best_deltaSED = torch.inf
                                for epoch in range(1, epochs + 1):

                                    optimizer.zero_grad()
                                    sed_tuples, lb_ccns, m_cc = model()

                                    deltaSED = 0.0

                                    # combine all SED differences for each pair of c,c'
                                    for s in sed_tuples:
                                        # We want to maximize SED_H - SED_X,
                                        # so we append a minus and minimize deltaSED
                                        deltaSED -= s[0]

                                    if lambda1 != 0.0:
                                        # compute the regularizer based on lower
                                        # bound of the CCNS
                                        loss_lb_ccns = 0.0
                                        for c_prime in range(model.C):
                                            for c in range(model.C):
                                                if c_prime == c:
                                                    # maximize intra-class ccns
                                                    loss_lb_ccns -= lb_ccns[
                                                        c, c_prime
                                                    ]
                                                else:
                                                    # minimize inter-class ccns
                                                    loss_lb_ccns += lb_ccns[
                                                        c, c_prime
                                                    ]

                                        loss = (
                                            deltaSED + lambda1 * loss_lb_ccns
                                        )
                                    else:
                                        loss = deltaSED

                                    loss.backward()

                                    optimizer.step()

                                    tepoch.set_description(f"Epoch {epoch}")
                                    tepoch.set_postfix(
                                        loss=loss.item(),
                                        deltaSED=deltaSED.item(),
                                    )
                                    tepoch.update(1)

                                    if deltaSED.item() < best_deltaSED:
                                        best_epoch = epoch
                                        best_deltaSED = (
                                            deltaSED.detach().item()
                                        )
                                        best_lb_ccns = (
                                            lb_ccns.detach()
                                            if lambda1 > 0.0
                                            else None
                                        )
                                        best_sed_tuples = (
                                            [
                                                (t[1], t[2], t[3], t[4])
                                                for t in sed_tuples
                                            ],
                                        )
                                        if not os.path.exists("./checkpoints"):
                                            os.makedirs("./checkpoints")
                                        torch.save(
                                            model.state_dict(),
                                            "./checkpoints/" + model_ckpt_pth,
                                        )

                                    if epoch - best_epoch > patience:
                                        print(
                                            f"Stopping training - early stopping"
                                        )
                                        break

                            if not os.path.exists("./RESULTS"):
                                os.makedirs("./RESULTS")

                            torch.save(
                                {
                                    "best_epoch": best_epoch,
                                    "best_sed_tuples": best_sed_tuples,
                                    "best_deltaSED": best_deltaSED,
                                    "best_lb_ccns": best_lb_ccns,
                                    "C": num_classes,
                                    "M": num_mixtures,
                                    "D": num_features,
                                    "lambda": lambda1,
                                    "seed": seed,
                                },
                                os.path.join("RESULTS", results_ckpt_pth),
                            )

                            print(f"Training of configuration has finished.")

                        except Exception as e:
                            print("Exception caught, retying...")
                            retry = True


if __name__ == "__main__":
    main()
