from logging import raiseExceptions

import numpy as np
import cvxpy as cp
from src.ybus import ybus


class distr_network:
    ## data class of distribution network
    def __init__(self, buses, lines, generators, baseMVA):
        self.buses = buses
        self.lines = lines
        self.generators = generators

        self.nline = len(lines)
        self.nbus = len(buses)
        self.ngen = len(generators)
        self.lineset = np.arange(0, self.nline)
        self.busset = np.arange(0, self.nbus)
        self.genset = np.arange(0, self.ngen)
        self.baseMVA = baseMVA

        self.B_g = []
        for g in self.genset:
            self.B_g.append(self.generators[g].location)

        self.B_gn = []
        for i in self.busset:
            self.B_gn.append([])

        for g in self.genset:
            self.B_gn[self.generators[g].location].append(int(g))

        #%%
        self.Ybus, self.yff, self.yft, self.ytf, self.ytt = ybus(buses, lines)

        self.yff_r = np.real(self.yff)
        self.yff_i = np.imag(self.yff)
        self.ytt_r = np.real(self.ytt)
        self.ytt_i = np.imag(self.ytt)
        self.yft_r = np.real(self.yft)
        self.yft_i = np.imag(self.yft)
        self.ytf_r = np.real(self.ytf)
        self.ytf_i = np.imag(self.ytf)




def DC_OPF_cvx(net, pg_was_idx, pg_han_idx, pg_han, Pd=None):  
    ## generate the DC OPF optimization problem with cvx. pg_was: wait and see generations, tbd, pg_han: here and now generation, given as either a cvxpy variable or a np array.
    if Pd is None:
        Pd = np.zeros(net.busset.shape[0])
        for b in net.busset:
            Pd[b] = net.buses[b].Pd

    ngen = len(net.genset)
    nbus = len(net.busset)
    nline = len(net.lineset)
    theta = cp.Variable(nbus)
    npg_was = pg_was_idx.shape[0]

    npg_han = pg_han.shape[0]

    pg_was = cp.Variable(npg_was)
    c1 = np.zeros(
        [ngen, npg_was]
    )  # this is due to cvxpy's setting that one cannot declare an empty expression, c1@pg_was + c2@pg_han = gen, the generation vector.
    for i in range(0, npg_was):
        c1[pg_was_idx[i], i] = 1
    c2 = np.zeros([ngen, npg_han])
    for i in range(0, npg_han):
        c2[pg_han_idx[i], i] = 1

    pg = c1 @ pg_was + c2 @ pg_han
    p_ft = cp.Variable(nline)
    p_tf = cp.Variable(nline)

    constraints = []
    for b in net.busset:
        constraints += [-np.pi <= theta[b]]
        constraints += [theta[b] <= +np.pi]
        if net.buses[b].btype == 3:
            constraints += [theta[b] == 0]

    for g in net.genset:
        constraints += [net.generators[g].Pmin <= pg[g]]
        constraints += [pg[g] <= net.generators[g].Pmax]

    for l in net.lineset:
        constraints += [
            p_ft[l]
            == float(net.yft_i[l])
            * (theta[net.lines[l].fbus] - theta[net.lines[l].tbus])
        ]
        constraints += [
            p_tf[l]
            == float(net.ytf_i[l])
            * (theta[net.lines[l].tbus] - theta[net.lines[l].fbus])
        ]

    for b in net.busset:
        constraints += [
            sum(p_ft[l] for l in net.buses[b].outline)
            + sum(p_tf[l] for l in net.buses[b].inline)
            - sum(pg[g] for g in net.B_gn[b])
            + Pd[b]
            + net.buses[b].Gs * 1.0 ** 2
            == 0
        ]

    for l in net.lineset:
        if net.lines[l].u != 0:
            constraints += [p_ft[l] <= +net.lines[l].u]
            constraints += [p_ft[l] >= -net.lines[l].u]
            constraints += [p_tf[l] <= +net.lines[l].u]
            constraints += [p_tf[l] >= -net.lines[l].u]
        constraints += [
            theta[net.lines[l].fbus] - theta[net.lines[l].tbus] <= net.lines[l].angmax
        ]
        constraints += [
            theta[net.lines[l].fbus] - theta[net.lines[l].tbus] >= net.lines[l].angmin
        ]
        # constraints += [ theta[net.lines[l].fbus] - theta[net.lines[l].tbus] <= min(net.lines[l].angmax, radians(60)) ]
        # constraints += [ theta[net.lines[l].fbus] - theta[net.lines[l].tbus] >= max(net.lines[l].angmin, -radians(60)) ]

    gen_cost = sum(
        (
            net.generators[g].cost[0] * (pg[g] * net.baseMVA) ** 2
            + net.generators[g].cost[1] * (pg[g] * net.baseMVA)
            + net.generators[g].cost[2]
        )
        for g in net.genset
    )

    return gen_cost, constraints, pg_han, pg_was


def DC_OPF_cvx_relaxed(net, pg_was_idx, pg_han_idx, pg_han, Pd=None):
    ## generate the DC OPF optimization problem with cvx, with slack relaxation on the equality constraints. pg_was: wait and see generations, tbd, pg_han: here and now generation, given as either a cvxpy variable or a np array.
    if Pd is None:
        Pd = np.zeros(net.busset.shape[0])
        for b in net.busset:
            Pd[b] = net.buses[b].Pd

    ngen = len(net.genset)
    nbus = len(net.busset)
    nline = len(net.lineset)
    theta = cp.Variable(nbus)
    npg_was = pg_was_idx.shape[0]
    npg_han = pg_han.shape[0]
    slack = cp.Variable(1)
    pg_was = cp.Variable(npg_was)
    constraints = []
    c1 = np.zeros(
        [ngen, npg_was]
    )  # this is due to cvxpy's setting that one cannot declare an empty expression, c1@pg_was + c2@pg_han = gen, the generation vector.
    for i in range(0, npg_was):
        c1[pg_was_idx[i], i] = 1
    c2 = np.zeros([ngen, npg_han])
    for i in range(0, npg_han):
        c2[pg_han_idx[i], i] = 1

    pg = c1 @ pg_was + c2 @ pg_han
    p_ft = cp.Variable(nline)
    p_tf = cp.Variable(nline)
    for b in net.busset:
        constraints += [-np.pi <= theta[b]]
        constraints += [theta[b] <= np.pi]
        if net.buses[b].btype == 3:
            constraints += [theta[b] == 0]

    for g in net.genset:
        constraints += [net.generators[g].Pmin <= pg[g]]
        constraints += [pg[g] <= net.generators[g].Pmax]

    for l in net.lineset:
        constraints += [
            p_ft[l]
            == float(net.yft_i[l])
            * (theta[net.lines[l].fbus] - theta[net.lines[l].tbus])
        ]
        constraints += [
            p_tf[l]
            == float(net.ytf_i[l])
            * (theta[net.lines[l].tbus] - theta[net.lines[l].fbus])
        ]

    for b in net.busset:
        constraints += [
            sum(p_ft[l] for l in net.buses[b].outline)
            + sum(p_tf[l] for l in net.buses[b].inline)
            - sum(pg[g] for g in net.B_gn[b])
            + Pd[b]
            + net.buses[b].Gs * 1.0 ** 2
            >= -slack,
            sum(p_ft[l] for l in net.buses[b].outline)
            + sum(p_tf[l] for l in net.buses[b].inline)
            - sum(pg[g] for g in net.B_gn[b])
            + Pd[b]
            + net.buses[b].Gs * 1.0 ** 2
            <= slack,
        ]

    for l in net.lineset:
        if net.lines[l].u != 0:
            constraints += [p_ft[l] <= +net.lines[l].u]
            constraints += [p_ft[l] >= -net.lines[l].u]
            constraints += [p_tf[l] <= +net.lines[l].u]
            constraints += [p_tf[l] >= -net.lines[l].u]
        constraints += [
            theta[net.lines[l].fbus] - theta[net.lines[l].tbus] <= net.lines[l].angmax
        ]
        constraints += [
            theta[net.lines[l].fbus] - theta[net.lines[l].tbus] >= net.lines[l].angmin
        ]
    constraints += [slack >= 0]
    # constraints += [ theta[net.lines[l].fbus] - theta[net.lines[l].tbus] <= min(net.lines[l].angmax, radians(60)) ]
    # constraints += [ theta[net.lines[l].fbus] - theta[net.lines[l].tbus] >= max(net.lines[l].angmin, -radians(60)) ]

    gen_cost = sum(
                (
                    net.generators[g].cost[0] * (pg[g] * net.baseMVA) ** 2
                    + net.generators[g].cost[1] * (pg[g] * net.baseMVA)
                    + net.generators[g].cost[2]
                )
                for g in net.genset
            )
    total_cost = gen_cost*1e-2+slack*1e4

    return total_cost, constraints, pg_han, pg_was, slack, theta

def DC_OPF_cvx_2s(net, pg_was_idx, pg_han_idx, pg_han, pg_was_ref, Pd=None, case=0):  
    ## generate the two-stage DC OPF optimization problem with adjusting cost. pg_was: wait and see generations, tbd, pg_han: here and now generation, given as either a cvxpy variable or a np array.
    if Pd is None:
        Pd = np.zeros(net.busset.shape[0])
        for b in net.busset:
            Pd[b] = net.buses[b].Pd

    ngen = len(net.genset)
    nbus = len(net.busset)
    nline = len(net.lineset)
    theta = cp.Variable(nbus)
    npg_was = pg_was_idx.shape[0]
    npg_han = pg_han.shape[0]
    slack = cp.Variable(1)
    constraints = []
    if case==0:
        pg_was = cp.Variable(npg_was)
    elif case==1:
        pos_err = cp.Variable(npg_was)
        neg_err = cp.Variable(npg_was)
        pg_was = pg_was_ref+pos_err-neg_err
        constraints+=[pos_err>=0,neg_err>=0]
    else:
        raise Exception("case not implemented")
    
    c1 = np.zeros(
        [ngen, npg_was]
    )
    for i in range(0, npg_was):
        c1[pg_was_idx[i], i] = 1
    c2 = np.zeros([ngen, npg_han])
    for i in range(0, npg_han):
        c2[pg_han_idx[i], i] = 1

    pg = c1 @ pg_was + c2 @ pg_han
    pg_ref = c1 @ pg_was_ref + c2@ pg_han
    p_ft = cp.Variable(nline)
    p_tf = cp.Variable(nline)
    for b in net.busset:
        constraints += [-np.pi <= theta[b]]
        constraints += [theta[b] <= np.pi]
        if net.buses[b].btype == 3:
            constraints += [theta[b] == 0]

    for g in net.genset:
        constraints += [net.generators[g].Pmin <= pg[g]]
        constraints += [pg[g] <= net.generators[g].Pmax]

    for l in net.lineset:
        constraints += [
            p_ft[l]
            == float(net.yft_i[l])
            * (theta[net.lines[l].fbus] - theta[net.lines[l].tbus])
        ]
        constraints += [
            p_tf[l]
            == float(net.ytf_i[l])
            * (theta[net.lines[l].tbus] - theta[net.lines[l].fbus])
        ]

    for b in net.busset:
        constraints += [
            sum(p_ft[l] for l in net.buses[b].outline)
            + sum(p_tf[l] for l in net.buses[b].inline)
            - sum(pg[g] for g in net.B_gn[b])
            + Pd[b]
            + net.buses[b].Gs * 1.0 ** 2
            >= -slack,
            sum(p_ft[l] for l in net.buses[b].outline)
            + sum(p_tf[l] for l in net.buses[b].inline)
            - sum(pg[g] for g in net.B_gn[b])
            + Pd[b]
            + net.buses[b].Gs * 1.0 ** 2
            <= slack,
        ]

    for l in net.lineset:
        if net.lines[l].u != 0:
            constraints += [p_ft[l] <= +net.lines[l].u]
            constraints += [p_ft[l] >= -net.lines[l].u]
            constraints += [p_tf[l] <= +net.lines[l].u]
            constraints += [p_tf[l] >= -net.lines[l].u]
        constraints += [
            theta[net.lines[l].fbus] - theta[net.lines[l].tbus] <= net.lines[l].angmax
        ]
        constraints += [
            theta[net.lines[l].fbus] - theta[net.lines[l].tbus] >= net.lines[l].angmin
        ]
    constraints += [slack >= 0]
    # constraints += [ theta[net.lines[l].fbus] - theta[net.lines[l].tbus] <= min(net.lines[l].angmax, radians(60)) ]
    # constraints += [ theta[net.lines[l].fbus] - theta[net.lines[l].tbus] >= max(net.lines[l].angmin, -radians(60)) ]

    if case==0:
        ## quadratic adjusting cost
        gen_cost = (
            sum(
                (
                    net.generators[g].cost[0] * (pg_ref[g] * net.baseMVA) ** 2
                    + net.generators[g].cost[1] * (pg_ref[g] * net.baseMVA)
                    + net.generators[g].cost[2]
                )
                for g in net.genset
            )
            + sum([net.generators[pg_was_idx[k]].cost[0]*100 * (pg_was[k]-pg_was_ref[k])**2 * net.baseMVA**2  for k in range(npg_was)])
        )
    elif case==1:
        ## piecewise linear adjusting cost
        gen_cost = (
            sum(
                (
                    net.generators[g].cost[0] * (pg_ref[g] * net.baseMVA) ** 2
                    + net.generators[g].cost[1] * (pg_ref[g] * net.baseMVA)
                    + net.generators[g].cost[2]
                )
                for g in net.genset
            )
            + sum([net.generators[pg_was_idx[k]].cost[1]*3 * pos_err[k] * net.baseMVA  for k in range(npg_was)])
        )
    total_cost = gen_cost*1e-2+slack*1e5

    return total_cost, constraints, pg_han, pg_was, slack, theta
