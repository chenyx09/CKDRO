from ckdro import *
import numpy as np
from utils import *
from DCOPF import *
import math
import dill
from functools import partial
import matplotlib.pyplot as plt
import random
import multiprocessing as mp


def DRO_solve(net, prob, x, y, pg_han_idx, pg_was_idx, solver, loss_case=0, top_N=None, eps=None):
    ## solve the DCOPF first stage problem with CKDRO

    res = prob.robust_opt(
        x, solver=solver, verbose=False, top_N=top_N, eps=eps)
    pg_han_DRO = res["theta"][:pg_han_idx.shape[0]]
    pg_was_DRO = res["theta"][pg_han_idx.shape[0]:]
    cost, constr, _, pg_was, slack, _ = DC_OPF_cvx_2s(
        net, pg_was_idx, pg_han_idx, pg_han_DRO, pg_was_DRO, y, case=loss_case
    )
    opt = cp.Problem(cp.Minimize(cost), constr)
    opt.solve(solver=solver, verbose=False)
    cost_DRO = cost.value
    return cost_DRO, pg_han_DRO, pg_was_DRO, slack.value


def worst_case_cost(net, pg_han, pg_was_ref, pg_han_idx, pg_was_idx, solver, Y_nom, loss_case=0, std=2):
    ## evaluate the worst case performance by randomly adding Gaussian noise to the load profile
    cost_set = list()
    for i in range(10):
        y = Y_nom+np.random.normal(np.zeros(8), np.ones(8)*std**2)
        cost, constr, _, pg_was, slack, theta = DC_OPF_cvx_2s(
            net, pg_was_idx, pg_han_idx, pg_han, pg_was_ref, y, case=loss_case
        )
        opt = cp.Problem(cp.Minimize(cost), constr)
        opt.solve(solver=solver, verbose=False)
        cost_set.append(cost.value)
    return max(cost_set)


def calc_OPF_res(sample_idx, dim_theta, Kern_x, Kern_y, Xobs, Yobs, Ycert, top_N, net, pg_han_idx, pg_was_idx, Y_mean, solver, std, loss_case=0):
    ## for a particular load case, evaluate various solution and return their performance
    print(sample_idx)
    dis = Kern_x(Xobs, Xobs[sample_idx:sample_idx+1]).flatten()
    indices = dis.argsort()[-top_N-1:-1]

    def loss_call(pg, pd): return DC_OPF_cvx_2s(
        net, pg_was_idx, pg_han_idx, pg[:pg_han_idx.shape[0]
                                        ], pg[pg_han_idx.shape[0]:], pd, case=loss_case
    )[0:2]
    prob = CKdrocombinedK(
        dim_theta,
        loss_call,
        Kern_x,
        Kern_y,
        Xobs[indices],
        Yobs[indices],
        Ycert,
        lam=1e-2,
        nested=True,
    )

    pg_han = cp.Variable(7)
    pg_was_ref = cp.Variable(6)
    cost, constr, _, pg_was, slack, theta = DC_OPF_cvx_2s(
        net, pg_was_idx, pg_han_idx, pg_han, pg_was_ref, Yobs[sample_idx], case=loss_case
    )
    opt = cp.Problem(cp.Minimize(cost), constr)
    opt.solve(solver=solver, verbose=False)
    pg_han_opt = pg_han.value
    c_opt = cost.value
    cost, pg_han_DRO, pg_was_DRO, slack_val = DRO_solve(
        net, prob, Xobs[sample_idx], Yobs[sample_idx], pg_han_idx, pg_was_idx, solver, loss_case, top_N)
    c_DRO = cost

    # interpolation
    Y_interp = prob.get_interpolation(Xobs[sample_idx])
    cost, constr, pg_han, pg_was, slack, theta = DC_OPF_cvx_relaxed(
        net, pg_was_idx, pg_han_idx, pg_han, Y_interp
    )
    opt = cp.Problem(cp.Minimize(cost), constr)
    opt.solve(solver=solver, verbose=False)
    pg_han_interp = pg_han.value
    pg_was_interp = pg_was.value
    cost, constr, _, pg_was, slack, theta = DC_OPF_cvx_2s(
        net, pg_was_idx, pg_han_idx, pg_han_interp, pg_was_interp, Yobs[
            sample_idx], case=loss_case
    )
    opt = cp.Problem(cp.Minimize(cost), constr)
    opt.solve(solver=solver, verbose=False)
    c_interp = cost.value
    c_interp_wc = worst_case_cost(
        net, pg_han_interp, pg_was_interp, pg_han_idx, pg_was_idx, solver, Yobs[sample_idx], loss_case, std)

    cost, pg_han_DRO, pg_was_DRO, slack_val = DRO_solve(
        net, prob, Xobs[sample_idx], Yobs[sample_idx], pg_han_idx, pg_was_idx, solver, loss_case, top_N)
    c_DRO = cost
    c_DRO_wc = worst_case_cost(
        net, pg_han_DRO, pg_was_DRO, pg_han_idx, pg_was_idx, solver, Yobs[sample_idx], loss_case, std)

    # high eps
    cost, pg_han_high, pg_was_high, slack_val = DRO_solve(
        net, prob, Xobs[sample_idx], Yobs[sample_idx], pg_han_idx, pg_was_idx, solver, loss_case, top_N, eps=0.5)
    c_DRO_high = cost
    c_DRO_high_wc = worst_case_cost(
        net, pg_han_high, pg_was_high, pg_han_idx, pg_was_idx, solver, Yobs[sample_idx], loss_case, std)
    # low eps
    cost, pg_han_low, pg_was_low, slack_val = DRO_solve(
        net, prob, Xobs[sample_idx], Yobs[sample_idx], pg_han_idx, pg_was_idx, solver, loss_case, top_N, eps=0.12)
    c_DRO_low = cost
    c_DRO_low_wc = worst_case_cost(
        net, pg_han_low, pg_was_low, pg_han_idx, pg_was_idx, solver, Yobs[sample_idx], loss_case, std)

    pg_han = cp.Variable(7)
    cost, constr, pg_han, pg_was, slack, theta = DC_OPF_cvx_relaxed(
        net, pg_was_idx, pg_han_idx, pg_han, Y_mean
    )

    opt = cp.Problem(cp.Minimize(cost), constr)
    opt.solve(solver=solver, verbose=False)
    pg_han_2step = pg_han.value
    pg_was_2step = pg_was.value
    cost, constr, _, pg_was, slack, theta = DC_OPF_cvx_2s(
        net, pg_was_idx, pg_han_idx, pg_han_2step, pg_was_2step, Yobs[
            sample_idx], case=loss_case
    )
    opt = cp.Problem(cp.Minimize(cost), constr)
    opt.solve(solver=solver, verbose=False)
    c_2step = cost.value
    c_2step_wc = worst_case_cost(
        net, pg_han_2step, pg_was_2step, pg_han_idx, pg_was_idx, solver, Yobs[sample_idx], loss_case, std)

    print(
        f"optimal: {c_opt}, DRO: {c_DRO}, 2 step: {c_2step}, DRO WC: {c_DRO_wc}, 2 step WC: {c_2step_wc}")
    return c_opt, c_DRO, c_2step, c_DRO_wc, c_2step_wc, c_DRO_high, c_DRO_low, c_DRO_high_wc, c_DRO_low_wc, c_interp, c_interp_wc


def disfun_temp(x, y):
    return abs(x - y) * 0.02


def disfun_rain(x, y):
    return abs(x - y) * 0.05



def record_result(result, result_list):
    result_list.append(result)


def main_demand(case=0):
    testsystem = "caseERCOT_new"
    buses, lines, generators, datamat = readnetwork(testsystem)
    net = distr_network(buses, lines, generators, datamat)

    Pd = np.zeros(net.nbus)
    for i in range(net.nbus):
        Pd[i] = net.buses[i].Pd
    pg_was_idx = np.array([0, 3, 6, 8, 9, 11])
    pg_han_idx = np.array([1, 2, 4, 5, 7, 10, 12])

    data_file = "data/ERCOT/ERCOT_data.pkl"
    with open(data_file, "rb") as f:
        weather_data, load_data = dill.load(f)

    std=1.0
    num_cert=100
    cert_std=0.5
    top_N=70
    sigma_x=0.1
    sigma_y=0.2

    disfun_hour = partial(cyclic_dis, cycle=24.0)

    disfun_date = partial(cyclic_dis, cycle=8760)

    disfuns = [disfun_hour, disfun_temp, disfun_rain, disfun_date]
    dis_fun_index = np.array(
        [0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3])
    comp_disfun = partial(compose_dis_fun, disfuns=disfuns,
                          dis_fun_index=dis_fun_index)
    Kern_x = partial(Gaussian_comp_kern, disfun=comp_disfun, sigma=sigma_x)

    Kern_y = partial(Gaussian_kern, sigma=sigma_y)
    Xobs = weather_data[:730].to_numpy()
    Yobs = load_data[:730].to_numpy() / net.baseMVA
    Ycert = Yobs[np.random.choice(Yobs.shape[0], num_cert)] + np.random.multivariate_normal(
        np.zeros(Yobs.shape[1]), cert_std*np.cov(Yobs.T), num_cert)
    solver = cp.MOSEK
    loss_case = case

    dim_theta = pg_han_idx.shape[0]+pg_was_idx.shape[0]

    result_list = list()
    sample_indices = random.choices(
        range(Xobs.shape[0]), weights=[1]*Xobs.shape[0], k=60)
    Y_mean = Yobs.mean(axis=0)
    pool = mp.Pool(mp.cpu_count())

    for idx in sample_indices:
        pool.apply_async(calc_OPF_res, args=(idx, dim_theta, Kern_x, Kern_y, Xobs, Yobs, Ycert, top_N, net, pg_han_idx,
                         pg_was_idx, Y_mean, solver, std, loss_case, ), callback=lambda x: record_result(x, result_list))

    pool.close()
    pool.join()
    results = np.array(result_list).squeeze(-1)

    sort_idx = results[:,0].argsort()
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,6))
    ax1.plot(np.arange(results.shape[0]),results[sort_idx,0],color='g',marker=".",linestyle = 'None',label="optimal")
    ax1.plot(np.arange(results.shape[0]),results[sort_idx,1],color='b',marker="v",linestyle = 'None',label="CKDRO, nominal")
    ax1.plot(np.arange(results.shape[0]),results[sort_idx,2],color='c',marker="o",linestyle = 'None',label="2 stage with average $P^d$, nominal")
    ax1.plot(np.arange(results.shape[0]),results[sort_idx,3],color='m',marker="x",linestyle = 'None',label="CKDRO, worst case")
    ax1.plot(np.arange(results.shape[0]),results[sort_idx,4],color='k',marker="s",linestyle = 'None',label="2 stage with average $P^d$, worst case")
    ax1.plot(np.arange(results.shape[0]),results[sort_idx,9],color='tab:orange',marker="d",linestyle = 'None',label="Interpolation, nominal")
    ax1.plot(np.arange(results.shape[0]),results[sort_idx,10],color='y',marker="h",linestyle = 'None',label="Interpolation, worst case")
    ax1.legend(prop={'size': 12},frameon=False)
    ax2.plot(np.arange(results.shape[0]),results[sort_idx,1],color='b',marker="v",linestyle = 'None',label="Adaptive $\epsilon$, nominal")
    ax2.plot(np.arange(results.shape[0]),results[sort_idx,5],color='k',marker="s",linestyle = 'None',label="$\epsilon=3$, nominal")
    ax2.plot(np.arange(results.shape[0]),results[sort_idx,6],color='c',marker="p",linestyle = 'None',label="$\epsilon=0.15$, nominal")
    ax2.plot(np.arange(results.shape[0]),results[sort_idx,3],color='m',marker="x",linestyle = 'None',label="Adaptive $\epsilon$, worst case")
    ax2.plot(np.arange(results.shape[0]),results[sort_idx,7],color='y',marker="h",linestyle = 'None',label="$\epsilon=3$, worst case")
    ax2.plot(np.arange(results.shape[0]),results[sort_idx,8],color='tab:orange',marker="d",linestyle = 'None',label="$\epsilon=0.15$, worst case")
    ax2.legend(prop={'size': 1},frameon=False)
    plt.show()



if __name__ == "__main__":
    main_demand()