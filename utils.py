import cvxpy as cp
import numpy as np
from functools import partial
import pdb


def Gaussian_kern(x, y, sigma=1):
    s2 = sigma ** 2
    K = np.zeros([x.shape[0], y.shape[0]])
    for i in range(x.shape[0]):
        for j in range(0, y.shape[0]):
            K[i, j] = np.exp(-((x[i] - y[j]).dot(x[i] - y[j]))/x.shape[-1] / 2 / s2)
    return K


class costFun:
    # cost function
    def __init__(self, method="boyd", model=None, mode="casadi"):
        # weights
        self.method = method
        self.model = model
        self.mode = mode

    def __call__(self, x, w):
        return self.eval(x, w)

    def eval(self, x, w):
        """
        evaluate the cost function, with casadi operation
        input:
        x: primal decision var
        w: RV, randomness, in ml: it's the data
        """
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        elif len(x.shape) == 2:
            pass
        else:
            raise NotImplementedError

        if self.method == "boyd":
            """
            Boyd & Vandenberghe book. Figure 6.15.

            min_x || A(w) - b0 ||^2 where
            A(w) := A0 + w * B0 and w is a scalar.
            """
            # boy data set
            A0, B0, b0 = self.model  # model of the optimization problem
            if self.mode == "casadi":
                from casadi import sumsqr

                cost_val = sumsqr((A0 + w * B0) @ x - b0)
            elif self.mode == "cvxpy":
                import cvxpy as cp

                cost_val = cp.sum_squares((A0 + w * B0) @ x - b0)
            elif self.mode == "numpy":
                cost_val = np.sum(((A0 + w * B0) @ x - b0) ** 2)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return cost_val


class sqaure_loss:
    # cost function
    def __init__(self, mode="casadi"):
        # weights

        self.mode = mode

    def __call__(self, x, w):
        return self.eval(x, w)

    def eval(self, x, w):
        """
        evaluate the cost function, with casadi operation
        input:
        x: primal decision var
        w: RV, randomness, in ml: it's the data
        """
        # if len(x.shape)==1:
        #     x=x.reshape(-1,1)
        # elif len(x.shape)==2:
        #     pass
        # else:
        #     raise NotImplementedError

        if self.mode == "casadi":
            from casadi import sumsqr

            return sumsqr(x - w)
        elif self.mode == "cvxpy":
            return cp.sum_squares(x - w)
        elif self.mode == "numpy":
            return np.sum((x - w) ** 2)

        else:
            raise NotImplementedError


def cyclic_dis(x, y, cycle, scale=1.0):
    return np.minimum(np.mod(x - y, cycle), np.mod(y - x, cycle)) / cycle * scale


def compose_dis_fun(x, y, disfuns, dis_fun_index):
    assert x.shape[1] == y.shape[1]
    dis = np.zeros([x.shape[0], y.shape[0], x.shape[1]])
    if x.shape[0] <= y.shape[0]:
        for i in range(x.shape[0]):
            for n, fun in enumerate(disfuns):
                dis[i] += fun(x[i], y) * np.expand_dims(dis_fun_index == n, 0)

    else:
        for i in range(y.shape[0]):
            for n, fun in enumerate(disfuns):
                dis[:, i] += fun(x, y[i]) * np.expand_dims(dis_fun_index == n, 0)
    return dis


def Gaussian_comp_kern(x, y, disfun, sigma):
    assert x.shape[1] == y.shape[1]
    dis = disfun(x, y)
    dis_norm_square = np.linalg.norm(dis, axis=-1) ** 2 / dis.shape[-1]
    return np.exp(-dis_norm_square / 2 / (sigma ** 2))


# fun1 = partial(cyclic_dis, cycle=0.05)
# fun2 = lambda x, y: abs(x - y)

# disfuns = [fun1, fun2]
# dis_fun_index = np.array([0, 1, 0, 1])
# x = np.random.rand(3, 4)
# y = np.random.rand(5, 4)
# dis = compose_dis_fun(x, y, disfuns, dis_fun_index)
# disfun = partial(compose_dis_fun, disfuns=disfuns, dis_fun_index=dis_fun_index)
# K = Gaussian_comp_kern(x, x, disfun, sigma=1)
# print(K)
