import numpy as np
import cvxpy as cp
from abc import ABC
import abc
import ecos
from scipy import sparse
import datetime


class AbstractKdro(ABC):
    @abc.abstractmethod
    def robust_opt(self):
        raise NotImplementedError()


def matDecomp(K):
    # import scipy
    # decompose matrix
    try:
        L = np.linalg.cholesky(K)
    except:
        # print('warning, Gram matrix K is singular')
        d, v = np.linalg.eigh(
            K
        )  # L == U*diag(d)*U'. the scipy function forces real eigs
        d[np.where(d < 0)] = 0  # get rid of small eigs
        L = v @ np.diag(np.sqrt(d))
    return L


class CKdrocombinedK(AbstractKdro):
    def __init__(
        self,
        dim_theta,
        loss_call,
        Kern_x,
        Kern_y,
        Xobs,
        Yobs,
        Ycert,
        lam,
        kappa=1e-3,
        nested=False,
    ):
        """
        dim_theta: the dimension of theta (parameter to be optimized)

        loss_call: a callable (function)
            (theta, xcert, ycert) |-> loss value

        K: a Gram matrix computed such that
            K_ij = k( (x_i, y_i), (x_j, y_j)) where (x_i, y_i) is an observation,
            in the set (Xobs, Yobs) (set of observations) and (Xcert, Ycert),
            in that order.

        Xobs: N x d numpy array containing N observations.

        Xcert: n x d numpy array containing n input locations to certify
            the robustness . Can be None (empty) in which case the
            optimization is less robust. obs is part of the set of certifying
            points by default.

        """
        assert dim_theta > 0

        self.dim_theta = dim_theta
        self.loss_call = loss_call
        self.Kern_x = Kern_x
        self.Kern_y = Kern_y
        self.KX = Kern_x(Xobs, Xobs)
        self.Xobs = Xobs
        self.Yobs = Yobs
        self.Ycert = Ycert
        self.lam = lam
        self.kappa = kappa
        self.Y = np.vstack((Yobs, Ycert))
        self.KY = Kern_y(self.Y, self.Y)
        self.nested = nested

    def get_interpolation(self, x):
        if len(x.shape) == 1:
            x = np.reshape(x, [1, -1])
        n_sample = self.Xobs.shape[0]
        kxx = np.ones([1, 1]) * self.Kern_x(x, x)
        kx = self.Kern_x(self.Xobs, x)
        K = np.vstack((np.hstack((self.KX, kx)), np.hstack((kx.T, kxx))))
        betax = np.linalg.solve(
            K + self.lam * np.eye(n_sample + 1), np.vstack((kx, kxx))
        ).flatten()
        betax = betax[:-1]/betax[:-1].sum()
        return betax @ self.Yobs

    def robust_opt(self, x, verbose=False, solver=cp.MOSEK, top_N=None, eps=None):
        """perform CKDRO

        Args:
            x (np.ndarray): query auxiliary variable
            verbose (bool, optional): Defaults to False.
            solver (optional): solver for cvxpy Defaults to cp.MOSEK.
            top_N (int, optional): number of closest points considered in CKDRO, when not given, all points are considered. Defaults to None.
            eps (float, optional): epsilon in CKDRO, if given, overwrites the one computed from the data characteristics. Defaults to None.

        Returns:
            A dictionary of results
        """
        if len(x.shape) == 1:
            x = np.reshape(x, [1, -1])
        n_sample = self.Xobs.shape[0]
        kxx = np.ones([1, 1]) * self.Kern_x(x, x)
        kx = self.Kern_x(self.Xobs, x)
        K = np.vstack((np.hstack((self.KX, kx)), np.hstack((kx.T, kxx))))
        betax = np.linalg.solve(
            K + self.lam * np.eye(n_sample + 1), np.vstack((kx, kxx))
        ).flatten()
        if top_N is not None and betax.shape[0] > top_N + 1:
            indices = np.hstack(
                (np.argsort(-abs(betax[:-1]))[:top_N], betax.shape[0] - 1))
            betax = betax[indices]
            betax = betax / betax.sum()
            n_sample = top_N
            Y_indices = np.hstack(
                (indices[:-1], np.arange(self.Yobs.shape[0], self.Y.shape[0]))
            )
            KY = self.KY[np.ix_(Y_indices, Y_indices)]

        else:
            indices = np.arange(betax.shape[0])
            betax = betax / betax.sum()
            KY = self.KY
        if eps is None:
            eps = betax[-1] + self.kappa * n_sample ** -0.25

        # sample size for the set of certification points
        n_certify = self.Ycert.shape[0]

        # All variables to be optimized
        theta = cp.Variable(self.dim_theta)

        # f0 = a bias term as part of the RKHS function. A scalar
        f0 = cp.Variable()

        # Beta is the vector of coefficients of the dual RKHS function.
        beta = cp.Variable(KY.shape[1])

        # function values at the kernel_points
        fvals = KY @ beta

        # List of constraints for cvxpy
        constraints = []
        loss_call = self.loss_call

        # always certify the observations
        if self.nested:
            for i in range(n_sample):
                cost, constr = loss_call(theta, self.Yobs[indices[i]])
                constraints += [cost <= f0 + fvals[i]] + constr

            # certify the certifying points
            for i in range(n_certify):
                # wi = self.cert_locs[i]
                cost, constr = loss_call(theta, self.Ycert[i])
                constraints += [cost <= f0 + fvals[i + n_sample]] + constr
        else:

            for i in range(n_sample):
                constraints += [
                    loss_call(theta, self.Yobs[indices[i]]) <= f0 + fvals[i]
                ]

            # certify the certifying points
            for i in range(n_certify):
                # wi = self.cert_locs[i]
                constraints += [
                    loss_call(theta, self.Ycert[i]) <= f0 + fvals[i + n_sample]
                ]

        emp = f0 + cp.sum(cp.multiply(fvals[:n_sample], betax[:n_sample]))
        # regularization term
        # rkhs_norm = cp.sqrt(cp.quad_form(beta, K + 1e+1*np.eye(K.shape[0])))
        rkhs_norm = cp.norm(beta.T @ matDecomp(KY))

        reg_term = eps * rkhs_norm

        # objective function
        obj = emp + reg_term

        opt = cp.Problem(cp.Minimize(obj), constraints)



        opt.solve(solver=solver, verbose=verbose)
        # result
        R = {
            "theta": theta.value,
            "obj": obj.value,
            "beta": beta.value,
            "f0": f0.value,
            "rkhs_norm": rkhs_norm.value,
        }
        return R


class CKdro_Rand_feat(AbstractKdro):
    """
    Using Gaussian kernel with the random feature approach to approximate functions in RKHS
    """

    def __init__(
        self, dim_theta, loss_call, Xobs, Yobs, Ycert, lam, sigma, kappa=1e-3, D=2000
    ):

        assert dim_theta > 0

        self.dim_theta = dim_theta
        self.loss_call = loss_call
        from utils import Gaussian_kern

        self.Kern = lambda x, y: Gaussian_kern(x, y, sigma)
        sigma2 = 1 / sigma
        self.KX = self.Kern(Xobs, Xobs)
        self.D = D
        n = Yobs.shape[1]
        self.omega = np.random.normal(0, sigma2 ** 2, [D, n])
        self.b = np.random.random(D) * 2 * np.pi
        self.phi_obs = np.zeros([Yobs.shape[0], 2 * D])
        self.phi_cert = np.zeros([Ycert.shape[0], 2 * D])
        for i in range(0, Yobs.shape[0]):
            self.phi_obs[i] = np.sqrt(1 / D) * np.append(
                np.cos(self.omega @ Yobs[i] + self.b),
                np.sin(self.omega @ Yobs[i] + self.b),
            )
        for i in range(0, Ycert.shape[0]):
            self.phi_cert[i] = np.sqrt(1 / D) * np.append(
                np.cos(self.omega @ Ycert[i] + self.b),
                np.sin(self.omega @ Ycert[i] + self.b),
            )
        self.phi = np.vstack((self.phi_obs, self.phi_cert))
        self.Xobs = Xobs
        self.Yobs = Yobs
        self.Ycert = Ycert
        self.lam = lam
        self.kappa = kappa
        self.Y = np.vstack((Yobs, Ycert))

    def robust_opt(self, x, verbose=False, solver=cp.MOSEK):
        """
        eps: epsilon to control the size of the norm ball constraint.

        Return a dictionary of optimization results.
        """
        if len(x.shape) == 1:
            x = np.reshape(x, [1, -1])
        n_sample = self.Xobs.shape[0]
        kxx = np.ones([1, 1]) * self.Kern(x, x)
        kx = self.Kern(self.Xobs, x)
        K = np.vstack((np.hstack((self.KX, kx)), np.hstack((kx.T, kxx))))
        betax = np.linalg.solve(
            K + self.lam * np.eye(n_sample + 1), np.vstack((kx, kxx))
        )

        eps = betax[-1] + self.kappa * n_sample ** -0.25

        # sample size for the set of certification points
        n_certify = self.Ycert.shape[0]

        # All variables to be optimized
        theta = cp.Variable(self.dim_theta)

        # f0 = a bias term as part of the RKHS function. A scalar
        f0 = cp.Variable()

        # Beta is the vector of coefficients of the dual RKHS function.
        beta = cp.Variable(n_sample + n_certify)

        # function values at the kernel_points
        fvals = beta @ self.phi @ self.phi.T

        # fvals = np.vstack((self.phi_obs,self.phi_cert)) @ beta

        # List of constraints for cvxpy
        constraints = []
        loss_call = self.loss_call
        # always certify the observations
        for i in range(n_sample):
            constraints += [loss_call(theta, self.Yobs[i]) <= f0 + fvals[i]]

        # certify the certifying points
        for i in range(n_certify):
            # wi = self.cert_locs[i]
            constraints += [loss_call(theta, self.Ycert[i])
                            <= f0 + fvals[n_sample + i]]

        emp = f0 + fvals[0:n_sample] @ betax[0:n_sample]
        # regularization term
        # rkhs_norm = cp.sqrt(cp.quad_form(beta, K + 1e+1*np.eye(K.shape[0])))
        rkhs_norm = cp.norm(beta @ self.phi)
        reg_term = eps * rkhs_norm

        # objective function
        obj = emp + reg_term
        opt = cp.Problem(cp.Minimize(obj), constraints)


        opt.solve(solver=solver, verbose=verbose)
        # result
        R = {
            "theta": theta.value,
            "obj": obj.value,
            "beta": beta.value,
            "f0": f0.value,
            "rkhs_norm": rkhs_norm.value,
        }
        return R


class CKdrocombinedK_ecos(AbstractKdro):
    def __init__(
        self, R, f, Quad_L, Kern, Xobs, Yobs, Ycert, lam, kappa=1e-3, Lin_L=None
    ):


        self.dim_theta = f.shape[0]
        self.RQ = matDecomp(R)
        self.f = f
        self.Kern = Kern
        self.KX = Kern(Xobs, Xobs)
        self.Xobs = Xobs
        self.Yobs = Yobs
        self.Ycert = Ycert
        self.Ltheta = Quad_L[0]
        self.LX = Quad_L[1]
        self.LY = Quad_L[2]
        self.C = Quad_L[3]
        self.lam = lam
        self.kappa = kappa
        self.Y = np.vstack((Yobs, Ycert))
        self.KY = Kern(self.Y, self.Y)
        self.KYQ = matDecomp(self.KY)
        self.built = False
        self.ndim = None
        self.ndx = None
        self.totaldim = None
        self.Gq0 = None
        self.hq0 = None
        self.dims = None
        if Lin_L is None:
            self.Atheta = None
            self.AX = None
            self.AY = None
            self.b = None
        else:
            self.Atheta = Lin_L[0]
            self.AX = Lin_L[1]
            self.AY = Lin_L[2]
            self.b = Lin_L[3]
        self.Gl0 = None
        self.hl0 = None

    def robust_opt(self, x, verbose=False):
        """
        Return a dictionary of optimization results.
        """
        if len(x.shape) == 1:
            x = np.reshape(x, [1, -1])
        n_sample = self.Xobs.shape[0]
        kxx = np.ones([1, 1]) * self.Kern(x, x)
        kx = self.Kern(self.Xobs, x)
        K = np.vstack((np.hstack((self.KX, kx)), np.hstack((kx.T, kxx))))
        betax = np.linalg.solve(
            K + self.lam * np.eye(n_sample + 1), np.vstack((kx, kxx))
        )

        eps = betax[-1] + self.kappa * n_sample ** -0.25

        # sample size for the set of certification points
        n_certify = self.Ycert.shape[0]
        KY = self.KY
        onevec = np.array([1])
        zerovec = np.array([0])
        Ynum = self.Y.shape[0]
        # All variables to be optimized
        if self.built:
            ndim = self.ndim
            ndx = self.ndx
            totaldim = self.totaldim
            Gq = self.Gq0
            dims = self.dims
            hq = self.hq0
            hqX = np.hstack((zerovec, 2 * self.LX @ np.squeeze(x), zerovec))
            hqX = np.kron(np.ones(Ynum), hqX)
            hqX = np.hstack((hqX, np.zeros(hq.shape[0] - hqX.shape[0])))
            hq = hq + hqX
        else:
            ndim = {
                "theta": self.dim_theta,
                "f0": 1,
                "beta": n_sample + n_certify,
                "J0": 1,
                "J": 1,
            }
            totaldim = 0
            ndx = {}
            for name in ndim:
                ndx[name] = totaldim
                totaldim += ndim[name]
            self.ndx = ndx
            self.ndim = ndim
            self.totaldim = totaldim

            dims = {"q": []}
            Gq = np.empty([0, totaldim])
            hq = np.empty(0)

            # certify the observations, encoded as SOCP constraints.
            for i in range(0, Ynum):
                G1 = np.zeros(totaldim)
                G2 = np.zeros([self.Ltheta.shape[0], totaldim])
                G1[ndx["beta"]: ndx["beta"] + ndim["beta"]] = -KY[i]
                G1[ndx["f0"]] = -1
                G2[:, ndx["theta"]: ndx["theta"] +
                    ndim["theta"]] = -2 * self.Ltheta
                G3 = -G1.copy()
                h2 = 2 * self.LY @ self.Y[i] + 2 * self.C
                Gtemp = np.vstack((G1, G2, G3))
                htemp = np.concatenate((onevec, h2, onevec))
                Gq = np.vstack((Gq, Gtemp))
                hq = np.append(hq, htemp)
                dims["q"].append(Gtemp.shape[0])

            if self.Atheta is not None:
                self.Gl0 = np.zeros([Ynum, totaldim])
                self.hl0 = -self.b * np.ones(Ynum) - self.Y @ self.AY
                self.Gl0[:, ndx["beta"]: ndx["beta"] + ndim["beta"]] = -KY
                self.Gl0[:, ndx["f0"]] = -np.ones(Ynum)
                self.Gl0[:, ndx["theta"]: ndx["theta"] + ndim["theta"]] = np.kron(
                    self.Atheta, np.ones([Ynum, 1])
                )
                dims["l"] = Ynum + 1
            else:
                dims["l"] = 1

            G1 = np.zeros(totaldim)
            G2 = np.zeros([self.dim_theta, totaldim])
            G1[ndx["theta"]: ndx["theta"] + ndim["theta"]] = self.f
            G1[ndx["J0"]] = -1
            G2[:, ndx["theta"]: ndx["theta"] + ndim["theta"]] = -2 * self.RQ
            G3 = -G1.copy()
            Gtemp = np.vstack((G1, G2, G3))
            htemp = np.concatenate((onevec, np.zeros(self.dim_theta), onevec))
            Gq = np.vstack((Gq, Gtemp))
            hq = np.append(hq, htemp)
            dims["q"].append(Gtemp.shape[0])

            dims["q"].append(KY.shape[0] + 1)

            self.dims = dims
            self.Gq0 = Gq
            self.hq0 = hq
            hqX = np.hstack((zerovec, 2 * self.LX @ np.squeeze(x), zerovec))
            hqX = np.kron(np.ones(Ynum), hqX)

            hqX = np.hstack((hqX, np.zeros(hq.shape[0] - hqX.shape[0])))

            hq = hq + hqX

        # encode the cost
        G1 = np.zeros(totaldim)
        G2 = np.zeros([KY.shape[0], totaldim])
        G1[ndx["J"]] = -1
        G1[ndx["J0"]] = 1
        G1[ndx["beta"]: ndx["beta"] + ndim["beta"]] = (
            betax[0:n_sample].T @ KY[0:n_sample]
        )
        G1[ndx["f0"]] = 1
        G2[:, ndx["beta"]: ndx["beta"] + ndim["beta"]] = -self.KYQ.T * eps
        Gtemp = np.vstack((G1, G2))
        htemp = np.zeros(Gtemp.shape[0])
        Gq = np.vstack((Gq, Gtemp))
        hq = np.append(hq, htemp)

        Gl = G1.copy()
        hl = np.zeros(1)
        hlX = -self.AX @ np.squeeze(x) * np.ones(self.Y.shape[0])
        if self.Atheta is not None:
            Gl = np.vstack((Gl, self.Gl0))
            hl = np.append(hl, self.hl0 + hlX)

        G = np.vstack((Gl, Gq))
        h = np.append(hl, hq)
        c = np.zeros(totaldim)
        c[ndx["J"]] = 1
        startTimer = datetime.datetime.now()
        sol = ecos.solve(c, sparse.csc_matrix(G), h, dims, verbose=verbose)
        endTimer = datetime.datetime.now()
        deltaTimer = endTimer - startTimer
        # print("solve time: ", deltaTimer.total_seconds(), " seconds.")
        xsol = sol["x"]

        R = {
            "theta": xsol[ndx["theta"]: ndx["theta"] + ndim["theta"]],
            "obj": xsol[ndx["J"]],
            "beta": xsol[ndx["beta"]: ndx["beta"] + ndim["beta"]],
            "J0": xsol[ndx["J0"]],
            "f0": xsol[ndx["f0"]],
        }
        self.built = True
        return R


class CKdro_Rand_feat_ecos(AbstractKdro):
    """
    Using Gaussian kernel with the random feature approach to approximate functions in RKHS
    """

    def __init__(
        self,
        dim_theta,
        Ltheta,
        LY,
        C,
        Xobs,
        Yobs,
        Ycert,
        lam,
        sigma,
        kappa=1e-3,
        D=2000,
    ):
        """
        loss function: (Ltheta*theta + LY*Y+C)^2
        """
        assert dim_theta > 0

        self.dim_theta = dim_theta
        self.Ltheta = Ltheta
        self.LY = LY
        self.C = C
        from utils import Gaussian_kern

        self.Kern = lambda x, y: Gaussian_kern(x, y, sigma)
        sigma2 = 1 / sigma
        self.KX = self.Kern(Xobs, Xobs)
        self.D = D
        n = Yobs.shape[1]
        self.omega = np.random.normal(0, sigma2 ** 2, [D, n])
        self.b = np.random.random(D) * 2 * np.pi
        self.phi_obs = np.zeros([Yobs.shape[0], 2 * D])
        self.phi_cert = np.zeros([Ycert.shape[0], 2 * D])
        for i in range(0, Yobs.shape[0]):
            self.phi_obs[i] = np.sqrt(1 / D) * np.append(
                np.cos(self.omega @ Yobs[i] + self.b),
                np.sin(self.omega @ Yobs[i] + self.b),
            )
        for i in range(0, Ycert.shape[0]):
            self.phi_cert[i] = np.sqrt(1 / D) * np.append(
                np.cos(self.omega @ Ycert[i] + self.b),
                np.sin(self.omega @ Ycert[i] + self.b),
            )
        self.phi = np.vstack((self.phi_obs, self.phi_cert))

        self.Xobs = Xobs
        self.Yobs = Yobs
        self.Ycert = Ycert
        self.Y = np.vstack((Yobs, Ycert))
        self.lam = lam
        self.kappa = kappa
        self.built = False
        self.ndim = None
        self.ndx = None
        self.totaldim = None
        self.Gq0 = None
        self.hq0 = None
        self.dims = None

    def robust_opt(self, x, verbose=False):
        """
        eps: epsilon to control the size of the norm ball constraint.

        Return a dictionary of optimization results.
        """
        if len(x.shape) == 1:
            x = np.reshape(x, [1, -1])
        n_sample = self.Xobs.shape[0]
        kxx = np.ones([1, 1]) * self.Kern(x, x)
        kx = self.Kern(self.Xobs, x)
        K = np.vstack((np.hstack((self.KX, kx)), np.hstack((kx.T, kxx))))
        betax = np.linalg.solve(
            K + self.lam * np.eye(n_sample + 1), np.vstack((kx, kxx))
        )

        eps = betax[-1] + self.kappa * n_sample ** -0.25

        # sample size for the set of certification points
        n_certify = self.Ycert.shape[0]
        KY = self.phi @ self.phi.T

        if self.built:
            ndim = self.ndim
            ndx = self.ndx
            totaldim = self.totaldim
            Gq = self.Gq0
            dims = self.dims
            hq = self.hq0
        else:
            # All variables to be optimized
            ndim = {
                "theta": self.dim_theta,
                "f0": 1,
                "beta": n_sample + n_certify,
                "J": 1,
            }
            totaldim = 0
            ndx = {}
            for name in ndim:
                ndx[name] = totaldim
                totaldim += ndim[name]
            self.ndx = ndx
            self.ndim = ndim
            self.totaldim = totaldim
            # certify the observations, encoded as SOCP constraints.
            dims = {"q": []}
            Gq = np.empty([0, totaldim])
            hq = np.empty(0)
            onevec = np.array([1])
            for i in range(0, self.Y.shape[0]):
                G1 = np.zeros(totaldim)
                G2 = np.zeros([self.Ltheta.shape[0], totaldim])
                G1[ndx["beta"]: ndx["beta"] + ndim["beta"]] = -KY[i]
                G1[ndx["f0"]] = -1
                G2[:, ndx["theta"]: ndx["theta"] +
                    ndim["theta"]] = -2 * self.Ltheta
                G3 = -G1.copy()
                h2 = 2 * self.LY @ self.Y[i] + 2 * self.C
                Gtemp = np.vstack((G1, G2, G3))
                htemp = np.concatenate((onevec, h2, onevec))
                Gq = np.vstack((Gq, Gtemp))
                hq = np.append(hq, htemp)
                dims["q"].append(Gtemp.shape[0])
            dims["q"].append(1 + self.D * 2)
            dims["l"] = 1
            self.dims = dims
            self.Gq0 = Gq
            self.hq0 = hq

        # encode the cost

        G1 = np.zeros(totaldim)
        G2 = np.zeros([2 * self.D, totaldim])
        G1[ndx["J"]] = -1
        G1[ndx["beta"]: ndx["beta"] + ndim["beta"]] = (
            betax[0:n_sample].T @ KY[0:n_sample]
        )
        G1[ndx["f0"]] = 1
        G2[:, ndx["beta"]: ndx["beta"] + ndim["beta"]] = -self.phi.T * eps
        Gtemp = np.vstack((G1, G2))
        htemp = np.zeros(Gtemp.shape[0])
        Gq = np.vstack((Gq, Gtemp))
        hq = np.append(hq, htemp)
        Gl = G1.copy()
        hl = np.zeros(1)
        G = np.vstack((Gl, Gq))
        h = np.append(hl, hq)
        c = np.zeros(totaldim)
        c[ndx["J"]] = 1
        startTimer = datetime.datetime.now()
        sol = ecos.solve(c, sparse.csc_matrix(G), h, dims, verbose=verbose)
        endTimer = datetime.datetime.now()
        deltaTimer = endTimer - startTimer
        print("solve time: ", deltaTimer.total_seconds(), " seconds.")
        x = sol["x"]

        R = {
            "theta": x[ndx["theta"]: ndx["theta"] + ndim["theta"]],
            "obj": x[ndx["J"]],
            "beta": x[ndx["beta"]: ndx["beta"] + ndim["beta"]],
            "f0": x[ndx["f0"]],
        }

        self.built = True
        return R
