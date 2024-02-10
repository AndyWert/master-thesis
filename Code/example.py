import numpy as np
import torch
from torch.utils.data import Dataset
from pymor.basic import *
import matplotlib.pyplot as plt
from pymor.algorithms.timestepping import TimeStepper


class CrankNicolsonTimeStepper(TimeStepper):
    """Crank-Nicolson time-stepper.

    Solves equations of the form ::

        M * d_t u + A(u, mu, t) = F(mu, t).

    Parameters
    ----------
    nt
        The number of time-steps the time-stepper will perform.
    solver_options
        The |solver_options| used to invert `M + dt*A`.
        The special values `'mass'` and `'operator'` are
        recognized, in which case the solver_options of
        M (resp. A) are used.
    """

    def __init__(self, nt, solver_options='operator'):
        self.__auto_init(locals())

    def estimate_time_step_count(self, initial_time, end_time):
        return self.nt

    def iterate(self, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None, num_values=None):

        from pymor.operators.interface import Operator
        from pymor.parameters.base import Mu
        from pymor.vectorarrays.interface import VectorArray
        from pymor.parameters.functionals import ConstantParameterFunctional
        from pymor.algorithms.timestepping import _depends_on_time

        A, F, M, U0, t0, t1, nt = operator, rhs, mass, initial_data, initial_time, end_time, self.nt

        assert isinstance(A, Operator)
        assert isinstance(F, (type(None), Operator, VectorArray))
        assert isinstance(M, (type(None), Operator))
        assert A.source == A.range
        num_values = num_values or nt + 1
        dt = (t1 - t0) / nt
        DT = (t1 - t0) / (num_values - 1)

        if F is None:
            F_time_dep = False
        elif isinstance(F, Operator):
            assert F.source.dim == 1
            assert F.range == A.range
            F_time_dep = _depends_on_time(F, mu)
            if not F_time_dep:
                dt_F0 = F.as_vector(mu) * dt * 0.5
                dt_F1 = F.as_vector(mu) * dt * 0.5
        else:
            assert len(F) == 1
            assert F in A.range
            F_time_dep = False
            dt_F0 = F * dt * 0.5
            dt_F1 = F * dt * 0.5

        if M is None:
            from pymor.operators.constructions import IdentityOperator
            M = IdentityOperator(A.source)

        assert A.source == M.source == M.range
        assert not M.parametric
        assert U0 in A.source
        assert len(U0) == 1

        num_ret_values = 1
        yield U0, t0

        options = (A.solver_options if self.solver_options == 'operator' else
                   M.solver_options if self.solver_options == 'mass' else
                   self.olver_options)
        M_dt_A_p = (M + A * dt * ConstantParameterFunctional(1/2)).with_(solver_options=options)
        if not _depends_on_time(M_dt_A_p, mu):
            M_dt_A_p = M_dt_A_p.assemble(mu)
        M_dt_A_m = (M - A * dt * ConstantParameterFunctional(1/2)).with_(solver_options=options)
        if not _depends_on_time(M_dt_A_m, mu):
            M_dt_A_m = M_dt_A_m.assemble(mu)

        t = t0
        U = U0.copy()
        if mu is None:
            mu = Mu()

        for n in range(nt):
            mu0 = mu.with_(t=t)
            t += dt
            mu1 = mu.with_(t=t)
            rhs = M_dt_A_m.apply(U, mu=mu0)
            if F_time_dep:
                dt_F0 = F.as_vector(mu0) * dt * 0.5
                dt_F1 = F.as_vector(mu1) * dt * 0.5

            if F:
                rhs += dt_F0 + dt_F1
            U = M_dt_A_p.apply_inverse(rhs, mu=mu1, initial_guess=U)
            while t - t0 + (min(dt, DT) * 0.5) >= num_ret_values * DT:
                num_ret_values += 1
                yield U, t


def parabolic_equation(q, T, grid_intervals=50, nt=50):
    """Return a parabolic equation.

    Parameters
    ----------
    q
        Control function q
    T
        Stopping time
    grid_intervals
        Number of intervals in each direction of the two-dimensional |RectDomain|.
    nt
        Number of time steps.

    Returns
    -------
    fom
        Parabolic equation problem as an |InstationaryModel|.
    """
    from pymor.analyticalproblems.domaindescriptions import RectDomain
    from pymor.analyticalproblems.elliptic import StationaryProblem
    from pymor.analyticalproblems.functions import ConstantFunction, ExpressionFunction
    from pymor.analyticalproblems.instationary import InstationaryProblem
    from pymor.discretizers.builtin import discretize_instationary_cg

    # setup analytical problem
    domain = RectDomain()

    diffusion = ConstantFunction(1, 2)

    f = ExpressionFunction('-pi**4*exp(a[0]*pi**2*T)*sin(pi*x[0])*sin(pi*x[1])', dim_domain=2, parameters={'a': 1}, values={'T': T})

    rhs = f+q

    problem = InstationaryProblem(

        StationaryProblem(domain, rhs, diffusion),

        T=T,

        initial_data=ExpressionFunction('-1/(2+a[0])*pi**2*sin(pi*x[0])*sin(pi*x[1])', dim_domain=2, parameters={'a': 1})
    )

    # discretize using continuous finite elements
    fom, data = discretize_instationary_cg(analytical_problem=problem, diameter=1./grid_intervals, time_stepper=CrankNicolsonTimeStepper(nt=nt))

    return fom, data


# objective function
def J(param, a, T, grid_intervals=50, nt=50):
    alpha = np.pi**(-4)
    # setting the control function q
    q_base = ExpressionFunction('sin(pi*x[0])*sin(pi*x[1])', dim_domain=2)
    paramFunc = ExpressionFunction('param*(t[0]<=0)*(0<=t[0])', dim_domain=2, parameters={'t': 1}, values={'param': param[0]})
    for i in range(nt):
        paramFunc = paramFunc+ExpressionFunction('((j+1-nt/T*t[0])*param1+(nt/T*t[0]-j)*param2)*(nt/T*t[0]<=j+1)*(j<nt/T*t[0])', dim_domain=2, parameters={'t': 1}, values={'j': i, 'param1': param[i], 'param2': param[i+1], 'nt': nt, 'T': T})
    q = paramFunc*q_base
    # solving the parabolic equation
    fom, data = parabolic_equation(q, T, grid_intervals, nt)
    u = fom.solve({'a': a})
    # calculating the objective function value
    uHat = ExpressionFunction('(a[0]**2-5)/(2+a[0])*pi**2*exp(a[0]*pi**2*t[0])*sin(pi*x[0])*sin(pi*x[1])+2*pi**2*exp(a[0]*pi**2*T)*sin(pi*x[0])*sin(pi*x[1])', dim_domain=2, parameters={'a': 1, 't': 1}, values={'T': T})
    uMinusUHath = u.space.empty(reserve=nt+1)
    qh = u.space.empty(reserve=nt+1)
    grid = data['grid']
    boundary_info = data['boundary_info']
    for i in range(nt+1):
        qh.append(u.space.from_numpy(q.evaluate(grid.centers(grid.dim), q.parameters.parse({'t': i/nt*T}))))
        uMinusUHath.append(u[i][0]-u.space.from_numpy(uHat.evaluate(grid.centers(grid.dim), uHat.parameters.parse({'a': a, 't': i/nt*T}))))
    y1Int = 0
    y2Int = 0
    for t in range(nt):
        y1Int += T/(3*nt) * (uMinusUHath[t].norm2(product=fom.l2_product) + uMinusUHath[t].inner(uMinusUHath[t+1], product=fom.l2_product) + uMinusUHath[t+1].norm2(product=fom.l2_product))
        y2Int += T/(3*nt) * (qh[t].norm2(product=fom.l2_product) + qh[t].inner(qh[t+1], product=fom.l2_product) + qh[t+1].norm2(product=fom.l2_product))
    y1Int = y1Int[0][0]
    y2Int = y2Int[0][0]
    out = 1/2*y1Int+alpha/2*y2Int
    return out, fom, data, 1/2*y1Int, alpha/2*y2Int


def L_BFGS_B_minimizer(init, a, T, grid_intervals=50, nt=50):
    from scipy.optimize import minimize
    assert len(init) == nt+1
    return minimize(lambda mu: J(mu, a, T, grid_intervals, nt)[0], init, method='L-BFGS-B').x


# L2- and LInf-error between a function u and the analytical minimizer uBar
def error(err, u, T, product, nt=50):
    absInfErr = np.max(err.sup_norm())
    relInfErr = np.max(err.sup_norm())/np.max(u.sup_norm())
    t = np.linspace(0, T, num=nt+1)
    yInt = 0
    uInt = 0
    for t in range(nt):
        yInt += T/(3*nt) * (err[t].norm2(product=product) + err[t].inner(err[t+1], product=product) + err[t+1].norm2(product=product))
        uInt += T/(3*nt) * (u[t].norm2(product=product) + u[t].inner(u[t+1], product=product) + u[t+1].norm2(product=product))
    yInt = yInt[0][0]
    uInt = uInt[0][0]
    absL2Err = np.sqrt(yInt)
    relL2Err = np.sqrt(yInt)/np.sqrt(uInt)
    return absL2Err, relL2Err, absInfErr, relInfErr


def lineSearch(F, u_k, d_k, beta, r, eps, nu_1):
    beta_k = beta
    u_k_new = u_k+beta_k*d_k
    nu = 0
    while (F(u_k_new)-F(u_k) <= eps and nu < nu_1):
        beta_k = r*beta_k
        u_k_new = u_k+beta_k*d_k
        nu = nu+1
    return u_k_new


def optStep(F, u_k, N, k, T_k, C_k, F_k, beta_1, beta_2, r, eps, nu_1, var, correlationCoeff):
    N_u = len(u_k)
    C_k_new = np.zeros((N_u, N_u))
    if k == 0:
        for i in range(N_u):
            for j in range(N_u):
                # change if more basis functions
                C_k_new[i][j] = var**2*correlationCoeff**(np.abs(i-j))*1/(1-correlationCoeff**2)
    else:
        d_k_cov = np.zeros((N_u, N_u))
        assert len(T_k) == N
        for m in range(N):
            d_k_cov = d_k_cov+(T_k[m][1]-F_k)*((T_k[m][0]-u_k).reshape((N_u, 1))*(T_k[m][0]-u_k).reshape((1, N_u))-C_k)
        d_k_cov = d_k_cov / N
        beta_3 = np.abs(np.max(C_k)) * beta_2
        C_k_new = C_k + beta_3*d_k_cov / np.abs(np.max(d_k_cov))
    sample = np.random.multivariate_normal(u_k, C_k_new, size=N)
    T_k_new = []
    for i in range(N):
        T_k_new.append([sample[i], F(sample[i])])
    C_F = np.zeros(N_u)
    for m in range(N):
        C_F = C_F+(T_k_new[m][0]-u_k)*(T_k_new[m][1]-F_k)
    C_F = 1/(N-1)*C_F
    d_k = C_F/np.abs(np.max(C_F))
    u_k_new = lineSearch(F, u_k, d_k, beta_1, r, eps, nu_1)
    return u_k_new, T_k_new, C_k_new, F(u_k_new)


def enOpt(F, u_0, N, eps, k_1, beta_1, beta_2, r, nu_1, var, correlationCoeff):
    F_k_prev = F(u_0)
    u_k, T_k, C_k, F_k = optStep(F, u_0, N, 0, [], 0, F_k_prev, beta_1, beta_2, r, eps, nu_1, var, correlationCoeff)
    k = 1
    while (F_k > F_k_prev+eps and k < k_1):
        F_k_prev = F_k
        u_k, T_k, C_k, F_k = optStep(F, u_k, N, k, T_k, C_k, F_k, beta_1, beta_2, r, eps, nu_1, var, correlationCoeff)
        k = k+1
    return u_k, k


def FOM_EnOpt(u_0, N, eps, k_1, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals=50, nt=50):
    return enOpt(lambda mu: -J(mu, a, T, grid_intervals, nt)[0], u_0, N, eps, k_1, beta_1, beta_2, r, nu_1, var, correlationCoeff)


class CustomDataset(Dataset):
    def __init__(self, sample, transform=None, target_transform=None):
        self.sample = sample
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        features = torch.from_numpy(self.sample[idx][0]).to(torch.float32)
        label = torch.tensor(self.sample[idx][1]).to(torch.float32)
        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)
        return features, label


def train_loop(dataloader, model, loss_fn, optimizer):
    # size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        def closure():
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            return loss
        optimizer.step(closure)
        """
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        """


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train(sample, V_DNN):
    from pymor.models.neural_network import FullyConnectedNN
    from torch import nn
    from torch.utils.data import DataLoader
    epochs = V_DNN[3]
    training_batch_size = V_DNN[4]
    testing_batch_size = V_DNN[5]
    learning_rate = V_DNN[6]
    training_data = CustomDataset(sample[V_DNN[2]:])
    test_data = CustomDataset(sample[:V_DNN[2]])
    train_dataloader = DataLoader(training_data, batch_size=training_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=testing_batch_size, shuffle=True)
    DNN = FullyConnectedNN(V_DNN[0], activation_function=V_DNN[1])
    """
    # initialization
    for name, param in DNN.named_parameters():
        if 'bias' in name:
            param = torch.zeros(param.size())
        else:
            param = torch.from_numpy(np.random.multivariate_normal(u_k, C_k_new)
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.LBFGS(DNN.parameters(), lr=learning_rate, line_search_fn='strong_wolfe')

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, DNN, loss_fn, optimizer)
        test_loop(test_dataloader, DNN, loss_fn)
        print("Done!")
    # DNN(torch.from_numpy(sample[m][0]).to(torch.float32)) approx sample[m][1] for all m
    return DNN # todo


def AML_EnOpt(F, u_0, N, eps_o, eps_i, k_1_o, k_1_i, V_DNN, beta_1, beta_2, r, nu_1, var, correlationCoeff):
    # V_DNN: neurons per hidden layer, activation function (like torch.tanh), size of test set, number of epochs, training batch size, testing batch size, learning rate
    V_DNN[0].insert(0, len(u_0))
    V_DNN[0].insert(len(V_DNN[0]), 1)
    F_k = F(u_0)
    u_k_tilde, T_k, C_k, F_k_tilde = optStep(F, u_0, N, 0, [], 0, F_k, beta_1, beta_2, r, eps_o, nu_1, var, correlationCoeff)
    k = 0
    u_k = u_0
    u_k_next = u_k.copy()
    while (F_k_tilde > F_k+eps_o and k < k_1_o):
        F_ML_k = train(T_k, V_DNN)
        u_k_next = enOpt(lambda mu: F_ML_k(torch.from_numpy(mu).to(torch.float32)).detach().numpy()[0], u_k, N, eps_i, k_1_i, beta_1/5, beta_2, r, nu_1, var/5, correlationCoeff)[0]
        F_k_next = F(u_k_next)
        print(u_k_next)
        print(F_k_next)
        print(F_k)
        print(F_ML_k(torch.from_numpy(u_k).to(torch.float32)).detach().numpy()[0])
        print(F_ML_k(torch.from_numpy(u_k_next).to(torch.float32)).detach().numpy()[0])
        # print(T_k)
        if F_k_next <= F_k+eps_o:
            print('fail')
            return u_k, k
        u_k_tilde, T_k, C_k, F_k_tilde = optStep(F, u_k_next, N, k, T_k, C_k, F_k, beta_1, beta_2, r, eps_o, nu_1, var, correlationCoeff)  # different C_k?
        F_k = F_k_next
        u_k = u_k_next
        k = k+1
    return u_k, k


def ROM_EnOpt(u_0, N, eps_o, eps_i, k_1_o, k_1_i, V_DNN, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals=50, nt=50):
    return AML_EnOpt(lambda mu: -J(mu, a, T, grid_intervals, nt)[0], u_0, N, eps_o, eps_i, k_1_o, k_1_i, V_DNN, beta_1, beta_2, r, nu_1, var, correlationCoeff)


def result(name, qParamOpt, qParam, out, fom, data, u, y1, y2, a, T, nt):
    assert len(u) == nt+1
    assert len(qParamOpt) == nt+1
    assert len(qParam) == nt+1
    # error of u
    uBar = ExpressionFunction('-1/(2+a[0])*pi**2*exp(a[0]*pi**2*t[0])*sin(pi*x[0])*sin(pi*x[1])', dim_domain=2, parameters={'a': 1, 't': 1})
    uBarh = u.space.empty(reserve=nt+1)
    for i in range(nt+1):
        uBarh.append(u.space.from_numpy(uBar.evaluate(data['grid'].centers(data['grid'].dim), uBar.parameters.parse({'a': a, 't': i/nt*T}))))
    err = uBarh-u
    absL2Err, relL2Err, absInfErr, relInfErr = error(err, u, T, fom.l2_product, nt)
    fom.visualize(u, title=name+' u')
    fom.visualize(err, title=name+' u error')
    print(name + ' y1 = {}'.format(y1))
    print(name + ' y2 = {}'.format(y2))
    print(name + ' u L2-error = {}'.format(absL2Err))
    print(name + ' u relative L2-error = {}'.format(relL2Err))
    print(name + ' u LInf-error = {}'.format(absInfErr))
    print(name + ' u relative LInf-error = {}'.format(relInfErr))
    # error of q
    print(name + ' q = {}'.format(qParamOpt))
    t = np.linspace(0, T, num=nt+1)
    qAnalytical = np.all(qParamOpt == qParam)
    fig, ax = plt.subplots(2-qAnalytical, 1)
    if qAnalytical:
        ax.plot(t, qParamOpt, label=name+' q')
        ax.legend()
    else:
        qErr = qParamOpt-qParam
        print(name + ' q LInf-error = {}'.format(np.max(np.abs(qErr))))
        print(name + ' q relative LInf-error = {}'.format(np.max(np.abs(qErr))/np.max((qParamOpt))))
        ax[0].plot(t, qParamOpt, label=name+' q')
        ax[0].plot(t, qParam, label='Analytical q')
        ax[0].legend()
        ax[1].plot(t, qErr, label=name+' q error')
        ax[1].legend()
    plt.show()
    print(name + ' J = {}'.format(out))


T = 0.1
nt = 10
grid_intervals = 50
a = -np.sqrt(5)
init = np.zeros(nt+1)-40


# optimized control function using the EnOpt minimizer
N = 300
eps = 1e-8
k_1 = 1000
beta_1 = 100
beta_2 = 0.1
r = 0.5
nu_1 = 20
var = 50
correlationCoeff = 0.1
"""
qParamOpt, k = FOM_EnOpt(init, N, eps, k_1, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt)
outOpt, fomOpt, dataOpt, y1Opt, y2Opt = J(qParamOpt, a, T, grid_intervals, nt)
uOpt = fomOpt.solve({'a': a})
"""
# optimized control function using the AML EnOpt minimizer
eps_o = 1e-4
eps_i = 1e-5
k_1_o = k_1
k_1_i = k_1
# V_DNN: neurons per hidden layer, activation function (like torch.tanh), size of test set, number of epochs, training batch size, testing batch size, learning rate
V_DNN = [[8, 5], torch.tanh, 30, 500, 100, 10, 1e-4]

qParamAMLOpt, kAML = ROM_EnOpt(init, N, eps_o, eps_i, k_1_o, k_1_i, V_DNN, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt)
print(qParamAMLOpt, kAML)
outAMLOpt, fomAMLOpt, dataAMLOpt, y1AMLOpt, y2AMLOpt = J(qParamAMLOpt, a, T, grid_intervals, nt)
uAMLOpt = fomAMLOpt.solve({'a': a})
"""
# optimized control function using the L_BFGS_B_minimizer
qParamOptBFGS = L_BFGS_B_minimizer(init, a, T, grid_intervals, nt)
outOptBFGS, fomOptBFGS, dataOptBFGS, y1OptBFGS, y2OptBFGS = J(qParamOptBFGS, a, T, grid_intervals, nt)
uOptBFGS = fomOptBFGS.solve({'a': a})
"""
# analytical minimizer
qParam = []
for i in range(nt+1):
    qParam.append(-np.pi**4*(np.exp(a*np.pi**2*i/nt*T)-np.exp(a*np.pi**2*T)))
out, fom, data, y1, y2 = J(qParam, a, T, grid_intervals, nt)
u = fom.solve({'a': a})


def analytical():
    result('Analytical', qParam, qParam, out, fom, data, u, y1, y2, a, T, nt)
"""

def opt1():
    result('EnOpt', qParamOpt, qParam, outOpt, fomOpt, dataOpt, uOpt, y1Opt, y2Opt, a, T, nt)
"""

def opt2():
    result('AML_EnOpt', qParamAMLOpt, qParam, outAMLOpt, fomAMLOpt, dataAMLOpt, uAMLOpt, y1AMLOpt, y2AMLOpt, a, T, nt)
"""

def opt3():
    result('L_BFGS_B', qParamOptBFGS, qParam, outOptBFGS, fomOptBFGS, dataOptBFGS, uOptBFGS, y1OptBFGS, y2OptBFGS, a, T, nt)
"""