import numpy as np
import torch
from torch.utils.data import Dataset
from pymor.basic import *
import matplotlib.pyplot as plt
from pymor.algorithms.timestepping import TimeStepper
from numpy import linalg as LA


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


def lineSearch(F, u_k, d_k, beta, r, eps, nu_1, proj):
    beta_k = beta
    u_k_new = proj(u_k+beta_k*d_k)
    nu = 0
    while (F(u_k_new)-F(u_k) <= eps and nu < nu_1):
        beta_k = r*beta_k
        u_k_new = proj(u_k+beta_k*d_k)
        nu = nu+1
    return u_k_new


def initCov(u_k, var, correlationCoeff):
    N_u = len(u_k)
    C_k_new = np.zeros((N_u, N_u))
    for i in range(N_u):
        for j in range(N_u):
            # change if more basis functions
            C_k_new[i, j] = var**2*correlationCoeff**(np.abs(i-j))*1/(1-correlationCoeff**2)
    return C_k_new


def updateCov(u_k, N, T_k, C_k, F_k, beta):
    N_u = len(u_k)
    d_k_cov = np.zeros((N_u, N_u))
    assert len(T_k) == N
    for m in range(N):
        d_k_cov = d_k_cov+(T_k[m][1]-F_k)*((T_k[m][0]-u_k).reshape((N_u, 1))*(T_k[m][0]-u_k).reshape((1, N_u))-C_k)
    d_k_cov = d_k_cov / N
    # beta_3 = np.max(np.abs(C_k)) * beta
    # return C_k + beta_3*d_k_cov / np.max(np.abs(d_k_cov))
    C_k_diag = np.zeros(N_u)
    d_k_cov_diag = np.zeros(N_u)
    for i in range(N_u):
        C_k_diag[i] = C_k[i, i]
        d_k_cov_diag[i] = d_k_cov[i, i]
    beta_iter = beta
    while (np.min(C_k_diag+beta_iter*d_k_cov_diag) < 0):
        beta_iter /= 2
    return C_k + beta_iter*d_k_cov


def optStep12(F, u_k, N, k, T_k, C_k, F_k, beta_1, beta_2, r, eps, nu_1, var, correlationCoeff, proj=lambda mu: mu):
    N_u = len(u_k)
    C_k_new = np.zeros((N_u, N_u))
    if k == 0:
        C_k_new = initCov(u_k, var, correlationCoeff)
    else:
        C_k_new = updateCov(u_k, N, T_k, C_k, F_k, beta_2)
    sample = np.random.multivariate_normal(u_k, C_k_new, size=N)
    T_k_new = []
    for i in range(N):
        T_k_new.append([proj(sample[i]), F(proj(sample[i]))])
    C_F = np.zeros(N_u)
    for m in range(N):
        C_F = C_F+(T_k_new[m][0]-u_k)*(T_k_new[m][1]-F_k)
    C_F = 1/(N-1)*C_F
    """
    print('\n')
    print('u_k: {}'.format(u_k))
    print('T_k_new: {}'.format(T_k_new[:5]))
    print('F_k: {}'.format(F_k))
    print('C_F: {}'.format(C_F))
    print('np.max(np.abs(C_F)): {}'.format(np.max(np.abs(C_F))))
    """
    d_k = C_F/np.max(np.abs(C_F))
    t = np.linspace(0, 0.1, N_u)
    """
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, d_k, label='EnOpt d_k: {}'.format(k))
    ax.legend()
    plt.show()
    """
    u_k_new = lineSearch(F, u_k, d_k, beta_1, r, eps, nu_1, proj)
    """
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, u_k_new, label='EnOpt u_k: {}'.format(k))
    ax.legend()
    plt.show()
    """
    return u_k_new, T_k_new, C_k_new, F(u_k_new)


def enOpt12(F, u_0, N, eps, k_1, beta_1, beta_2, r, nu_1, var, correlationCoeff, proj=lambda mu: mu):
    F_k_prev = F(u_0)
    u_k, T_k, C_k, F_k = optStep(F, u_0, N, 0, [], 0, F_k_prev, beta_1, beta_2, r, eps, nu_1, var, correlationCoeff, proj)
    k = 1
    while (F_k > F_k_prev+eps and k < k_1):
        F_k_prev = F_k
        u_k, T_k, C_k, F_k = optStep(F, u_k, N, k, T_k, C_k, F_k, beta_1, beta_2, r, eps, nu_1, var, correlationCoeff, proj)
        k = k+1
    return u_k, k


def optStep(F, u_k, N, k, T_k, C_k, F_k, beta_1, beta_2, r, eps, nu_1, var, correlationCoeff, proj=lambda mu: mu):
    N_u = len(u_k)
    C_k_new = np.zeros((N_u, N_u))
    if k == 0:
        if C_k is None:
            C_k_new = initCov(u_k, var, correlationCoeff)
        else:
            C_k_new = C_k.copy()
    else:
        C_k_new = updateCov(u_k, N, T_k, C_k, F_k, beta_2)
    sample = np.random.multivariate_normal(u_k, C_k_new, size=N)
    T_k_new = []
    for i in range(N):
        T_k_new.append([proj(sample[i]), F(proj(sample[i]))])
        # T_k_new.append([sample[i], F(sample[i])])
    C_F = np.zeros(N_u)
    for m in range(N):
        C_F = C_F+(T_k_new[m][0]-u_k)*(T_k_new[m][1]-F_k)
    C_F = 1/(N-1)*C_F
    """
    print('\n')
    print('u_k: {}'.format(u_k))
    print('T_k_new: {}'.format(T_k_new[:5]))
    print('F_k: {}'.format(F_k))
    print('C_F: {}'.format(C_F))
    print('np.max(np.abs(C_F)): {}'.format(np.max(np.abs(C_F))))
    """
    d_k = C_F/np.max(np.abs(C_F))
    t = np.linspace(0, 0.1, N_u)
    """
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, d_k, label='EnOpt d_k: {}'.format(k))
    ax.legend()
    plt.show()
    """
    u_k_new = lineSearch(F, u_k, d_k, beta_1, r, eps, nu_1, proj)
    """
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, u_k_new, label='EnOpt u_k: {}'.format(k))
    ax.legend()
    plt.show()
    """
    return u_k_new, T_k_new, C_k_new, F(u_k_new)


def enOpt(F, u_0, N, eps, k_1, beta_1, beta_2, r, nu_1, var, correlationCoeff, proj=lambda mu: mu, Cov=None):
    F_k_prev = F(u_0)
    u_k, T_k, C_k, F_k = optStep(F, u_0, N, 0, [], Cov, F_k_prev, beta_1, beta_2, r, eps, nu_1, var, correlationCoeff, proj)
    k = 1
    while (F_k > F_k_prev+eps and k < k_1):
        F_k_prev = F_k
        u_k, T_k, C_k, F_k = optStep(F, u_k, N, k, T_k, C_k, F_k, beta_1, beta_2, r, eps, nu_1, var, correlationCoeff, proj)
        k = k+1
    return u_k, k


def enOpt1(F, u_0, N, eps, k_1, beta_1, beta_2, r, nu_1, var, correlationCoeff, J, proj=lambda mu: mu, Cov=None):
    F_k_prev = F(u_0)
    u_k, T_k, C_k, F_k = optStep(F, u_0, N, 0, [], Cov, F_k_prev, beta_1, beta_2, r, eps, nu_1, var, correlationCoeff, proj)
    k = 1
    errorList = [F_k_prev-J(u_0), F_k-J(u_k)]
    while (F_k > F_k_prev+eps and k < k_1):
        F_k_prev = F_k
        # u_k_FOM, T_k_FOM, C_k_FOM, F_k_FOM = optStep(J, u_k, N, k, T_k, C_k, J(u_k), beta_1, beta_2, r, eps, nu_1, var, correlationCoeff, proj)
        u_k, T_k, C_k, F_k = optStep(F, u_k, N, k, T_k, C_k, F_k, beta_1, beta_2, r, eps, nu_1, var, correlationCoeff, proj)
        errorList.append(F_k-J(u_k))
        k = k+1
    tError = range(len(errorList))
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(tError, errorList, label='error')
    # ax.legend()
    # plt.show()
    plt.bar(tError, errorList)
    plt.suptitle('error')
    plt.show()
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
            pred = model(X).reshape(len(y))
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
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X).reshape(len(y))
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    # print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss


def DNN_optimizer(DNN, train_dataloader, test_dataloader, loss_fn, optimizer, epochs):
    wait = 0
    minimal_validation_loss = test_loop(test_dataloader, DNN, loss_fn)
    torch.save(DNN.state_dict(), 'checkpoint.pth')
    for t in range(epochs):
        # print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, DNN, loss_fn, optimizer)
        validation_loss = test_loop(test_dataloader, DNN, loss_fn)
        if validation_loss < minimal_validation_loss:
            wait = 0
            minimal_validation_loss = validation_loss
            torch.save(DNN.state_dict(), 'checkpoint.pth')
        else:
            wait += 1
        if wait >= 20:
            DNN.load_state_dict(torch.load('checkpoint.pth'))
            print(t)
            break
        # print("Done!")


def evaluate_DNN(model, training_data, test_data, loss_fn):
    train_loss = 0
    test_loss = 0
    # with torch.no_grad():
    """
    for X, y in training_data.sample:
        pred = model(torch.tensor(X).to(torch.float32))
        train_loss += loss_fn(torch.tensor(pred), torch.tensor(y)).item()
    for X, y in test_data.sample:
        pred = model(torch.tensor(X).to(torch.float32))
        test_loss += loss_fn(torch.tensor(pred), torch.tensor(y)).item()
    """
    for i in range(training_data.__len__()):
        X, y = training_data.__getitem__(i)
        pred = model(X)
        train_loss += loss_fn(pred, y).item()
    for i in range(test_data.__len__()):
        X, y = test_data.__getitem__(i)
        pred = model(X)
        test_loss += loss_fn(pred, y).item()
    train_loss /= training_data.__len__()
    test_loss /= test_data.__len__()
    loss = train_loss + test_loss
    """
    print('train loss: {}'.format(train_loss))
    print('test loss: {}'.format(test_loss))
    print('loss: {}'.format(loss))
    """
    return loss


def train123(sample, V_DNN):
    from pymor.models.neural_network import FullyConnectedNN
    from torch import nn
    from torch.utils.data import DataLoader
    epochs = V_DNN[3]
    training_batch_size = V_DNN[4]
    testing_batch_size = V_DNN[5]
    learning_rate = V_DNN[6]
    # scaling
    normSample = []
    for i in range(len(sample)):
        normSample.append([sample[i][0].copy(), sample[i][1]])
    #normSample = sample.copy()
    minIn = np.min(normSample[0][0])
    maxIn = np.max(normSample[0][0])
    minOut = normSample[0][1]
    maxOut = normSample[0][1]
    for i in range(1, len(normSample)):
        minInI = np.min(normSample[i][0])
        maxInI = np.max(normSample[i][0])
        minOutI = normSample[i][1]
        maxOutI = normSample[i][1]
        if minInI < minIn:
            minIn = minInI
        if maxInI > maxIn:
            maxIn = maxInI
        if minOutI < minOut:
            minOut = minOutI
        if maxOutI > maxOut:
            maxOut = maxOutI
    assert minIn != maxIn
    assert minOut != maxOut
    for i in range(len(normSample)):
        normSample[i][0] = (normSample[i][0]-minIn)/(maxIn-minIn)
        normSample[i][1] = (normSample[i][1]-minOut)/(maxOut-minOut)
    training_data = CustomDataset(normSample[V_DNN[2]:])
    test_data = CustomDataset(normSample[:V_DNN[2]])
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

    DNN_optimizer(DNN, train_dataloader, test_dataloader, loss_fn, optimizer, epochs)
    DNN_eval = evaluate_DNN(DNN, training_data, test_data, loss_fn)
    for i in range(25):
        DNN_i = FullyConnectedNN(V_DNN[0], activation_function=V_DNN[1])
        optimizer = torch.optim.LBFGS(DNN_i.parameters(), lr=learning_rate, line_search_fn='strong_wolfe')
        DNN_optimizer(DNN_i, train_dataloader, test_dataloader, loss_fn, optimizer, epochs)
        DNN_i_eval = evaluate_DNN(DNN_i, training_data, test_data, loss_fn)
        if DNN_i_eval < DNN_eval:
            DNN_eval = DNN_i_eval
            DNN = DNN_i
    for i in range(len(normSample)):
        print('DNN output: {}'.format(DNN(torch.from_numpy(normSample[i][0]).to(torch.float32)).detach().numpy()))
        print('FOM output: {}'.format(normSample[i][1]))
    sortedDNNOutputs = []
    sortedFOMOutputs = []
    sortSample = normSample.copy()
    for i in range(len(normSample)):
        jMin = 0
        sampleMin = sortSample[0][1]
        for j in range(len(sortSample)):
            if sortSample[j][1] < sampleMin:
                jMin = j
                sampleMin = sortSample[j][1]
        sortedFOMOutputs.append(sampleMin)
        sortedDNNOutputs.append(DNN(torch.from_numpy(sortSample[jMin][0]).to(torch.float32)).detach().numpy()[0])
        sortSample.pop(jMin)
    print(range(len(sortedDNNOutputs)))
    print(sortedDNNOutputs)
    plt.bar(range(len(sortedDNNOutputs)), sortedDNNOutputs)
    plt.suptitle('sortedDNNOutputs')
    plt.show()
    plt.bar(range(len(sortedFOMOutputs)), sortedFOMOutputs)
    plt.suptitle('sortedFOMOutputs')
    plt.show()
    # DNN(torch.from_numpy(sample[m][0]).to(torch.float32)) approx sample[m][1] for all m
    return lambda mu: DNN(torch.from_numpy((mu-minIn)/(maxIn-minIn)).to(torch.float32)).detach().numpy()[0]*(maxOut-minOut)+minOut
    """
    def out(x):
        if np.min(x) >= minIn and np.max(x) <= maxIn:
            return DNN(torch.from_numpy((x-minIn)/(maxIn-minIn)).to(torch.float32)).detach().numpy()[0]
        else:
            return 0.
    def out1(x):
        out = DNN(torch.from_numpy((x-minIn)/(maxIn-minIn)).to(torch.float32)).detach().numpy()[0]
        if out < 0:
            return 0.
        elif out > 1:
            return 1.
        else:
            return out
    # return out
    # return out1
    return lambda mu: DNN(torch.from_numpy((mu-minIn)/(maxIn-minIn)).to(torch.float32)).detach().numpy()[0]
    """


def projectionPreCond(x, x0, delta, M):
    # dist = LA.norm(M.dot(x-x0))
    dist = np.max(np.abs(M.dot(x-x0)))
    if dist > delta:
        return x0+delta*(x-x0)/dist
    else:
        return x


def projection1(x, x0, delta):
    dist = np.max(np.abs(x-x0))
    if dist > delta:
        return x0+delta*(x-x0)/dist
    else:
        return x


def projection(x, x0, delta):
    dist = LA.norm(x-x0)
    if dist > delta:
        return x0+delta*(x-x0)/dist
    else:
        return x


def projectionSample(x, minIn, maxIn):
    xProj = x.copy()
    for i in range(len(xProj)):
        if xProj[i] < minIn[i]:
            xProj[i] = minIn[i]
        if xProj[i] > maxIn[i]:
            xProj[i] = maxIn[i]
    return xProj


def evalMLM(F_ML_k, T_k, k):
    x = range(len(T_k))
    y_FOM = np.zeros(len(T_k))
    y_ML = np.zeros(len(T_k))
    y_diff = np.zeros(len(T_k))
    for i in range(len(T_k)):
        y_FOM[i] = T_k[i][1]
        y_ML[i] = F_ML_k(T_k[i][0])
        y_diff[i] = F_ML_k(T_k[i][0])-T_k[i][1]
    plt.bar(x, y_FOM)
    plt.suptitle('FOM values: {}'.format(k))
    plt.show()
    plt.bar(x, y_ML)
    plt.suptitle('ML values: {}'.format(k))
    plt.show()
    plt.bar(x, y_diff)
    plt.suptitle('ML-error: {}'.format(k))
    plt.show()


def evalMLM_delta(F_ML_k, F, u_k, u_k_next, delta):
    MLM_val = []
    FOM_val = []
    diff = []
    l2_dist = []
    u = u_k.copy()-delta
    tr = F_ML_k(u_k)
    for i in range(21):
        F_ML_k_u = F_ML_k(u)-tr
        F_u = F(u)-tr
        MLM_val.append(F_ML_k_u)
        FOM_val.append(F_u)
        diff.append(F_ML_k_u-F_u)
        l2_dist.append(LA.norm(u-u_k_next))
        u += delta/10
    F_ML_k_u_k_next = F_ML_k(u_k_next)-tr
    F_u_k_next = F(u_k_next)-tr
    MLM_val.append(F_ML_k_u_k_next)
    FOM_val.append(F_u_k_next)
    diff.append(F_ML_k_u_k_next-F_u_k_next)
    plt.bar(range(len(MLM_val)), MLM_val)
    plt.suptitle('translated u_k and u_k_next MLM-values')
    plt.show()
    plt.bar(range(len(FOM_val)), FOM_val)
    plt.suptitle('translated u_k and u_k_next FOM-values')
    plt.show()
    plt.bar(range(len(l2_dist)), l2_dist)
    plt.suptitle('l2 distance to u_k_next')
    plt.show()


def evalMLM_delta1(F_ML_k, F, u_k, u_k_next, delta):
    MLM_val = []
    FOM_val = []
    diff = []
    l2_dist = []
    u = u_k.copy()-delta
    tr = F(u_k)
    for i in range(21):
        F_ML_k_u = F_ML_k(u)-tr
        F_u = F(u)-tr
        MLM_val.append(F_ML_k_u)
        FOM_val.append(F_u)
        diff.append(F_ML_k_u-F_u)
        l2_dist.append(LA.norm(u-u_k_next))
        u += delta/10
    F_ML_k_u_k_next = F_ML_k(u_k_next)-tr
    F_u_k_next = F(u_k_next)-tr
    MLM_val.append(F_ML_k_u_k_next)
    FOM_val.append(F_u_k_next)
    diff.append(F_ML_k_u_k_next-F_u_k_next)
    plt.bar(range(len(MLM_val)), MLM_val)
    plt.suptitle('translated u_k and u_k_next MLM-values')
    plt.show()
    plt.bar(range(len(FOM_val)), FOM_val)
    plt.suptitle('translated u_k and u_k_next FOM-values')
    plt.show()
    plt.bar(range(len(l2_dist)), l2_dist)
    plt.suptitle('l2 distance to u_k_next')
    plt.show()


def test_DNN(DNN, x_test, y_test, loss_fn):
    DNN.eval()
    with torch.inference_mode():
        test_pred = DNN(x_test).reshape(len(y_test))
        test_loss = loss_fn(test_pred, y_test)
        return test_loss


def train_DNN(DNN, x, y, x_train, y_train, x_test, y_test, normSample, normVal, loss_fn, optimizer, epochs):
    wait = 0
    minimal_test_loss = test_DNN(DNN, x_test, y_test, loss_fn)
    test_iteration = [minimal_test_loss]
    torch.save(DNN.state_dict(), 'checkpoint.pth')
    for epoch in range(epochs):
        # Training
        """
        with torch.inference_mode():
            # print(x[0])
            # y_preds = DNN(x).reshape(len(y))
            # print('y_preds: {}'.format(y_preds))
            # print('y: {}'.format(y))
            # print('diff: {}'.format(y_preds-y))

            sortedDNNOutputs = []
            sortedFOMOutputs = []
            sortSample = []
            for i in range(len(normSample)):
                sortSample.append([normSample[i, :], normVal[i]])
            for i in range(len(normSample)):
                jMin = 0
                sampleMin = sortSample[0][1]
                for j in range(len(sortSample)):
                    if sortSample[j][1] < sampleMin:
                        jMin = j
                        sampleMin = sortSample[j][1]
                sortedFOMOutputs.append(sampleMin)
                sortedDNNOutputs.append(DNN(torch.from_numpy(sortSample[jMin][0]).to(torch.float32)).detach().numpy()[0])
                sortSample.pop(jMin)
            # print(range(len(sortedDNNOutputs)))
            # print(sortedDNNOutputs)
            minSortedDNNOutputs = np.min(sortedDNNOutputs)
            plt.bar(range(len(sortedDNNOutputs)), sortedDNNOutputs)
            plt.suptitle('sortedDNNOutputs, epoch {}'.format(epoch))
            plt.show()
            # plt.bar(range(len(sortedDNNOutputs)), sortedDNNOutputs-minSortedDNNOutputs)
            # plt.suptitle('translated sortedDNNOutputs, epoch {}'.format(epoch))
            # plt.show()
            plt.bar(range(len(sortedFOMOutputs)), sortedFOMOutputs)
            plt.suptitle('sortedFOMOutputs, epoch {}'.format(epoch))
            plt.show()
        """
        DNN.train()

        def closure():
            y_pred = DNN(x_train).reshape(len(y_train))
            loss = loss_fn(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            return loss
        optimizer.step(closure)

        # Testing
        test_loss = test_DNN(DNN, x_test, y_test, loss_fn)
        test_iteration.append(test_loss)
        if epoch % 1 == 0:
            print(f"Epoch: {epoch} | Test loss: {test_loss}")
            # print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")
        if test_loss < minimal_test_loss:
            wait = 0
            minimal_test_loss = test_loss
            torch.save(DNN.state_dict(), 'checkpoint.pth')
        else:
            wait += 1
        if wait >= 40:
            DNN.load_state_dict(torch.load('checkpoint.pth'))
            fig, ax = plt.subplots(1, 1)
            ax.plot(range(len(test_iteration)), test_iteration, label='DNN training test loss')
            ax.legend()
            plt.show()
            if len(test_iteration) > 25:
                fig, ax = plt.subplots(1, 1)
                ax.plot(range(25), test_iteration[(len(test_iteration)-25):], label='DNN training test loss, last 25 iterations')
                ax.legend()
                plt.show()
            return  # break
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(len(test_iteration)), test_iteration, label='DNN training test loss')
    ax.legend()
    plt.show()
    if len(test_iteration) > 25:
        fig, ax = plt.subplots(1, 1)
        ax.plot(range(25), test_iteration[(len(test_iteration)-25):], label='DNN training test loss, last 25 iterations')
        ax.legend()
        plt.show()


def train(sample, V_DNN, minIn, maxIn):
    from pymor.models.neural_network import FullyConnectedNN
    from torch import nn
    # from torch.utils.data import DataLoader
    epochs = V_DNN[3]
    training_batch_size = V_DNN[4]
    testing_batch_size = V_DNN[5]
    learning_rate = V_DNN[6]
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # scaling
    normSample = np.zeros((len(sample), len(sample[0][0])))
    normVal = np.zeros(len(sample))
    for i in range(len(sample)):
        normSample[i, :] = sample[i][0]
        normVal[i] = sample[i][1]
    minOut = np.min(normVal)
    maxOut = np.max(normVal)
    print('normSample: {}'.format(normSample))
    print('normVal: {}'.format(normVal))
    normSample -= minIn
    normVal -= minOut
    print('normSample: {}'.format(normSample))
    print('normVal: {}'.format(normVal))
    normSample /= maxIn-minIn
    normVal /= maxOut-minOut
    print('normSample: {}'.format(normSample))
    print('normVal: {}'.format(normVal))
    """
    normSample = []
    for i in range(len(sample)):
        normSample.append([sample[i][0].copy(), sample[i][1]])
    #normSample = sample.copy()
    minIn = np.min(normSample[0][0])
    maxIn = np.max(normSample[0][0])
    minOut = normSample[0][1]
    maxOut = normSample[0][1]
    for i in range(1, len(normSample)):
        minInI = np.min(normSample[i][0])
        maxInI = np.max(normSample[i][0])
        minOutI = normSample[i][1]
        maxOutI = normSample[i][1]
        if minInI < minIn:
            minIn = minInI
        if maxInI > maxIn:
            maxIn = maxInI
        if minOutI < minOut:
            minOut = minOutI
        if maxOutI > maxOut:
            maxOut = maxOutI
    assert minIn != maxIn
    assert minOut != maxOut
    for i in range(len(normSample)):
        normSample[i][0] = (normSample[i][0]-minIn)/(maxIn-minIn)
        normSample[i][1] = (normSample[i][1]-minOut)/(maxOut-minOut)
    x = []
    y = []
    for i in range(len(normSample)):
        x.append(torch.from_numpy(normSample[i][0]).to(torch.float32))
        y.append(torch.tensor(normSample[i][1]).to(torch.float32))
    x_stacked = torch.stack(x, dim=0)
    y_stacked = torch.stack(y, dim=0)
    print(x_stacked[0])
    print(y_stacked[0])
    # Create train/test split
    train_split = int(0.8 * len(x_stacked)) # 80% of data used for training set, 20% for testing 
    x_train, y_train = x_stacked[:train_split], y_stacked[:train_split]
    x_test, y_test = x_stacked[train_split:], y_stacked[train_split:]
    """
    x = torch.from_numpy(normSample).to(torch.float32)
    y = torch.from_numpy(normVal).to(torch.float32)
    print(x)
    print(y)
    # Create train/test split
    train_split = int(0.8 * len(x))
    x_train, y_train = x[:train_split], y[:train_split]
    x_test, y_test = x[train_split:], y[train_split:]
    # print(x_train)
    # print(y_train)
    # print(x_test)
    # print(y_test)
    # print(len(x_train), len(y_train), len(x_test), len(y_test))

    DNN = FullyConnectedNN(V_DNN[0], activation_function=V_DNN[1])

    """
    class DNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_1 = nn.Linear(in_features=len(x[0, :]), out_features=250)
            self.layer_2 = nn.Linear(in_features=250, out_features=250)
            self.layer_3 = nn.Linear(in_features=250, out_features=1)

        def forward(self, x):
            return self.layer_3(torch.tanh(self.layer_2(torch.tanh(self.layer_1(x)))))
    """
    """
    class DNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_layer_stack = nn.Sequential(
                nn.Linear(in_features=len(x[0, :]), out_features=250),
                nn.ReLU(),
                nn.Linear(in_features=250, out_features=250),
                nn.ReLU(),
                nn.Linear(in_features=250, out_features=1),
            )

        def forward(self, x):
            return self.linear_layer_stack(x)
    """

    # DNN = DNNModel().to(device)

    # print(DNN.state_dict())
    # print(list(DNN.parameters()))
    # print(DNN.parameters().dtype)
    """
    with torch.inference_mode():
        print(x[0])
        y_preds = DNN(x).reshape(len(y))
        print('y_preds: {}'.format(y_preds))
        print('y: {}'.format(y))
        print('diff: {}'.format(y_preds-y))
    """
    DNN.to(device)
    loss_fn = nn.MSELoss()
    # optimizer = torch.optim.SGD(params=DNN.parameters(), lr=learning_rate)
    optimizer = torch.optim.LBFGS(DNN.parameters(), lr=learning_rate, line_search_fn='strong_wolfe')

    x_train = x_train.to(device)
    x_test = x_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    train_DNN(DNN, x, y, x_train, y_train, x_test, y_test, normSample, normVal, loss_fn, optimizer, epochs)
    DNN_eval = test_DNN(DNN, x_test, y_test, loss_fn)
    test_loss = [DNN_eval]
    for i in range(10):
        DNN_i = FullyConnectedNN(V_DNN[0], activation_function=V_DNN[1])
        optimizer = torch.optim.LBFGS(DNN_i.parameters(), lr=learning_rate, line_search_fn='strong_wolfe')
        train_DNN(DNN_i, x, y, x_train, y_train, x_test, y_test, normSample, normVal, loss_fn, optimizer, epochs)
        DNN_i_eval = test_DNN(DNN_i, x_test, y_test, loss_fn)
        test_loss.append(DNN_i_eval)
        if DNN_i_eval < DNN_eval:
            DNN_eval = DNN_i_eval
            DNN = DNN_i
    plt.bar(range(len(test_loss)), test_loss)
    plt.suptitle('test_loss')
    plt.show()

    with torch.inference_mode():
        print(x[0])
        y_preds = DNN(x).reshape(len(y))
        print('y_preds: {}'.format(y_preds))
        print('y: {}'.format(y))
        print('diff: {}'.format(y_preds-y))

        sortedDNNOutputs = []
        sortedFOMOutputs = []
        sortSample = []
        for i in range(len(normSample)):
            sortSample.append([normSample[i, :], normVal[i]])
        for i in range(len(normSample)):
            jMin = 0
            sampleMin = sortSample[0][1]
            for j in range(len(sortSample)):
                if sortSample[j][1] < sampleMin:
                    jMin = j
                    sampleMin = sortSample[j][1]
            sortedFOMOutputs.append(sampleMin)
            sortedDNNOutputs.append(DNN(torch.from_numpy(sortSample[jMin][0]).to(torch.float32)).detach().numpy()[0])
            sortSample.pop(jMin)
        print(range(len(sortedDNNOutputs)))
        print(sortedDNNOutputs)
        minSortedDNNOutputs = np.min(sortedDNNOutputs)
        plt.bar(range(len(sortedDNNOutputs)), sortedDNNOutputs)
        plt.suptitle('sortedDNNOutputs')
        plt.show()
        """
        plt.bar(range(len(sortedDNNOutputs)), sortedDNNOutputs-minSortedDNNOutputs)
        plt.suptitle('translated sortedDNNOutputs')
        plt.show()
        """
        plt.bar(range(len(sortedFOMOutputs)), sortedFOMOutputs)
        plt.suptitle('sortedFOMOutputs')
        plt.show()

    def f(inp):
        scaledInput = torch.from_numpy((inp-minIn)/(maxIn-minIn)).to(torch.float32)
        with torch.inference_mode():
            scaledOutput = DNN(scaledInput)
        # return scaledOutput.numpy()
        print('scaledOutput.numpy()[0]: {}'.format(scaledOutput.numpy()[0]))
        return scaledOutput.numpy()[0]*(maxOut-minOut)+minOut
    return f
    """
    training_data = CustomDataset(normSample[V_DNN[2]:])
    test_data = CustomDataset(normSample[:V_DNN[2]])
    train_dataloader = DataLoader(training_data, batch_size=training_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=testing_batch_size, shuffle=True)
    DNN = FullyConnectedNN(V_DNN[0], activation_function=V_DNN[1])
    """
    """
    # initialization
    for name, param in DNN.named_parameters():
        if 'bias' in name:
            param = torch.zeros(param.size())
        else:
            param = torch.from_numpy(np.random.multivariate_normal(u_k, C_k_new)
    """
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.LBFGS(DNN.parameters(), lr=learning_rate, line_search_fn='strong_wolfe')

    DNN_optimizer(DNN, train_dataloader, test_dataloader, loss_fn, optimizer, epochs)
    DNN_eval = evaluate_DNN(DNN, training_data, test_data, loss_fn)
    for i in range(25):
        DNN_i = FullyConnectedNN(V_DNN[0], activation_function=V_DNN[1])
        optimizer = torch.optim.LBFGS(DNN_i.parameters(), lr=learning_rate, line_search_fn='strong_wolfe')
        DNN_optimizer(DNN_i, train_dataloader, test_dataloader, loss_fn, optimizer, epochs)
        DNN_i_eval = evaluate_DNN(DNN_i, training_data, test_data, loss_fn)
        if DNN_i_eval < DNN_eval:
            DNN_eval = DNN_i_eval
            DNN = DNN_i
    # DNN(torch.from_numpy(sample[m][0]).to(torch.float32)) approx sample[m][1] for all m
    return lambda mu: DNN(torch.from_numpy((mu-minIn)/(maxIn-minIn)).to(torch.float32)).detach().numpy()[0]*(maxOut-minOut)+minOut
    """
    """
    def out(x):
        if np.min(x) >= minIn and np.max(x) <= maxIn:
            return DNN(torch.from_numpy((x-minIn)/(maxIn-minIn)).to(torch.float32)).detach().numpy()[0]
        else:
            return 0.
    def out1(x):
        out = DNN(torch.from_numpy((x-minIn)/(maxIn-minIn)).to(torch.float32)).detach().numpy()[0]
        if out < 0:
            return 0.
        elif out > 1:
            return 1.
        else:
            return out
    # return out
    # return out1
    return lambda mu: DNN(torch.from_numpy((mu-minIn)/(maxIn-minIn)).to(torch.float32)).detach().numpy()[0]
    """


def AML_EnOpt(F, u_0, N, eps_o, eps_i, k_1_o, k_1_i, V_DNN, beta_1, beta_2, r, nu_1, var, correlationCoeff):
    # V_DNN: neurons per hidden layer, activation function (like torch.tanh), size of test set, number of epochs, training batch size, testing batch size, learning rate
    V_DNN[0].insert(0, len(u_0))
    V_DNN[0].insert(len(V_DNN[0]), 1)
    var_o = var
    var_i = var_o  # /100
    # var_i_list = [var_i]
    F_k = F(u_0)
    u_k_tilde, T_k, C_k, F_k_tilde = optStep(F, u_0, N, 0, [], None, F_k, beta_1, beta_2, r, eps_o, nu_1, var_o, correlationCoeff)
    t = np.linspace(0, T, num=nt+1)
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, u_k_tilde, label='u_k_tilde: {}'.format(1))
    ax.legend()
    plt.show()
    k = 1
    u_k = u_0
    u_k_next = u_k.copy()
    u_k_tilde_check = []
    fails_iter = []
    fails = 0
    delta = 100
    while (F_k_tilde > F_k+eps_o and k < k_1_o):
        # u_k = u_k_tilde.copy()
        # F_k = F_k_tilde
        T_k_x = np.zeros((N, nt+1))
        for i in range(N):
            T_k_x[i, :] = T_k[i][0]
        minIn = np.zeros(nt+1)
        maxIn = np.zeros(nt+1)
        for i in range(nt+1):
            minIn[i] = np.min(T_k_x[:, i])
            maxIn[i] = np.max(T_k_x[:, i])
        u_k_tilde_check_iter = 0
        if np.all(u_k_tilde == projectionSample(u_k_tilde, minIn, maxIn)):
            u_k_tilde_check_iter = 1
        u_k_tilde_check.append(u_k_tilde_check_iter)
        F_ML_k = train(T_k, V_DNN, minIn, maxIn)
        """
        evalMLM(F_ML_k, T_k, k)
        F_k_next = F_k
        F_ML_k_u_k = F_ML_k(u_k)
        # C_k_inv = LA.inv(C_k)
        maxDiff = 0
        for i in range(len(T_k)):
            diff = np.min(np.abs(u_k-T_k[i][0]))
            if diff > maxDiff:
                maxDiff = diff
        """
        # u_k_next = enOpt1(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff, F, proj=lambda mu: projectionSample(mu, minIn, maxIn))[0]
        # u_k_next = enOpt1(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff, F)[0]
        u_k_next = enOpt(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff, Cov=C_k)[0]
        # u_k_next = enOpt(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff)[0]
        F_k_next = F(u_k_next)
        T_k_min_u = T_k[0][0].copy()
        T_k_min_dist = LA.norm(T_k_min_u-u_k)
        T_k_max_u = T_k[0][0].copy()
        T_k_max_dist = LA.norm(T_k_max_u-u_k)
        for i in range(len(T_k)):
            dist = LA.norm(T_k[i][0]-u_k)
            if dist < T_k_min_dist:
                T_k_min_dist = dist
                T_k_min_u = T_k[i][0].copy()
            if dist > T_k_max_dist:
                T_k_max_dist = dist
                T_k_max_u = T_k[i][0].copy()
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k, label='u_k: {}'.format(k))
        ax.plot(t, u_k_tilde, label='u_k_tilde: {}'.format(k))
        ax.plot(t, u_k_next, label='u_k_next: {}'.format(k))
        ax.plot(t, T_k_min_u, label='T_k_min: {}'.format(k))
        ax.plot(t, T_k_max_u, label='T_k_max: {}'.format(k))
        ax.legend()
        plt.show()
        fig, ax = plt.subplots(1, 1)
        ax.plot([0,1], [LA.norm(u_k_tilde-u_k), LA.norm(u_k_tilde-u_k)], label='u_k_tilde dist: {}'.format(k))
        # ax.plot([0,1], [LA.norm(u_k_next-u_k), LA.norm(u_k_next-u_k)], label='u_k_next dist: {}'.format(k))
        ax.plot([0,1], [T_k_min_dist, T_k_min_dist], label='T_k_min dist: {}'.format(k))
        ax.plot([0,1], [T_k_max_dist, T_k_max_dist], label='T_k_max dist: {}'.format(k))
        ax.legend()
        plt.show()
        fig, ax = plt.subplots(1, 1)
        ax.plot([0,1], [LA.norm(u_k_tilde-u_k), LA.norm(u_k_tilde-u_k)], label='u_k_tilde dist: {}'.format(k))
        ax.plot([0,1], [LA.norm(u_k_next-u_k), LA.norm(u_k_next-u_k)], label='u_k_next dist: {}'.format(k))
        ax.plot([0,1], [T_k_min_dist, T_k_min_dist], label='T_k_min dist: {}'.format(k))
        ax.plot([0,1], [T_k_max_dist, T_k_max_dist], label='T_k_max dist: {}'.format(k))
        ax.legend()
        plt.show()
        # evalMLM_delta(F_ML_k, F, u_k, u_k_next, del
        # var_i = maxDiff/50ta)
        # evalMLM_delta1(F_ML_k, F, u_k, u_k_next, delta)

        if F_k_next <= F_k+eps_o:
            fails += 1
            # evalMLM(F_ML_k, T_k, k)
            F_k_next = F_k
            F_ML_k_u_k = F_ML_k(u_k)
            deltaList = [delta]
            # C_k_inv = LA.inv(C_k)
            """
            maxDiff = 0
            for i in range(len(T_k)):
                diff = np.min(np.abs(u_k-T_k[i][0]))
                if diff > maxDiff:
                    maxDiff = diff
            var_i = maxDiff/50
            """
            trustRegionFlag = True
            while trustRegionFlag:
                u_k_tilde_check2 = 0
                if np.all(u_k_tilde == projection(u_k_tilde, u_k, delta)):
                    u_k_tilde_check2 = 1
                fig, ax = plt.subplots(1, 1)
                ax.plot([0, 1], [u_k_tilde_check2, u_k_tilde_check2], label='u_k_tilde_check2')
                ax.legend()
                plt.show()
                T_k_check = 0
                for i in range(len(T_k)):
                    if np.all(T_k[i][0] == projection(T_k[i][0], u_k, delta)):
                        T_k_check += 1
                fig, ax = plt.subplots(1, 1)
                ax.plot([0, 1], [T_k_check, T_k_check], label='T_k_check')
                ax.legend()
                plt.show()
                # u_k_next = enOpt1(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff, F, proj=lambda mu: projection(mu, u_k, delta), Cov=C_k)[0]
                u_k_next = enOpt(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff, proj=lambda mu: projection(mu, u_k, delta), Cov=C_k)[0]
                print(u_k_next)
                F_k_next = F(u_k_next)
                rho_k = (F_k_next-F_k)/(F_ML_k(u_k_next)-F_ML_k_u_k)
                if rho_k < 0.25:
                    delta *= 0.25
                else:
                    # if rho_k > 0.75 and np.max(np.abs(C_k_inv.dot(u_k-u_k_next))) == delta:
                    if rho_k > 0.75 and np.max(np.abs(u_k-u_k_next)) == delta:
                        delta *= 2
                if rho_k <= 0:
                    u_k_next = u_k.copy()
                else:
                    trustRegionFlag = False
                deltaList.append(delta)
                fig, ax = plt.subplots(1, 1)
                ax.plot(range(len(deltaList)), deltaList, label='delta')
                ax.legend()
                plt.show()
            # evalMLM_delta(F_ML_k, F, u_k, u_k_next, delta)
            # evalMLM_delta1(F_ML_k, F, u_k, u_k_next, delta)
            fig, ax = plt.subplots(1, 1)
            ax.plot(range(len(deltaList)), deltaList, label='delta')
            ax.legend()
            plt.show()

        fails_iter.append(fails)
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k_next, label='u_k_next: {}'.format(k))
        ax.legend()
        plt.show()
        fig, ax = plt.subplots(1, 1)
        for i in range(len(T_k)):
            ax.plot(t, T_k[i][0])
        plt.show()
        """
        u_k_diff = u_k_next-u_k_tilde
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k_diff, label='u_k_diff: {}'.format(k))
        ax.legend()
        plt.show()
        """
        u_k_tilde, T_k, C_k, F_k_tilde = optStep(F, u_k_next, N, k, T_k, C_k, F_k_next, beta_1, beta_2, r, eps_o, nu_1, var_o, correlationCoeff)
        covList = []
        for i in range(len(C_k)):
            covList.append(C_k[i, i])
        tCov = range(len(C_k))
        fig, ax = plt.subplots(1, 1)
        ax.plot(tCov, covList, label='Cov')
        ax.legend()
        plt.show()
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k_tilde, label='u_k_tilde: {}'.format(k+1))
        ax.legend()
        plt.show()
        F_k = F_k_next
        u_k = u_k_next.copy()
        k = k+1
        # var_i *= 0.1
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(len(u_k_tilde_check)), u_k_tilde_check, label='u_k_tilde_check')
    ax.legend()
    plt.show()
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(len(fails_iter)), fails_iter, label='fails')
    ax.legend()
    plt.show()
    print('fails: {}'.format(fails))
    return u_k, k


def AML_EnOptNoTR(F, u_0, N, eps_o, eps_i, k_1_o, k_1_i, V_DNN, beta_1, beta_2, r, nu_1, var, correlationCoeff):
    # V_DNN: neurons per hidden layer, activation function (like torch.tanh), size of test set, number of epochs, training batch size, testing batch size, learning rate
    V_DNN[0].insert(0, len(u_0))
    V_DNN[0].insert(len(V_DNN[0]), 1)
    var_o = var
    var_i = var_o  # /100
    # var_i_list = [var_i]
    F_k = F(u_0)
    u_k_tilde, T_k, C_k, F_k_tilde = optStep(F, u_0, N, 0, [], None, F_k, beta_1, beta_2, r, eps_o, nu_1, var_o, correlationCoeff)
    t = np.linspace(0, T, num=nt+1)
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, u_k_tilde, label='u_k_tilde: {}'.format(1))
    ax.legend()
    plt.show()
    k = 1
    u_k = u_0
    u_k_next = u_k.copy()
    u_k_tilde_check = []
    fails_iter = []
    fails = 0
    while (F_k_tilde > F_k+eps_o and k < k_1_o):
        # u_k = u_k_tilde.copy()
        # F_k = F_k_tilde
        T_k_x = np.zeros((N, nt+1))
        for i in range(N):
            T_k_x[i, :] = T_k[i][0]
        minIn = np.zeros(nt+1)
        maxIn = np.zeros(nt+1)
        for i in range(nt+1):
            minIn[i] = np.min(T_k_x[:, i])
            maxIn[i] = np.max(T_k_x[:, i])
        u_k_tilde_check_iter = 0
        if np.all(u_k_tilde == projectionSample(u_k_tilde, minIn, maxIn)):
            u_k_tilde_check_iter = 1
        u_k_tilde_check.append(u_k_tilde_check_iter)
        F_ML_k = train(T_k, V_DNN, minIn, maxIn)
        """
        evalMLM(F_ML_k, T_k, k)
        F_k_next = F_k
        F_ML_k_u_k = F_ML_k(u_k)
        # C_k_inv = LA.inv(C_k)
        maxDiff = 0
        for i in range(len(T_k)):
            diff = np.min(np.abs(u_k-T_k[i][0]))
            if diff > maxDiff:
                maxDiff = diff
        """
        # u_k_next = enOpt1(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff, F, proj=lambda mu: projectionSample(mu, minIn, maxIn))[0]
        # u_k_next = enOpt1(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff, F)[0]
        u_k_next = enOpt(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff, Cov=C_k)[0]
        # u_k_next = enOpt(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff)[0]
        F_k_next = F(u_k_next)
        T_k_min_u = T_k[0][0].copy()
        T_k_min_dist = LA.norm(T_k_min_u-u_k)
        T_k_max_u = T_k[0][0].copy()
        T_k_max_dist = LA.norm(T_k_max_u-u_k)
        for i in range(len(T_k)):
            dist = LA.norm(T_k[i][0]-u_k)
            if dist < T_k_min_dist:
                T_k_min_dist = dist
                T_k_min_u = T_k[i][0].copy()
            if dist > T_k_max_dist:
                T_k_max_dist = dist
                T_k_max_u = T_k[i][0].copy()
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k, label='u_k: {}'.format(k))
        ax.plot(t, u_k_tilde, label='u_k_tilde: {}'.format(k))
        ax.plot(t, u_k_next, label='u_k_next: {}'.format(k))
        ax.plot(t, T_k_min_u, label='T_k_min: {}'.format(k))
        ax.plot(t, T_k_max_u, label='T_k_max: {}'.format(k))
        ax.legend()
        plt.show()
        fig, ax = plt.subplots(1, 1)
        ax.plot([0,1], [LA.norm(u_k_tilde-u_k), LA.norm(u_k_tilde-u_k)], label='u_k_tilde dist: {}'.format(k))
        # ax.plot([0,1], [LA.norm(u_k_next-u_k), LA.norm(u_k_next-u_k)], label='u_k_next dist: {}'.format(k))
        ax.plot([0,1], [T_k_min_dist, T_k_min_dist], label='T_k_min dist: {}'.format(k))
        ax.plot([0,1], [T_k_max_dist, T_k_max_dist], label='T_k_max dist: {}'.format(k))
        ax.legend()
        plt.show()
        fig, ax = plt.subplots(1, 1)
        ax.plot([0,1], [LA.norm(u_k_tilde-u_k), LA.norm(u_k_tilde-u_k)], label='u_k_tilde dist: {}'.format(k))
        ax.plot([0,1], [LA.norm(u_k_next-u_k), LA.norm(u_k_next-u_k)], label='u_k_next dist: {}'.format(k))
        ax.plot([0,1], [T_k_min_dist, T_k_min_dist], label='T_k_min dist: {}'.format(k))
        ax.plot([0,1], [T_k_max_dist, T_k_max_dist], label='T_k_max dist: {}'.format(k))
        ax.legend()
        plt.show()
        # evalMLM_delta(F_ML_k, F, u_k, u_k_next, del
        # var_i = maxDiff/50ta)
        # evalMLM_delta1(F_ML_k, F, u_k, u_k_next, delta)

        if F_k_next <= F_k+eps_o:
            fig, ax = plt.subplots(1, 1)
            ax.plot(range(len(u_k_tilde_check)), u_k_tilde_check, label='u_k_tilde_check')
            ax.legend()
            plt.show()
            print('fail')
            print('F_k_next: {}'.format(F_k_next))
            print('F_k: {}'.format(F_k))
            print('F_k_tilde: {}'.format(F_k_tilde))
            print(k)
            u_k_next = u_k_tilde.copy()
            F_k_next = F_k_tilde
            fails += 1
            return u_k, k

        fails_iter.append(fails)
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k_next, label='u_k_next: {}'.format(k))
        ax.legend()
        plt.show()
        fig, ax = plt.subplots(1, 1)
        for i in range(len(T_k)):
            ax.plot(t, T_k[i][0])
        plt.show()
        """
        u_k_diff = u_k_next-u_k_tilde
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k_diff, label='u_k_diff: {}'.format(k))
        ax.legend()
        plt.show()
        """
        u_k_tilde, T_k, C_k, F_k_tilde = optStep(F, u_k_next, N, k, T_k, C_k, F_k_next, beta_1, beta_2, r, eps_o, nu_1, var_o, correlationCoeff)
        covList = []
        for i in range(len(C_k)):
            covList.append(C_k[i, i])
        tCov = range(len(C_k))
        fig, ax = plt.subplots(1, 1)
        ax.plot(tCov, covList, label='Cov')
        ax.legend()
        plt.show()
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k_tilde, label='u_k_tilde: {}'.format(k+1))
        ax.legend()
        plt.show()
        F_k = F_k_next
        u_k = u_k_next.copy()
        k = k+1
        # var_i *= 0.1
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(len(u_k_tilde_check)), u_k_tilde_check, label='u_k_tilde_check')
    ax.legend()
    plt.show()
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(len(fails_iter)), fails_iter, label='fails')
    ax.legend()
    plt.show()
    print('fails: {}'.format(fails))
    return u_k, k


def AML_EnOptOld(F, u_0, N, eps_o, eps_i, k_1_o, k_1_i, V_DNN, beta_1, beta_2, r, nu_1, var, correlationCoeff):
    # V_DNN: neurons per hidden layer, activation function (like torch.tanh), size of test set, number of epochs, training batch size, testing batch size, learning rate
    V_DNN[0].insert(0, len(u_0))
    V_DNN[0].insert(len(V_DNN[0]), 1)
    var_o = var
    var_i = var_o/20
    # var_i_list = [var_i]
    delta = 10
    F_k = F(u_0)
    u_k_tilde, T_k, C_k, F_k_tilde = optStep(F, u_0, N, 0, [], 0, F_k, beta_1, beta_2, r, eps_o, nu_1, var_o, correlationCoeff)
    t = np.linspace(0, T, num=nt+1)
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, u_k_tilde, label='u_k_tilde: {}'.format(1))
    ax.legend()
    plt.show()
    k = 1
    u_k = u_0
    u_k_next = u_k.copy()
    while (F_k_tilde > F_k+eps_o and k < k_1_o):
        # u_k = u_k_tilde.copy()
        # F_k = F_k_tilde
        F_ML_k = train(T_k, V_DNN)
        evalMLM(F_ML_k, T_k, k)
        F_k_next = F_k
        F_ML_k_u_k = F_ML_k(u_k)
        deltaList = [delta]
        # C_k_inv = LA.inv(C_k)
        maxDiff = 0
        for i in range(len(T_k)):
            diff = np.min(np.abs(u_k-T_k[i][0]))
            if diff > maxDiff:
                maxDiff = diff
        var_i = maxDiff/50
        trustRegionFlag = True
        while trustRegionFlag:
            u_k_tilde_check = 0
            if np.all(u_k_tilde == projection(u_k_tilde, u_k, delta)):
                u_k_tilde_check = 1
            fig, ax = plt.subplots(1, 1)
            ax.plot([0, 1], [u_k_tilde_check, u_k_tilde_check], label='u_k_tilde_check')
            ax.legend()
            plt.show()
            T_k_check = 0
            for i in range(len(T_k)):
                if np.all(T_k[i][0] == projection(T_k[i][0], u_k, delta)):
                    T_k_check += 1
            fig, ax = plt.subplots(1, 1)
            ax.plot([0, 1], [T_k_check, T_k_check], label='T_k_check')
            ax.legend()
            plt.show()
            if u_k_tilde_check == 1:
                # u_k_next = enOpt(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff, proj=lambda mu: projection(mu, u_k, delta, C_k_inv))[0]
                # u_k_next = enOpt(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff, proj=lambda mu: projection(mu, u_k, delta))[0]
                u_k_next = enOpt1(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff, F, proj=lambda mu: projection(mu, u_k, delta))[0]
                # evalMLM_delta(F_ML_k, F, u_k, u_k_next, delta)
                # evalMLM_delta1(F_ML_k, F, u_k, u_k_next, delta)
                print(u_k_next)
                F_k_next = F(u_k_next)
                rho_k = (F_k_next-F_k)/(F_ML_k(u_k_next)-F_ML_k_u_k)
                if rho_k < 0.25:
                    delta *= 0.25
                else:
                    # if rho_k > 0.75 and np.max(np.abs(C_k_inv.dot(u_k-u_k_next))) == delta:
                    if rho_k > 0.75 and np.max(np.abs(u_k-u_k_next)) == delta:
                        delta *= 2
                if rho_k <= 0:
                    u_k_next = u_k.copy()
                else:
                    trustRegionFlag = False
            else:
                u_k_next = u_k_tilde
                trustRegionFlag = False
            deltaList.append(delta)
            fig, ax = plt.subplots(1, 1)
            ax.plot(range(len(deltaList)), deltaList, label='delta')
            ax.legend()
            plt.show()
        evalMLM_delta(F_ML_k, F, u_k, u_k_next, delta)
        evalMLM_delta1(F_ML_k, F, u_k, u_k_next, delta)
        tDelta = range(len(deltaList))
        fig, ax = plt.subplots(1, 1)
        ax.plot(tDelta, deltaList, label='delta')
        ax.legend()
        plt.show()
        """
        if F_k_next <= F_k+eps_o:
            print('fail')
            print(k)
            return u_k, k
        """
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k_next, label='u_k_next: {}'.format(k))
        ax.legend()
        plt.show()
        fig, ax = plt.subplots(1, 1)
        for i in range(len(T_k)):
            ax.plot(t, T_k[i][0])
        plt.show()
        u_k_diff = u_k_next-u_k_tilde
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k_diff, label='u_k_diff: {}'.format(k))
        ax.legend()
        plt.show()
        u_k_tilde, T_k, C_k, F_k_tilde = optStep(F, u_k_next, N, k, T_k, C_k, F_k_next, beta_1, beta_2, r, eps_o, nu_1, var_o, correlationCoeff)
        covList = []
        for i in range(len(C_k)):
            covList.append(C_k[i, i])
        tCov = range(len(C_k))
        fig, ax = plt.subplots(1, 1)
        ax.plot(tCov, covList, label='Cov')
        ax.legend()
        plt.show()
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k_tilde, label='u_k_tilde: {}'.format(k+1))
        ax.legend()
        plt.show()
        F_k = F_k_next
        u_k = u_k_next.copy()
        k = k+1
        # var_i *= 0.1
    return u_k, k


def AML_EnOptOldOld(F, u_0, N, eps_o, eps_i, k_1_o, k_1_i, V_DNN, beta_1, beta_2, r, nu_1, var, correlationCoeff):
    # V_DNN: neurons per hidden layer, activation function (like torch.tanh), size of test set, number of epochs, training batch size, testing batch size, learning rate
    V_DNN[0].insert(0, len(u_0))
    V_DNN[0].insert(len(V_DNN[0]), 1)
    var_o = var
    var_i = var_o/20
    # var_i_list = [var_i]
    delta = 100
    F_k = F(u_0)
    u_k_tilde, T_k, C_k, F_k_tilde = optStep(F, u_0, N, 0, [], 0, F_k, beta_1, beta_2, r, eps_o, nu_1, var_o, correlationCoeff)
    t = np.linspace(0, T, num=nt+1)
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, u_k_tilde, label='u_k_tilde: {}'.format(1))
    ax.legend()
    plt.show()
    k = 1
    u_k = u_0
    u_k_next = u_k.copy()
    while (F_k_tilde > F_k+eps_o and k < k_1_o):
        # u_k = u_k_tilde.copy()
        # F_k = F_k_tilde
        T_k_x = np.zeros((N, nt+1))
        for i in range(N):
            T_k_x[i, :] = T_k[i][0]
        minIn = np.zeros(nt+1)
        maxIn = np.zeros(nt+1)
        for i in range(nt+1):
            minIn[i] = np.min(T_k_x[:, i])
            maxIn[i] = np.max(T_k_x[:, i])
        F_ML_k = train(T_k, V_DNN, minIn, maxIn)
        # evalMLM(F_ML_k, T_k, k)
        F_k_next = F_k
        F_ML_k_u_k = F_ML_k(u_k)
        deltaList = [delta]
        # C_k_inv = LA.inv(C_k)
        maxDiff = 0
        for i in range(len(T_k)):
            diff = np.min(np.abs(u_k-T_k[i][0]))
            if diff > maxDiff:
                maxDiff = diff
        var_i = maxDiff/50
        trustRegionFlag = True
        while trustRegionFlag:
            u_k_tilde_check = 0
            if np.all(u_k_tilde == projection(u_k_tilde, u_k, delta)):
                u_k_tilde_check = 1
            fig, ax = plt.subplots(1, 1)
            ax.plot([0, 1], [u_k_tilde_check, u_k_tilde_check], label='u_k_tilde_check')
            ax.legend()
            plt.show()
            T_k_check = 0
            for i in range(len(T_k)):
                if np.all(T_k[i][0] == projection(T_k[i][0], u_k, delta)):
                    T_k_check += 1
            fig, ax = plt.subplots(1, 1)
            ax.plot([0, 1], [T_k_check, T_k_check], label='T_k_check')
            ax.legend()
            plt.show()
            # u_k_next = enOpt(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff, proj=lambda mu: projection(mu, u_k, delta, C_k_inv))[0]
            # u_k_next = enOpt(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff, proj=lambda mu: projection(mu, u_k, delta))[0]
            u_k_next = enOpt1(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var, correlationCoeff, F, proj=lambda mu: projection(mu, u_k, delta))[0]
            print(u_k_next)
            F_k_next = F(u_k_next)
            rho_k = (F_k_next-F_k)/(F_ML_k(u_k_next)-F_ML_k_u_k)
            if rho_k < 0.25:
                delta *= 0.25
            else:
                # if rho_k > 0.75 and np.max(np.abs(C_k_inv.dot(u_k-u_k_next))) == delta:
                if rho_k > 0.75 and np.max(np.abs(u_k-u_k_next)) == delta:
                    delta *= 2
            if rho_k <= 0:
                u_k_next = u_k.copy()
            else:
                trustRegionFlag = False
            deltaList.append(delta)
            fig, ax = plt.subplots(1, 1)
            ax.plot(range(len(deltaList)), deltaList, label='delta')
            ax.legend()
            plt.show()
        # evalMLM_delta(F_ML_k, F, u_k, u_k_next, delta)
        # evalMLM_delta1(F_ML_k, F, u_k, u_k_next, delta)
        tDelta = range(len(deltaList))
        fig, ax = plt.subplots(1, 1)
        ax.plot(tDelta, deltaList, label='delta')
        ax.legend()
        plt.show()
        """
        if F_k_next <= F_k+eps_o:
            print('fail')
            print(k)
            return u_k, k
        """
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k_next, label='u_k_next: {}'.format(k))
        ax.legend()
        plt.show()
        fig, ax = plt.subplots(1, 1)
        for i in range(len(T_k)):
            ax.plot(t, T_k[i][0])
        plt.show()
        """
        u_k_diff = u_k_next-u_k_tilde
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k_diff, label='u_k_diff: {}'.format(k))
        ax.legend()
        plt.show()
        """
        u_k_tilde, T_k, C_k, F_k_tilde = optStep(F, u_k_next, N, k, T_k, C_k, F_k_next, beta_1, beta_2, r, eps_o, nu_1, var_o, correlationCoeff)
        covList = []
        for i in range(len(C_k)):
            covList.append(C_k[i, i])
        tCov = range(len(C_k))
        fig, ax = plt.subplots(1, 1)
        ax.plot(tCov, covList, label='Cov')
        ax.legend()
        plt.show()
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k_tilde, label='u_k_tilde: {}'.format(k+1))
        ax.legend()
        plt.show()
        F_k = F_k_next
        u_k = u_k_next.copy()
        k = k+1
        # var_i *= 0.1
    return u_k, k


def AML_EnOpt1(F, u_0, N, eps_o, eps_i, k_1_o, k_1_i, V_DNN, beta_1, beta_2, r, nu_1, var, correlationCoeff):
    # V_DNN: neurons per hidden layer, activation function (like torch.tanh), size of test set, number of epochs, training batch size, testing batch size, learning rate
    V_DNN[0].insert(0, len(u_0))
    V_DNN[0].insert(len(V_DNN[0]), 1)
    var_o = var
    var_i = var_o#/20
    # var_i_list = [var_i]
    delta = var
    F_k = F(u_0)
    u_k_tilde, T_k, C_k, F_k_tilde = optStep(F, u_0, N, 0, [], 0, F_k, beta_1, beta_2, r, eps_o, nu_1, var_o, correlationCoeff)
    k = 1
    u_k = u_0
    u_k_next = u_k.copy()
    while (F_k_tilde > F_k+eps_o and k < k_1_o):
        # u_k = u_k_tilde.copy()
        # F_k = F_k_tilde
        F_ML_k = train(T_k, V_DNN)
        F_k_next = F_k
        F_ML_k_u_k = F_ML_k(u_k)
        deltaList = [delta]
        #C_k_inv = LA.inv(C_k)
        while np.all(u_k_next == u_k):
            # u_k_next = enOpt(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff, proj = lambda mu: projection(mu, u_k, delta, C_k_inv))[0]
            u_k_next = enOpt(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff, proj = lambda mu: projection(mu, u_k, delta))[0]
            print(u_k_next)
            F_k_next = F(u_k_next)
            rho_k = (F_k_next-F_k)/(F_ML_k(u_k_next)-F_ML_k_u_k)
            if rho_k < 0.25:
                delta *= 0.25
            else:
                #if rho_k > 0.75 and np.max(np.abs(C_k_inv.dot(u_k-u_k_next))) == delta:
                if rho_k > 0.75 and np.max(np.abs(u_k-u_k_next)) == delta:
                    delta *= 2
            if rho_k <= 0:
                u_k_next = u_k.copy()
            deltaList.append(delta)
        tDelta = range(len(deltaList))
        fig, ax = plt.subplots(1, 1)
        ax.plot(tDelta, deltaList, label='delta')
        ax.legend()
        plt.show()
        """
        if F_k_next <= F_k+eps_o:
            print('fail')
            print(k)
            return u_k, k
        """
        t = np.linspace(0, T, num=nt+1)
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k_next, label='u_k_next: {}'.format(k))
        ax.legend()
        plt.show()
        fig, ax = plt.subplots(1, 1)
        for i in range(len(T_k)):
            ax.plot(t, T_k[i][0])
        plt.show()
        u_k_diff = u_k_next-u_k_tilde
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k_diff, label='u_k_diff: {}'.format(k))
        ax.legend()
        plt.show()
        u_k_tilde, T_k, C_k, F_k_tilde = optStep(F, u_k_next, N, k, T_k, C_k, F_k_next, beta_1, beta_2, r, eps_o, nu_1, var_o, correlationCoeff)
        covList = []
        for i in range(len(C_k)):
            covList.append(C_k[i,i])
        tCov = range(len(C_k))
        fig, ax = plt.subplots(1, 1)
        ax.plot(tCov, covList, label='Cov')
        ax.legend()
        plt.show()
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k_tilde, label='u_k_tilde: {}'.format(k+1))
        ax.legend()
        plt.show()
        F_k = F_k_next
        u_k = u_k_next.copy()
        k = k+1
    return u_k, k


def AML_EnOpt2(F, u_0, N, eps_o, eps_i, k_1_o, k_1_i, V_DNN, beta_1, beta_2, r, nu_1, var, correlationCoeff):
    # V_DNN: neurons per hidden layer, activation function (like torch.tanh), size of test set, number of epochs, training batch size, testing batch size, learning rate
    V_DNN[0].insert(0, len(u_0))
    V_DNN[0].insert(len(V_DNN[0]), 1)
    var_o = var
    var_i = var_o#/20
    # var_i_list = [var_i]
    delta = var
    F_k = F(u_0)
    u_k_tilde, T_k, C_k, F_k_tilde = optStep(F, u_0, N, 0, [], 0, F_k, beta_1, beta_2, r, eps_o, nu_1, var_o, correlationCoeff)
    k = 1
    u_k = u_0
    u_k_next = u_k.copy()
    while (F_k_tilde > F_k+eps_o and k < k_1_o):
        # u_k = u_k_tilde.copy()
        # F_k = F_k_tilde
        delta = 0
        for i in range(len(T_k)):
          diff = np.min(np.abs(u_k-T_k[i][0]))
          if diff > delta:
              delta = diff
        F_ML_k = train(T_k, V_DNN)
        F_k_next = F_k
        # deltaList = [delta]
        deltaList = [delta, delta]
        #C_k_inv = LA.inv(C_k)
        u_k_next = enOpt(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff, proj = lambda mu: projection(mu, u_k, delta))[0]
        F_k_next = F(u_k_next)
        """
        while np.all(u_k_next == u_k):
            # u_k_next = enOpt(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff, proj = lambda mu: projection(mu, u_k, delta, C_k_inv))[0]
            u_k_next = enOpt(F_ML_k, u_k, N, eps_i, k_1_i, beta_1, beta_2, r, nu_1, var_i, correlationCoeff, proj = lambda mu: projection(mu, u_k, delta))[0]
            F_k_next = F(u_k_next)
            rho_k = (F_k_next-F_k)/(F_ML_k(u_k_next)-F_ML_k_u_k)
            if rho_k < 0.25:
                delta *= 0.25
            else:
                #if rho_k > 0.75 and np.max(np.abs(C_k_inv.dot(u_k-u_k_next))) == delta:
                if rho_k > 0.75 and np.max(np.abs(u_k-u_k_next)) == delta:
                    delta *= 2
            if rho_k <= 0:
                u_k_next = u_k.copy()
            deltaList.append(delta)
        """
        tDelta = range(len(deltaList))
        fig, ax = plt.subplots(1, 1)
        ax.plot(tDelta, deltaList, label='delta')
        ax.legend()
        plt.show()
        """
        if F_k_next <= F_k+eps_o:
            print('fail')
            print(k)
            return u_k, k
        """
        t = np.linspace(0, T, num=nt+1)
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k_next, label='u_k_next: {}'.format(k))
        ax.legend()
        plt.show()
        fig, ax = plt.subplots(1, 1)
        for i in range(len(T_k)):
            ax.plot(t, T_k[i][0])
        plt.show()
        u_k_diff = u_k_next-u_k_tilde
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k_diff, label='u_k_diff: {}'.format(k))
        ax.legend()
        plt.show()
        u_k_tilde, T_k, C_k, F_k_tilde = optStep(F, u_k_next, N, k, T_k, C_k, F_k_next, beta_1, beta_2, r, eps_o, nu_1, var_o, correlationCoeff)
        covList = []
        for i in range(len(C_k)):
            covList.append(C_k[i,i])
        tCov = range(len(C_k))
        fig, ax = plt.subplots(1, 1)
        ax.plot(tCov, covList, label='Cov')
        ax.legend()
        plt.show()
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, u_k_tilde, label='u_k_tilde: {}'.format(k+1))
        ax.legend()
        plt.show()
        F_k = F_k_next
        u_k = u_k_next.copy()
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
        print(name + ' q error = {}'.format(qErr))
        print(name + ' q LInf-error = {}'.format(np.max(np.abs(qErr))))
        print(name + ' q average error = {}'.format(np.sum(np.abs(qErr))/len(qErr)))
        # print(name + ' q relative LInf-error = {}'.format(np.max(np.abs(qErr))/np.max((qParamOpt))))
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
N = 100
eps = 1e-9
k_1 = 1000
beta_1 = 1
beta_2 = 1
r = 0.5
nu_1 = 10
var = 0.001
correlationCoeff = 0.9


# analytical minimizer
qParam = []
for i in range(nt+1):
    qParam.append(-np.pi**4*(np.exp(a*np.pi**2*i/nt*T)-np.exp(a*np.pi**2*T)))
out, fom, data, y1, y2 = J(qParam, a, T, grid_intervals, nt)
u = fom.solve({'a': a})


def analytical():
    result('Analytical', qParam, qParam, out, fom, data, u, y1, y2, a, T, nt)

"""
# EnOpt
qParamOpt, k = FOM_EnOpt(init, N, eps, k_1, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt)
outOpt, fomOpt, dataOpt, y1Opt, y2Opt = J(qParamOpt, a, T, grid_intervals, nt)
uOpt = fomOpt.solve({'a': a})


def opt1():
    result('EnOpt', qParamOpt, qParam, outOpt, fomOpt, dataOpt, uOpt, y1Opt, y2Opt, a, T, nt)
"""

# optimized control function using the AML EnOpt minimizer
eps_o = 1e-9
eps_i = 1e-9
k_1_o = k_1
k_1_i = k_1
# V_DNN: neurons per hidden layer, activation function (like torch.tanh), size of test set, number of epochs, training batch size, testing batch size, learning rate
# V_DNN = [[100, 100, 40], torch.tanh, 50, 100, 100, 10, 1e-4]
V_DNN = [[500, 500], torch.tanh, 50, 2000, 100, 10, 1e-2]
# V_DNN = [[200, 200, 200], torch.tanh, 50, 5000, 100, 10, 1e-5]
# V_DNN = [[200, 200, 100, 50], torch.tanh, 50, 100, 100, 10, 1e-4]
# V_DNN = [[25, 25], torch.tanh, 50, 100, 50, 5, 1e-4]

qParamAMLOpt, kAML = ROM_EnOpt(init, N, eps_o, eps_i, k_1_o, k_1_i, V_DNN, beta_1, beta_2, r, nu_1, var, correlationCoeff, a, T, grid_intervals, nt)
print(qParamAMLOpt, kAML)
outAMLOpt, fomAMLOpt, dataAMLOpt, y1AMLOpt, y2AMLOpt = J(qParamAMLOpt, a, T, grid_intervals, nt)
uAMLOpt = fomAMLOpt.solve({'a': a})


def opt2():
    result('AML_EnOpt', qParamAMLOpt, qParam, outAMLOpt, fomAMLOpt, dataAMLOpt, uAMLOpt, y1AMLOpt, y2AMLOpt, a, T, nt)

"""
# optimized control function using the L_BFGS_B_minimizer
qParamOptBFGS = L_BFGS_B_minimizer(init, a, T, grid_intervals, nt)
outOptBFGS, fomOptBFGS, dataOptBFGS, y1OptBFGS, y2OptBFGS = J(qParamOptBFGS, a, T, grid_intervals, nt)
uOptBFGS = fomOptBFGS.solve({'a': a})


def opt3():
    result('L_BFGS_B', qParamOptBFGS, qParam, outOptBFGS, fomOptBFGS, dataOptBFGS, uOptBFGS, y1OptBFGS, y2OptBFGS, a, T, nt)
"""
"""
T_k = [[np.array([-40.00487241, -40.00407639, -40.0063089 , -40.00636447,
       -40.00510211, -40.0063463 , -40.0053436 , -40.00658185,
       -40.00562901, -40.00481124, -40.00262891]), -4.3219718214993685], [np.array([-39.99810605, -39.99671477, -39.9974675 , -39.99884012,
       -39.99768732, -39.9966866 , -39.99910849, -40.000109  ,
       -40.00200245, -40.00197303, -40.00326857]), -4.321961182329264], [np.array([-39.99948767, -40.00015782, -39.99968415, -40.00097932,
       -40.00030969, -40.00080768, -40.00110685, -40.00035003,
       -40.00012607, -39.99993789, -39.9997851 ]), -4.321956214877475], [np.array([-39.99746972, -39.99570938, -39.99616834, -39.99762459,
       -39.99808248, -39.99751036, -39.9989023 , -39.99999128,
       -40.00198238, -40.00179074, -40.0007511 ]), -4.3219614459522795], [np.array([-40.00163663, -40.00364395, -40.0045095 , -40.00554076,
       -40.00812964, -40.00906352, -40.00725983, -40.00583655,
       -40.00411683, -40.00349097, -40.00380864]), -4.32197506922419], [np.array([-40.00186155, -40.00224204, -40.00418156, -40.00340717,
       -40.00317383, -40.00247862, -40.00164247, -39.9994556 ,
       -39.99989034, -39.99954583, -39.99944015]), -4.321952664127266], [np.array([-40.00016798, -39.99917946, -39.99987463, -39.99924288,
       -39.99944679, -39.99994732, -40.00006716, -40.00122806,
       -40.00278103, -40.00326205, -40.00254897]), -4.321962889060513], [np.array([-39.99726598, -39.99698666, -39.99787161, -39.99727513,
       -39.99751131, -39.99672849, -39.99708057, -39.99865431,
       -39.99933507, -39.99719985, -39.9994125 ]), -4.32194968409204], [np.array([-40.00002613, -40.00030734, -40.00184325, -40.00045929,
       -39.9997424 , -39.99998407, -40.00083086, -39.99983679,
       -39.99892994, -39.99902631, -39.99901122]), -4.321951629538084], [np.array([-39.99935655, -40.00097682, -40.00202206, -40.00179537,
       -40.00285204, -40.00431062, -40.00457095, -40.00490957,
       -40.0025601 , -40.00139029, -40.00017704]), -4.321967180847871]]
JFirst = -J(np.array([-40.00487241, -40.00407639, -40.0063089, -40.00636447, -40.00510211, -40.0063463, -40.0053436, -40.00658185, -40.00562901, -40.00481124, -40.00262891]), -np.sqrt(5), 0.1, nt=10)[0]
JSec = -J(np.array([-39.99810605, -39.99671477, -39.9974675, -39.99884012, -39.99768732, -39.9966866, -39.99910849, -40.000109, -40.00200245, -40.00197303, -40.00326857]), -np.sqrt(5), 0.1, nt=10)[0]
JLast = -J(np.array([-39.99935655, -40.00097682, -40.00202206, -40.00179537, -40.00285204, -40.00431062, -40.00457095, -40.00490957, -40.0025601, -40.00139029, -40.00017704]), -np.sqrt(5), 0.1, nt=10)[0]
# V_DNN: neurons per hidden layer, activation function (like torch.tanh), size of test set, number of epochs, training batch size, testing batch size, learning rate
V_DNN = [[300, 300], torch.tanh, 2, 300, 5, 1, 1e-5]
V_DNN[0].insert(0, 11)
V_DNN[0].insert(len(V_DNN[0]), 1)
print(T_k[0])
# T_k_x = np.zeros((N, nt+1))
T_k_x = np.zeros((10, nt+1))
for i in range(10):
    T_k_x[i, :] = T_k[i][0]
"""
"""
minIn = []
maxIn = []
for i in range(nt+1):
    minIn.append(np.min(T_k_x[:, i]))
    maxIn.append(np.max(T_k_x[:, i]))
"""
"""
minIn = np.zeros(nt+1)
maxIn = np.zeros(nt+1)
for i in range(nt+1):
    minIn[i] = np.min(T_k_x[:, i])
    maxIn[i] = np.max(T_k_x[:, i])
print('T_k_x: {}'.format(T_k_x))
torch.manual_seed(10)
f = trainOld(T_k, V_DNN)
fFirst = f(np.array([-39.99935655, -40.00097682, -40.00202206, -40.00179537, -40.00285204, -40.00431062, -40.00457095, -40.00490957, -40.0025601, -40.00139029, -40.00017704]))
fSec = f(np.array([-39.99810605, -39.99671477, -39.9974675, -39.99884012, -39.99768732, -39.9966866, -39.99910849, -40.000109, -40.00200245, -40.00197303, -40.00326857]))
fLast = f(np.array([-40.00487241, -40.00407639, -40.0063089, -40.00636447, -40.00510211, -40.0063463, -40.0053436, -40.00658185, -40.00562901, -40.00481124, -40.00262891]))
print(fFirst)
print(JFirst)
print(fFirst-JFirst)
print(fSec)
print(JSec)
print(fSec-JSec)
print(fLast)
print(JLast)
print(fLast-JLast)
"""
