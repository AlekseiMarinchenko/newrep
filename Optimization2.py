import numpy as np
import matplotlib.pyplot as plt
import time
def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad
def numerical_hessian(f, x, h=1e-5):
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_pp = x.copy()
            x_pp[i] += h
            x_pp[j] += h
            x_pm = x.copy()
            x_pm[i] += h
            x_pm[j] -= h
            x_mp = x.copy()
            x_mp[i] -= h
            x_mp[j] += h
            x_mm = x.copy()
            x_mm[i] -= h
            x_mm[j] -= h
            hess[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h ** 2)
    return hess
def f_trig(x):
    return np.sum(np.sin(x) + np.cos(x))
def f_poly(x):
    return np.sum(x ** 3 + x ** 2 * np.roll(x, -1))  # x_i^3 + x_i^2 * x_{i+1}
def f_comp(x):
    return np.sum(np.exp(x) + np.log(1 + x ** 2) + np.arctan(x))
def grad_trig(x):
    return np.cos(x) - np.sin(x)
def hess_trig(x):
    return np.diag(-np.sin(x) - np.cos(x))
def grad_poly(x):
    n = len(x)
    grad = 3 * x ** 2 + 2 * x * np.roll(x, -1) + np.roll(x ** 2, 1)
    grad[-1] = 3 * x[-1] ** 2 + 2 * x[-1] * x[0]
    return grad
def hess_poly(x):
    n = len(x)
    hess = np.zeros((n, n))
    for i in range(n):
        hess[i, i] = 6 * x[i] + 2 * np.roll(x, -1)[i]
        hess[i, (i + 1) % n] = 2 * x[i]
        hess[(i + 1) % n, i] = 2 * x[i]
    return hess
def grad_comp(x):
    return np.exp(x) + (2 * x) / (1 + x ** 2) + 1 / (1 + x ** 2)
def hess_comp(x):
    return np.diag(np.exp(x) + (2 * (1 - x ** 2)) / (1 + x ** 2) ** 2 - (2 * x) / (1 + x ** 2) ** 2)

def evaluate_accuracy(f, grad_analytical, hess_analytical, x, h_values):
    grad_errors = []
    hess_errors = []
    for h in h_values:
        grad_num = numerical_gradient(f, x, h)
        hess_num = numerical_hessian(f, x, h)
        grad_error = np.linalg.norm(grad_num - grad_analytical(x))
        hess_error = np.linalg.norm(hess_num - hess_analytical(x))
        grad_errors.append(grad_error)
        hess_errors.append(hess_error)
    return grad_errors, hess_errors

def plot_errors(h_values, errors, title):
    plt.loglog(h_values, errors, marker='o')
    plt.xlabel('Шаг h (лог. шкала)')
    plt.ylabel('Норма ошибки (лог. шкала)')
    plt.title(title)
    plt.grid(True)
    plt.show()


n = 2  
x_test = np.array([0.5, 1.0])  
h_values = np.logspace(-10, -1, 20)  
grad_errors, hess_errors = evaluate_accuracy(f_trig, grad_trig, hess_trig, x_test, h_values)
plot_errors(h_values, grad_errors, 'Ошибка градиента (тригонометрическая функция)')
plot_errors(h_values, hess_errors, 'Ошибка гессиана (тригонометрическая функция)')
grad_errors, hess_errors = evaluate_accuracy(f_poly, grad_poly, hess_poly, x_test, h_values)
plot_errors(h_values, grad_errors, 'Ошибка градиента (полином)')
plot_errors(h_values, hess_errors, 'Ошибка гессиана (полином)')
grad_errors, hess_errors = evaluate_accuracy(f_comp, grad_comp, hess_comp, x_test, h_values)
plot_errors(h_values, grad_errors, 'Ошибка градиента (композиция)')
plot_errors(h_values, hess_errors, 'Ошибка гессиана (композиция)')


def f_linear(x):
    return np.sum(x ** 2)


def time_gradient_hessian(n_values):
    grad_times = []
    hess_times = []
    for n in n_values:
        x = np.random.rand(n)
        start = time.time()
        numerical_gradient(f_linear, x)
        grad_times.append(time.time() - start)

        start = time.time()
        numerical_hessian(f_linear, x)
        hess_times.append(time.time() - start)
    return grad_times, hess_times


n_values = np.arange(1, 51)  
grad_times, hess_times = time_gradient_hessian(n_values)

plt.plot(n_values, grad_times, label='Градиент')
plt.plot(n_values, hess_times, label='Гессиан')
plt.xlabel('Размерность n')
plt.ylabel('Время (сек)')
plt.title('Зависимость времени вычисления от размерности')
plt.legend()
plt.grid(True)
plt.show()
