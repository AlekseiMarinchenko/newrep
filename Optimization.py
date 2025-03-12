import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
f1 = lambda x: -5*x**5 + 4*x**4 -12*x**3 +11*x**2 -2*x +1 #[-0.5, 0.5]
f2 = lambda x: (lambda xc: -(np.log(xc-2))**2 + (np.log(10-xc))**2 - xc*0.2)(np.clip(x, 2 + 1e-15, 10 - 1e-15)) #[6, 9.9]
f3 = lambda x: -3*x*np.sin(0.75*x) + np.exp(-2*x) #[0, 2pi]
f4 = lambda x: np.exp(3*x) + 5*np.exp(-2*x) #[0, 1]
f5 = lambda x: 0.2*x*np.log(x) + (x -2.3)**2 #[0.5, 2.5]
f6 = lambda x: x**2 + 10 * np.sin(3 * x) + np.cos(5 * x) #[-5, 5]
f7 = lambda x: x**2 + 5 * np.sin(10 * x) #[-3, 3]
E = np.exp(-10)
N = 1000

def golden_section(f, a, b, E, N):
    count = 0
    K = (np.sqrt(5) - 1) / 2
    L = [0 for i in range(N + 2)]
    L[0] = b - a
    L[1] = K * L[0]
    x = b - L[1]
    y = a + L[1]
    fx = f(x)
    fy = f(y)
    for k in range(1, N+1):
        L[k + 1] = K * L[k]
        if fx < fy:
            b = y
            y = x
            fy = fx
            x = b - L[k + 1]
            fx = f(x)
        else:
            a = x
            x = y
            fx = fy
            y = a + L[k + 1]
            fy = f(y)
        if L[k + 1] < E:
            x_min = x if fx < fy else y
            fx_min = min(fx, fy)
            count += 1
            break
    if count == 0:
        raise 'Слишком малый N для определения значений, при данном эпсилон'
    else:
        return x_min, fx_min


print('Метод золотого сечения для N=', N, 'и эпсилон =', E)
print(f'Для первой функции: {golden_section(f1, -0.5, 0.5, E, N)} ')
print(f'Для второй функции: {golden_section(f2, 6, 9.9, E, N)} ')
print(f'Для третьей функции: {golden_section(f3, 0, 2* np.pi, E, N)} ')
print(f'Для четвертой функции: {golden_section(f4, 0, 1, E, N)} ')
print(f'Для пятой функции: {golden_section(f5, 0.5, 2.5, E, N)} ')


def parabol(f, a, b, E, N):
    while True:
        c = random.uniform(a, b)
        if f(c) < f(a) and f(c) < f(b):
            break
    fa, fb, fc = f(a), f(b), f(c)
    count = 0
    for i in range(1, N + 1):
        u = c - ((c - a)**2 * (f(c) - f(b)) - (c - b)**2 * (f(c) - f(a)))/(2 * ((c - a)* (f(c) - f(b)) - (c - b)* (f(c) - f(a))))
        fu = f(u)
        if fu < fc:
            b = c
            fb = fc
            c = u
            fc = fu
        else:
            a = c
            fa = fc
            c = u
            fc = fu
        if abs(a - c) < E:
            x_min = a if fa < fb else b
            fx_min = min(fa, fb)
            count += 1
            break
    if count == 0:
        raise 'Слишком малый N для определения значений, при данном эпсилон'
    else:
        return x_min, fx_min

def parabol_ls(f, x1, x3, tol, max_iter):
    while True:
        x2 = random.uniform(x1, x3)
        if f(x2) < f(x1) and f(x2) < f(x3):
            break
    best_x = x2
    best_fx = f(x2)
    for _ in range(max_iter):
        A = np.array([
            [x1 ** 2, x1, 1],
            [x2 ** 2, x2, 1],
            [x3 ** 2, x3, 1]
        ])
        b = np.array([f(x1), f(x2), f(x3)])
        try:
            a, b_coef, c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        if a == 0:
            break
        x_min = -b_coef / (2 * a)
        f_min = f(x_min)
        if abs(x_min - best_x) < tol:
            return (x_min, f_min)
        if f_min < best_fx:
            best_x = x_min
            best_fx = f_min
        if x_min < x2:
            x3 = x2 if f_min < f(x2) else x_min
            x2 = x_min if f_min < f(x2) else x2
        else:
            x1 = x2 if f_min < f(x2) else x_min
            x2 = x_min if f_min < f(x2) else x2
    return (best_x, best_fx)

print('\nМетод парабол через linealg.solve для N=', N, 'и эпсилон =', E)
print(f'Для первой функции {parabol_ls(f1, -0.5, 0.5, E, N)}')
print(f'Для второй функции{parabol_ls(f2, 6, 9.9, E, N)}')
print(f'Для третьей функции{parabol_ls(f3, 0, 2* np.pi, E, N)}')
print(f'Для четвертой функции{parabol_ls(f4, 0, 1, E, N)}')
print(f'Для пятой функции{parabol_ls(f5, 0.5, 2.5, E, N)}')


print('\nМетод парабол через формулу для u для N=', N, 'и эпсилон =', E)
print(f'Для первой функции {parabol(f1, -0.5, 0.5, E, N)}')
print(f'Для второй функции {parabol(f2, 6, 9.9, E, N)}')
print(f'Для третьей функции{parabol(f3, 0, 2 * np.pi, E, N)}')
print(f'Для четвертой функции {parabol(f4, 0, 1, E, N)}')
print(f'Для пятой функции {parabol(f5, 0.5, 2.5, E, N)}')


def brent_method(f, a, b, E, N):
    golden_ratio = (np.sqrt(5) - 1) / 2
    x = w = v = a + golden_ratio * (b - a)
    fx = fw = fv = f(x)
    d = e = b - a
    history = [x]
    for _ in range(N):
        tol_act = E * abs(x) + 1e-10
        mid = (a + b) / 2
        if abs(x - mid) <= 2 * tol_act - (b - a) / 2:
            break
        if abs(e) > tol_act:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2 * (q - r)
            if q > 0:
                p = -p
            else:
                q = -q
            if abs(p) < abs(0.5 * q * e) and p > q * (a - x) and p < q * (b - x):
                e = d
                d = p / q
                u = x + d
                if (u - a) < 2 * tol_act or (b - u) < 2 * tol_act:
                    d = -tol_act if x < mid else tol_act
                else:
                    history.append(u)
                    continue
        if x < mid:
            e = b - x
        else:
            e = a - x
        d = golden_ratio * e
        u = x + d if abs(d) >= tol_act else x + (tol_act if d > 0 else -tol_act)
        fu = f(u)
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
        history.append(x)
        e = d
    return x, f(history[-1])


print('\nМетод Брента для N=', N, 'и эпсилон =', E)
print(f'Для первой функции {brent_method(f1, -0.5, 0.5, E, N)}')
print(f'Для второй функции {brent_method(f2, 6, 9.9, E, N)}')
print(f'Для третьей функции {brent_method(f3, 0, 2*np.pi, E, N)}')
print(f'Для четвертой функции {brent_method(f4, 0, 1, E, N)}')
print(f'Для пятой функции {brent_method(f5, 0.5, 2.5, E, N)}')


print('\nДля придуманных мультимодальных функций: \nДля x**2 + 10 * np.sin(3 * x) + np.cos(5 * x) на отрезке[-5, 5]')
print(golden_section(f6, -5, 5, E, N))
print(parabol(f6, -5, 5, E, N))
print(parabol_ls(f6, -5, 5, E, N))
print(brent_method(f6, -5, 5, E, N))
print('\nДля  x**2 + 5 * np.sin(10 * x) на отрезке[-3, 3]')
print(golden_section(f7, -3, 3, E, N))
print(parabol(f7, -3, 3, E, N))
print(parabol_ls(f7, -3, 3, E, N))
print(brent_method(f7, -3, 3, E, N))


