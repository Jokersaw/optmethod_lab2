
import math
import time

import numpy as np
import matplotlib.pyplot as plt

# начало работы программы (время):
start = time.time()

def grad(x, y):
    return 400 * x**3 + (2 - 400 * y) * x - 2, 200 * y - 200 * x**2

# основная функция:
def f(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# стартовая точка:
x_start = 0
y_start = 0
x = x_start
y = y_start

func = f(x, y)
new_func = func

step = 1
eps = 0.001

# счетчик количества итераций цикла:
it = 0

# счетчик количества вызовов функции
count_function_runs = 1
count_gradient_runs = 0
count_hessian_runs = 0

# массивы, содержащие координаты точек, по которым проходим во время цикла
# для дальнейшего построения точек на графике:
X_graph = []
Y_graph = []
Z_graph = []

# matr:
#  | a b |
#  | c d |
def getMatr(x, y):
    a = 1200 * x**2 - 400 * y + 2
    b = -400 * x
    c = -400 * x
    d = 200
    coef = 1 / (a*d - b*c)
    return d*coef, -b*coef, -c*coef, a*coef

# Matr^(-1) * grad
def findDiffVector(x, y):
    grad_x, grad_y = grad(x, y)
    matr_a, matr_b, matr_c, matr_d = getMatr(x, y)
    return matr_a * grad_x + matr_b * grad_y, matr_c * grad_x + matr_d * grad_y

while True:

    X_graph.append(x)
    Y_graph.append(y)
    Z_graph.append(new_func)

    delta_f = new_func - func
    diff_x, diff_y = findDiffVector(x, y)
    count_gradient_runs += 1
    count_hessian_runs += 1
    delta_arg = math.sqrt(step * diff_x * step * diff_x + step * diff_y * step * diff_y)

    if it != 0 and delta_arg < eps:
        break

    func = new_func

    x = x - step * diff_x
    y = y - step * diff_y

    new_func = f(x, y)

    it += 1
    count_function_runs += 1


# конец работы программы (время):
end = time.time()

print(f'function: (1 - x)**2 + 100 * (y - x**2)**2')
print(f'start points: x = {x_start}, y = {y_start}')
print(f'step: {step}, eps: {eps}\n')
print(f'x: {x}, y: {y}\nf: {f(x, y)}, iterations: {it}\nwork time: {(end-start) * 10**3} ms')
print(f'count_function_runs: {count_function_runs}')
print(f'count_gradient_runs: {count_gradient_runs}')
print(f'count_hessian_runs: {count_hessian_runs}\n')


fig = plt.figure(figsize=(10, 10))
fig.set_figheight(5)

# построение точек на графике
ax = fig.add_subplot(111, projection='3d')
plt.plot(X_graph, Y_graph, Z_graph, 'r')
plt.plot(X_graph, Y_graph, Z_graph, 'bo')


# построение поверхности функции:
x = np.arange(-1, 2, 0.6)
y = np.arange(-1, 2, 0.6)
X, Y = np.meshgrid(x, y)
z = np.array(f(np.ravel(X), np.ravel(Y)))
Z = z.reshape(X.shape)

ax.plot_wireframe(X, Y, Z, cmap='viridis', edgecolor='green')
ax.set_title('Surface plot of f(x,y)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# вывод графика поверхности функции и отмеченных точек:
# (для просмотра следующего окна нужно закрыть текущее)
plt.show()

# построение линий уровня функции в окрестности точки минимума:
x = np.arange(-1, 2, 0.6)
y = np.arange(-1, 2, 0.6)
X, Y = np.meshgrid(x, y)
z = np.array(f(np.ravel(X), np.ravel(Y)))
Z = z.reshape(X.shape)

cs = plt.contour(X, Y, Z, levels=50)
plt.clabel(cs)
plt.plot(X_graph, Y_graph, 'r')
plt.plot(X_graph, Y_graph, 'bo')

# вывод линий уровня функции в окрестности точки минимума:
# (для просмотра этого окна нужно закрыть предыдущее)
plt.show()

