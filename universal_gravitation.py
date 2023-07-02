import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 物理定数
M_pa = 1.0
G_pa = 1.0

# 時間変数
tmin = 0.0
tmax = 20.0
dt = 0.001
t = tmin

# 微分方程式の右辺  式(2.4)


def f(G, M, x_mat):
    # パラメーター
    _GM = -1.0*G*M
    # 各成分
    x = x_mat[0][0]
    y = x_mat[1][0]
    vx = x_mat[0][1]
    vy = x_mat[1][1]
    # 距離の3乗
    r3 = (x**2 + y**2)**1.5
    # 戻り値
    re = np.array([
        [vx, _GM*x/r3],
        [vy, _GM*y/r3]
    ])
    return re


# 数値解用変数
x_mat = np.zeros((2, 2))

# 初期条件
r0 = 2
# 第一宇宙速度
v0 = (G_pa*M_pa/r0)**0.5   # 式(1.7)
#
# 式(2.12)
x_mat[:, 0] = np.array([
    r0,
    0
])
x_mat[:, 1] = np.array([
    0,
    0.7*v0
])


# ルンゲクッタ変数
beta = [0.0, 1.0, 2.0, 2.0, 1.0]   # 式(2.6)
delta = [0.0, 0.0, dt/2, dt/2, dt]  # 式(2.7)
k_rk = np.zeros((5, 2, 2))

# データカット
sb_dt = 0.1
Nt = 1
count = 1
cat_val = int(sb_dt/dt)
x_lis = [x_mat.copy()]

# 時間積分
while t < tmax:
    sum_x = np.zeros((2, 2))
    for k in [1, 2, 3, 4]:
        k_rk[k, :, :] = f(G_pa, M_pa, x_mat + delta[k]
                          * k_rk[k-1, :, :])  # 式(2.9)
        sum_x += beta[k]*k_rk[k, :, :]  # 式(2.10)の途中計算
    #
    x_mat += (dt/6)*sum_x  # 式(2.10)
    # データカット
    if count % cat_val == 0:
        x_lis.append(x_mat.copy())
        Nt += 1
    #
    count += 1
    t += dt


# データ取得
x_t = np.array(x_lis)[:, 0, 0]
y_t = np.array(x_lis)[:, 1, 0]
vx_t = np.array(x_lis)[:, 0, 1]
vy_t = np.array(x_lis)[:, 1, 1]
xmin, xmax = x_t.min(), x_t.max()
ymin, ymax = y_t.min(), y_t.max()
width = xmax - xmin
hight = ymax - ymin

# 可視化、アニメーション

fig, ax = plt.subplots()

lim_add = 0.5
vec_size = 0.2


def animate_move(i):
    plt.cla()
    plt.xlabel('x')
    plt.xlim(xmin - lim_add*width, xmax + lim_add*width)
    plt.ylim(ymin - lim_add*hight, ymax + lim_add*hight)
    plt.scatter(0, 0, s=300)
    # 距離
    r_val = (x_t[i]**2 + y_t[i]**2)**0.5
    # 動径方向の単位ベクトル
    er = np.array([x_t[i], y_t[i]])/r_val
    # 万有引力ベクトル
    m_pa = 1.0
    f_vec = -1.0*(G_pa*M_pa*m_pa/(r_val**2)) * er.copy()
    plt.quiver(x_t[i], y_t[i], f_vec[0], f_vec[1],
               scale=vec_size**-1, color='red')
    # 速度ベクトル
    plt.quiver(x_t[i], y_t[i], vx_t[i], vy_t[i],
               scale=vec_size**-1, color='blue')
    # 速度の大きさ
    v_val = (vx_t[i]**2 + vy_t[i]**2)**0.5
    # 遠心力ベクトル
    m_pa = 1.0
    f_vec = (m_pa*v_val**2)/r_val * er.copy()
    plt.quiver(x_t[i], y_t[i], f_vec[0], f_vec[1],
               scale=vec_size**-1, color='green')
    # 軌跡
    plt.plot(x_t[0:i], y_t[0:i], '--')
    plt.scatter(x_t[i], y_t[i], s=300)
    plt.grid()


animate = animate_move
kaisu = Nt - 1
anim = animation.FuncAnimation(fig, animate, frames=kaisu, interval=10)
anim.save("animete_1.gif", writer="imagemagick")
plt.show()
